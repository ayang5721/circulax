# Un-vmap the circulax device assembly (fix the BSIM4 compile/OOM blowup)

**Status:** implementation plan, scoped to **this circulax repo**. Last updated 2026-08-10.
**Owner action:** implement per §5, validate per §6 — all in-repo.
**Scope:** `circulax/solvers/assembly.py` (device-group assembly) plus two static fields on
`ComponentGroup` in `circulax/compiler.py`. No changes to any BSIM4/device kernel — only how
the device group is *mapped* during assembly.
**Provenance note:** §1–§3 and §7 quote measurements/validation gates (G0/G2/G3/G8/G9/G10,
`validate_pmos_circuit.py`, `demo_circuit.py`) from the **base-model repo** where the BSIM4
kernel and its bit-exactness harness live. Those are cited as *evidence for the diagnosis*;
they are **not** steps to run here. The runnable validation for this change is §6, which is
entirely in-circulax.

---

## 1. The error (measured)

Running a BSIM4 device in a circulax circuit compiles an enormous XLA program:

- A single **forward DC solve** compiles for minutes ("very slow compile" alarms on
  `jit_cond`) and needs several GB; it *fits* (<~9 GB) for one strategy (demo NMOS
  solved to 0.833 V), but stacking homotopy strategies (`gmin`/`source`/`auto`) **crashed
  WSL** (16 GB) — the Gate D run that ran four solves at once.
- A **reverse-mode gradient through the solve OOMs** (>12 GB); it has crashed WSL twice
  (memory `circulax-grad-oom`).
- A too-tight `MemoryHigh` cap turns the compile into a multi-hour livelock (observed:
  3h44m elapsed, 8 min CPU) — throttle, not progress.

This is a **tractability** problem (compile size / memory), not a correctness problem
(see §7).

## 2. Root cause

Circulax evaluates a device *group* with `jax.vmap` over device instances
([`assembly.py:353`](circulax/solvers/assembly.py)). `vmap` batches the port
voltages, so every **voltage-dependent** `lax.cond` in the BSIM kernel (~525 of them) gets
a **batched predicate**. Vectorized execution can't take different branches per lane, so
JAX lowers each `lax.cond` → `lax.select`: **both branches always execute** and are
inlined into one giant straight-line graph. XLA's passes are superlinear in graph size →
the multi-GB, minutes-long compile.

- The **forward stamp** differentiates that graph in **forward mode** (`jvp`) — memory
  streams, so it fits.
- The **backward pass** differentiates it in **reverse mode** (`vjp`) — must tape every
  intermediate of the giant graph at once → OOM.

The safe-math helpers (`_jx_div`/`_jx_safe_log`/`sqrt`) are a *consequence* of both
branches executing (the off-domain branch must stay finite), **not** the cost driver.
The cost driver is the both-branch materialization of the conditionals under `vmap`.

## 3. The un-vmapped reference already in THIS repo (the template)

This repo's base model (`jax/jacobian_jax.py`, the LEAN kernel in `jax/generated/`) is the
existence proof that the same kernel is cheap when **un-vmapped**:

- It evaluates one device at a bias with **scalar predicates**, so each `lax.cond` stays a
  **real branch (taken side only)** — no both-branch blowup.
- Its Jacobian is *16 un-vmapped `jax.jvp`s* over the residual (memory
  `jax-voltage-jacobian-via-ad`) — forward-mode, one direction at a time.

Circulax's `_primal_and_jac_real` ([`assembly.py:223`](circulax/solvers/assembly.py))
is *already exactly this* per device (an inner `vmap` over the 16 tangent unit vectors of a
single device — whose predicate is **not** batched, so it stays a real cond). The only
thing that breaks it is the **outer `vmap` over devices** wrapping it. Remove that outer
`vmap` and each device compiles with base-model economics.

## 4. Goal

Give circulax's DC assembly a **selectable per-group device-map strategy** instead of a
hard-wired `jax.vmap`:

- `"vmap"` (**default, unchanged**): batched predicate → both-branch `select`. Correct and
  fastest for cheap, branch-free kernels (R/L/C, linear stamps) and for large groups of
  identical simple devices.
- `"scan"` (**opt-in, new**): `jax.lax.map`/`lax.scan` over devices → **scalar predicate** →
  real `lax.cond` (taken branch only) → the compile collapses to base-model scale, for
  **both** the forward solve and the reverse gradient. This is what BSIM4-class kernels
  (~525 voltage-dependent conds) need.
- `"chunked"` (**optional hybrid**): `lax.map` over chunks, `vmap` within a small chunk —
  trades a little compile size for parallelism when a heavy kernel has *many* instances.

The strategy is chosen **per `ComponentGroup`** (§5.2.1) so a mixed circuit keeps resistors
vmapped while its transistors scan. Whatever the strategy, **all gradients must remain
exact and numerically equal to the vmap path** (§5.3, §6A).

Why per-group and not per-device: a `ComponentGroup` batches all instances of one component
type through a single compiled kernel, so the branch behaviour is a property of the *kernel*,
not the individual device — every instance in a group takes the same path. The only
meaningful "sub-group" granularity is the chunk size of the `"chunked"` hybrid.

## 5. Where in circulax to edit, and how

### 5.1 Sites (in `circulax/solvers/assembly.py`)
Primary (DC / transient real system — what the PMOS/NMOS circuits use):
- **`assemble_system_real` line ~353** — `jax.vmap(functools.partial(_primal_and_jac_real,
  physics_at_t1))(v_locs, params)` — the residual **and** Jacobian stamp. **Main target.**
- **`assemble_system_real` line ~342** — the `group.combined_func` bypass
  (`jax.vmap(lambda v,p: group.combined_func(v,p,t1))`). Only active if a VA emitter set
  `_has_combined_fn`/`_combined_fn` on the component class. **Verified dead for the
  Python-JAX BSIM4 group in this tree:** nothing in `circulax/` sets those attributes, so
  `combined_func is None` and the live full-stamp path is line 353. (Also confirmed OSDI is
  not taken — the device is a plain `@component` Python-JAX kernel, not an `OsdiComponentGroup`.)
  Left in the dispatch anyway so it composes with the strategy switch if a combined_fn ever appears.
- **`assemble_residual_only_real` line ~477** — `jax.vmap(physics_at_t1)(v, group.params)`
  — residual-only path used by the homotopy rescues (`solve_dc_source`/gmin scans).

Secondary (AC / GC / harmonic-balance — do later if needed):
- `assemble_gc_real` line ~418, `assemble_system_complex` line ~584,
  `assemble_residual_only_complex` line ~654.

Leave `_primal_and_jac_real`'s **inner** tangent-`vmap` (line 232) intact — its predicate
is scalar (single device), so it does not cause the blowup.

### 5.2 How — a single dispatch helper, not scattered `vmap` calls
Rather than hand-editing each `jax.vmap` site, introduce **one** device-map dispatcher and
route every batched device evaluation through it. Each call site keeps its own `body`
(a per-device closure taking a single sliced pytree) and passes `group`'s strategy.

```python
def _map_devices(body, xs, *, strategy="vmap", chunk_size=1):
    """Map `body` over the leading (device) axis of pytree `xs`.

    - "vmap":  jax.vmap  -> batched predicate -> both-branch select (fastest for
               cheap/branch-free kernels; pathological for BSIM-class conds).
    - "scan":  jax.lax.map -> body compiled once, devices sequential, SCALAR predicate
               -> real lax.cond (taken branch only). Differentiable fwd AND reverse.
    - "chunked": lax.map over chunks of `chunk_size`, vmap within each chunk.
    All three return the same pytree of stacked per-device results (up to float
    reassociation), so call sites are agnostic to the choice.
    """
    if strategy == "vmap":
        return jax.vmap(body)(xs)
    if strategy == "scan":
        return jax.lax.map(body, xs)
    if strategy == "chunked":
        # PREFERRED (JAX >= 0.4.31): jax.lax.map has a built-in batch_size that does
        # exactly vmap-within-chunk / scan-across, correctly handling the ragged tail.
        return jax.lax.map(body, xs, batch_size=max(1, chunk_size))
    raise ValueError(f"unknown device-map strategy {strategy!r}")
```

> If your pinned JAX predates `lax.map(..., batch_size=)`, hand-roll it: reshape the leading
> axis `N → (N//c, c)` (falling back to pure scan when `c<=1` or `N % c != 0`), `lax.map`
> `jax.vmap(body)` over the chunks, then flatten the two leading axes back to `N`. Prefer the
> built-in — it handles the non-tiling remainder that the naive reshape drops.

The edit at line 353 then becomes (mechanically the same shape as the current call, but
strategy-driven):

```python
# BEFORE (hard-wired device vmap):
(f_l, q_l), (df_l, dq_l) = jax.vmap(
    functools.partial(_primal_and_jac_real, physics_at_t1))(v_locs, params)

# AFTER (dispatch on the group's strategy; body takes a single sliced (v, p) tuple):
_body = lambda vp: _primal_and_jac_real(physics_at_t1, vp[0], vp[1])
(f_l, q_l), (df_l, dq_l) = _map_devices(
    _body, (v_locs, params),
    strategy=group.assembly_strategy, chunk_size=group.assembly_chunk_size,
)
```

Apply the same one-line dispatch at every batched device site:
- **`assemble_system_real` 353** — full stamp (primary). Body as above.
- **`assemble_residual_only_real` 477** — `_body = lambda vp: physics_at_t1(vp[0], vp[1])`,
  `xs=(v, group.params)`. (Homotopy rescues; must match 353's strategy or the residual and
  Jacobian disagree only at the reassociation floor — fine, but keep them consistent.)
- **`assemble_gc_real` 418** — same body as 353 (AC/GC path), for consistency.
- **`combined_func` bypass 342/408** — `_body = lambda vp: group.combined_func(vp[0], vp[1], t1)`,
  `xs=(v_locs, params)`. **Note:** at 342 the second element must be the **source-scaled**
  `params` (the local variable built via `eqx.tree_at`), *not* `group.params` — same as the
  current code. At 408 (`assemble_gc_real`) it is `group.params` evaluated at `t1=0.0`.
  Dead today (§5.1) but routed through `_map_devices` so it inherits the switch for free.
- Secondary complex sites (584/654) — defer; wrap identically when needed.

Notes / gotchas:
- `params` is a batched eqx pytree (leaves have leading dim `N` = devices); both `vmap` and
  `lax.map` slice axis 0 of every leaf, so `_body` receives a single-device `params`.
  Confirmed all `group.params` leaves are batched on axis 0 (the current `vmap` maps them there).
- `_map_devices` collapses to *exactly* the current code when `strategy="vmap"` — so a group
  that never opts in is byte-for-byte unchanged (see §8).
- Start heavy groups on pure `"scan"` (smallest compile) and only reach for `"chunked"` if
  throughput matters and the group has many instances — measure before tuning `chunk_size`.

### 5.2.1 Where the strategy is chosen (per-group wiring)
Add two static fields to `ComponentGroup` (`circulax/compiler.py`, alongside `combined_func`):

```python
assembly_strategy: str = eqx.field(static=True, default="vmap")
assembly_chunk_size: int = eqx.field(static=True, default=1)
```

Selection precedence at compile time (in `compile_netlist`, high → low):
1. **Explicit user override** — a new **keyword-only, defaulted** arg, e.g.
   `compile_netlist(net, models, *, assembly_strategy=None)` accepting a `{group_name: strategy}`
   dict (or a callable `group_name -> strategy`). Keyword-only-with-default is required:
   `compile_netlist` currently has the signature `(netlist, models_map)` with ~45 call sites
   across `circulax/` and `tests/`; a defaulted kwarg leaves every one of them untouched.
2. **Component-class hint** — the kernel author declares its preference:
   `getattr(comp_cls, "_assembly_strategy", None)` (e.g. BSIM4 sets `_assembly_strategy = "scan"`).
   This is where the BSIM4 opt-in lives by default.
3. **Auto heuristic (optional stretch)** — at compile, trace the kernel once and count
   `cond`/`select` primitives in its jaxpr (or a cheap `getattr(comp_cls, "_is_branchy", False)`
   flag); pick `"scan"` above a threshold. Keep this off by default until validated — a static
   flag (1/2) is more predictable.
4. **Default** — `"vmap"`, preserving today's behaviour for everything unmarked.

Because the choice is baked into the (static) group field at compile time, it is a
compile-time constant inside the jitted assembly — no runtime branching, no extra tracing.

### 5.3 Why gradients stay exact (the important part)
- Each device's Jacobian is still the same forward-mode `jvp` (`_primal_and_jac_real`);
  `lax.map` just sequences the per-device evaluations. Per-device stamps are **identical**
  to the vmapped ones (mode-agnostic; only reassociation-floor float noise, §6).
- `lax.scan`/`lax.map` is differentiable forward **and** reverse; the circuit's
  implicit-diff adjoint composes the per-device contributions exactly as before.
- With scalar predicates, `lax.cond` keeps its clean single-branch **jvp and transpose**
  rules — so the reverse tape is over one path per device, not both. That is precisely why
  the backward pass stops OOMing.

Net: same numbers, ~base-model compile size, gradients preserved.

## 6. Validation (self-contained, in this circulax repo)

Everything here runs against circulax alone — no base-model repo, no `circulax_model/`
scripts. Add the checks below as a new test module (e.g. `tests/test_assembly_strategy.py`)
plus one manual tractability script. Run the heavy items under a hard memory cap
(`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, **no** `MemoryHigh` — that
throttle caused the 3h44m livelock in §1).

**Fixtures.** Reuse an existing device netlist fixture (see `tests/conftest.py` and
`tests/test_dc.py` for the LRC/transistor netlists already in-repo). If a BSIM4 group is not
yet wired into a test fixture, any multi-instance nonlinear group (e.g. several `Diode`s or
`MOSFET`s from `circulax/components/electronic.py`) exercises the same `_primal_and_jac_real`
device-map path and is sufficient for the equivalence checks A/D; use the heaviest available
kernel for the tractability check B.

**A. Strategy-equivalence — every strategy computes the same stamp:**
- Compile one nonlinear group and call `assemble_system_real` at a fixed `y_guess` with the
  group forced to `"vmap"`, `"scan"`, and `"chunked"` (chunk sizes 1, 2, and a size that does
  not evenly divide `N`, to exercise the ragged tail). Assert `total_f`, `total_q`, and
  `jac_vals` agree with `jnp.allclose(rtol=1e-9, atol=1e-15)`.
  - This tolerance is a deliberate **float64 numerical-identity bound**: the three paths run
    identical arithmetic and differ only by reduction reassociation, so they must agree to a
    few ULP — far tighter than the solver's own `1e-6` convergence tolerance
    (`FixedPointIteration` default in `circulax/solvers/linear.py`). It is chosen here, not
    inherited from any external harness.
- Pin the baseline: capture `assemble_system_real` output with `strategy="vmap"` and assert
  it is **bit-for-bit** equal (`jnp.array_equal`) to the output of the pre-change code path
  — i.e. `_map_devices(..., "vmap")` must lower to the same call as today's `jax.vmap`.
  (`git stash` the change, record the arrays, compare — or assert against a checked-in golden.)
- Repeat for `assemble_residual_only_real` (residual-only) and `assemble_gc_real` (G/C split).

**B. Tractability — the actual win (manual script, capped):**
- Build the heaviest available nonlinear circuit as a `Circuit` (see `circulax/circuit.py`)
  and run one DC solve (`solve_dc`) with the group on `"vmap"` vs `"scan"`. Record wall-clock
  **compile** time and **peak RSS** (`/usr/bin/time -v`, or `resource.getrusage`) for each.
  Expect `"scan"` to drop from minutes / multi-GB to seconds / ~1–2 GB for a BSIM-class kernel.
- Under the 8 GB cap, the `"vmap"` path may OOM/livelock for the heavy kernel (reproducing §1)
  while `"scan"` completes — that contrast *is* the result. Record both.

**C. Gradient — the thing that OOM'd:**
- Take `jax.grad` of a scalar of the DC solution (e.g. `lambda p: solve_dc(...)[out_node]`)
  w.r.t. a device parameter, with the group on `"scan"`. Assert it **completes under the cap**
  and returns a finite value. On `"vmap"` this is the reverse-mode OOM from §1; on `"scan"`
  the per-device tape is one real branch, so it fits. (Cross-check the gradient against a
  finite-difference of the forward `solve_dc` to a loose tol, e.g. `rtol=1e-3`, to confirm
  correctness, not just finiteness.)

**D. Regression — nothing else moves:**
- Run the existing suite: `pixi run pytest_run`. With every group defaulting to `"vmap"`
  (§5.2.1 precedence #4), all current tests must pass unchanged — the switch is inert until
  a group opts in. Then flip the equivalence-test group to `"scan"` and confirm the DC/AC
  results in the reused fixtures still match their existing assertions.

## 7. Is this fix enough? Is the model good, or also wrong?

**The device model is good — validated, not suspect.** Both polarities match the base
model: NMOS is **bit-identical** (G2 worst rel 0.0), PMOS is at the float64 reassociation
floor (G8 across all 4 pfet bins, 0 crashes; standalone rel ~1.6e-16). The device current
is correct even at extreme bias (it matches base at the −6.24 V spurious point:
1247.86 µA). NaN-safety (G0) and taken-path bitwise-vs-lean (G10) hold. So the physics is
right; nothing points to a wrong model.

**There are two INDEPENDENT blockers — un-vmap fixes only the first:**

1. **Compile/OOM (this doc).** Tractability. Un-vmap fixes it → forward solves and
   gradients become cheap and runnable.
2. **PMOS circuit solver convergence** (the −6.24 V spurious root). Phase 0
   (`circulax_fix2_tangent_safe_ops.md`, memory `pmos-circuit-spurious-root`) proved this
   is **not** a model bug and **not** a safe-op artifact: it is an *inherent* root of the
   BSIM equations extrapolated to Vsd≈8 V that Newton/homotopy can land on from a cold
   start. It is a **solver-robustness** issue, fixed by warm-start / limiting (Fix 1),
   orthogonal to un-vmap.

**So:**
- **NMOS circuits:** un-vmap is very likely **sufficient** — the model is correct, NMOS
  already converges (demo solves), and un-vmap just makes it tractable + enables gradients.
- **PMOS circuits:** un-vmap is **necessary but likely not sufficient** — it makes the
  solve runnable (and finally lets you *test* convergence, which we never could before it
  crashed), but the spurious-root problem is expected to persist and will additionally need
  **Fix 1** (targeted warm-start out≈0.2 V/src=1.8 V, or solver-side limiting). Ground truth
  for that circuit is Vout* = 0.2005 V (`validate_pmos_circuit.py`).

Bottom line: the model isn't wrong — it's held back by (1) this compile/memory error, and
(for PMOS) (2) a separate solver-convergence issue. Un-vmap clears (1) and unblocks testing
(2); it does not by itself guarantee PMOS convergence.

## 8. Rollback / risk

Risk is very low because the change is **opt-in and default-off**:

- The default `assembly_strategy="vmap"` makes `_map_devices` collapse to the exact current
  `jax.vmap(body)(xs)` call — every group that doesn't opt in is behaviourally identical to
  today. Rollback of the *behaviour* is just "don't set the flag"; rollback of the *code* is
  reverting the `_map_devices` helper + the field additions.
- `"scan"` is a pure scheduling change over the same per-device stamp — identical numerics
  (float reassociation floor), gradients preserved forward and reverse (§5.3).
- The only real knob to tune is `"chunked"` throughput for large heavy groups; if `"scan"`
  is too slow for a big transistor array, raise `assembly_chunk_size` and re-measure. If a
  strategy ever misbehaves, flip that one group back to `"vmap"` without touching others.

Because the switch is per-group, a regression is contained to the group that opted in — the
rest of the circuit is unaffected.
