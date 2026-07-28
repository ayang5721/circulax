# Hierarchical Subcircuit Composition

## Overview

| Property | Value |
|----------|-------|
| Description | Define reusable subcircuits from compositions of primitive components |
| SPICE Equivalent | `.subckt` / `.ends` |
| Data Model | SAX `RecursiveNetlist` (`dict[str, Netlist]`) |
| Compilation Strategy | Netlist-level flattening (pre-compilation) |
| Status | V1 implemented ([PR #39](https://github.com/gdsfactory/circulax/pull/39)) |

## Problem Statement

Circulax currently requires every instance in a netlist to be a leaf `CircuitComponent`.
There is no mechanism to define a component as a composition of other components — for
example, two resistors in parallel, an H-bridge, or a differential pair with load resistors.

SAX already has the hierarchy data model (`RecursiveNetlist`) and a `flatten_netlist`
utility. Circulax should reuse the `RecursiveNetlist` format rather than introducing a new
class. The `Circuit` class should be extended to support reuse as a subcircuit.

---

## V1 Specification

### Scope

| Feature | Included |
|---------|----------|
| `compile_circuit` accepts `RecursiveNetlist` | Yes |
| `Circuit` object usable in `models_map` | Yes |
| `Circuit` stores source netlist for reuse | Yes |
| `Circuit.ports` property | Yes |
| Circulax-native flattener (handles tuple targets) | Yes |
| Recursive nesting (subcircuit within subcircuit) | Yes |
| Parameterized subcircuits (settings propagation) | No (V2) |
| kfnetlist-native hierarchy | No (V2) |
| Bottom-up composition (no flattening) | No (V2) |

### API

#### RecursiveNetlist input

A `RecursiveNetlist` is a `dict[str, Netlist]` where the first key is the top-level
circuit and remaining keys define subcircuits. An instance whose `component` string
matches a key in the dict is treated as a subcircuit reference.

```python
from circulax import compile_circuit
from circulax.components.electronic import Resistor, VoltageSource

recnet = {
    "top": {
        "instances": {
            "RP1": {"component": "parallel_R"},
            "V1": {"component": "VDC", "settings": {"V": 1.0}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "V1,p2": "RP1,p1",
            "GND,p1": ("V1,p1", "RP1,p2"),
        },
    },
    "parallel_R": {
        "instances": {
            "R1": {"component": "Resistor", "settings": {"R": 100.0}},
            "R2": {"component": "Resistor", "settings": {"R": 100.0}},
        },
        "connections": {"R1,p1": "R2,p1", "R1,p2": "R2,p2"},
        "ports": {"p1": "R1,p1", "p2": "R1,p2"},
    },
}

models = {"Resistor": Resistor, "VDC": VoltageSource, "ground": lambda: 0}
circuit = compile_circuit(recnet, models)
```

The `models_map` contains only leaf models. Subcircuit component types are resolved from
the `RecursiveNetlist` keys — they must not appear in `models_map`.

#### Circuit-in-models_map

A compiled `Circuit` can be passed as a value in another circuit's `models_map`.
`compile_circuit` detects `Circuit` objects, extracts their stored source netlists, builds
a `RecursiveNetlist`, and flattens before compilation.

```python
sub = compile_circuit(sub_netlist, sub_models)

parent = compile_circuit(
    parent_netlist,
    {**parent_models, "my_sub": sub},
)
```

The subcircuit's `ports` field in its source netlist defines the external interface.

#### Circuit.ports property

```python
circuit = compile_circuit(netlist_with_ports, models)
circuit.ports  # → ("p1", "p2")
```

Returns the external port names from the circuit's source netlist. Empty tuple if no ports
were declared.

### Flattening Algorithm

| Property | Value |
|----------|-------|
| Instance separator | `~` (configurable) |
| Naming convention | `parent_instance~child_instance` |
| Depth | Unlimited (recursive) |
| Ground handling | Global — `GND` instances are never prefixed |

#### Steps

1. Take the first key in the `RecursiveNetlist` as the top-level circuit. Deep-copy it.
2. For each instance whose `component` matches a key in the dict:
   a. Recursively flatten the child netlist first.
   b. Prefix all child instance names: `"R1"` → `"RP1~R1"`.
   c. Inline child `connections` with prefixed instance references.
   d. Build port mapping from child's `ports` field: `"RP1,p1"` → `"RP1~R1,p1"`.
   e. Rewrite all parent `connections` referencing the subcircuit through the mapping.
   f. Rewrite parent `ports` referencing the subcircuit through the mapping.
   g. Remove the subcircuit instance from parent `instances`.
3. Repeat until no subcircuit instances remain.

#### Connection format support

| Format | Example | Handled |
|--------|---------|---------|
| SAX 1:1 | `{"R1,p1": "R2,p1"}` | Yes |
| Circulax tuple | `{"GND,p1": ("V1,p1", "R1,p2")}` | Yes |
| `nets` list | `[{"p1": "R1,p1", "p2": "R2,p1"}]` | Yes |

#### Ground handling

`GND` instances inside subcircuits represent the same global ground node. During
flattening:
- `GND` instances are NOT prefixed (no `"RP1~GND"`).
- Internal `"GND,p1"` references are preserved as-is.
- If the parent netlist does not have a `GND` instance, one is added.

### Implementation Files

| File | Change |
|------|--------|
| `circulax/netlist.py` | Added `flatten_recursive_netlist()`, `_is_recursive_netlist()`, `_flatten_into()`, `_inline_subcircuit()`, `_rewrite_ref()`, `_rewrite_connection_value()`, `_prefix_ref()` |
| `circulax/circuit.py` | Extended `compile_circuit` to detect RecursiveNetlist and Circuit-in-models_map; added `_embed_circuit_subcircuits()` helper; stored source data on `Circuit`; added `ports`, `source_netlist`, `source_models` properties |
| `circulax/__init__.py` | Exported `flatten_recursive_netlist` |
| `tests/test_subcircuit.py` | New test file (22 tests across 4 classes) |

#### Key functions

**`flatten_recursive_netlist(recnet, sep="~")`** — Public entry point. Deep-copies the
top-level netlist and delegates to `_flatten_into`.

**`_flatten_into(recnet, net, sep)`** — Iterates over instances, identifies subcircuit
references (component matches a key in `recnet`), recursively flattens children, then
calls `_inline_subcircuit` to merge each child into the parent.

**`_inline_subcircuit(net, inst_name, child, sep)`** — Merges a flattened child netlist
into the parent: prefixes child instances, builds port mapping, rewrites parent
connections through the mapping, inlines child connections, removes the subcircuit
instance.

**`_embed_circuit_subcircuits(net_dict, models_map, circuit_models)`** — Extracts
`Circuit` objects from `models_map`, builds a `RecursiveNetlist` from their stored source
netlists, and merges their leaf models into `models_map`.

#### `Circuit` additions

```python
# Properties (public API)
circuit.ports           # → tuple[str, ...] from source netlist
circuit.source_netlist  # → dict | None (stored for reuse as subcircuit)
circuit.source_models   # → dict | None (leaf models for reuse)
```

#### RecursiveNetlist detection (`_is_recursive_netlist`)

A `dict` is a `RecursiveNetlist` if:
- It does NOT have an `"instances"` key (which would make it a flat Netlist).
- At least one value is a `dict` with an `"instances"` key.

### Test Matrix

All tests in `tests/test_subcircuit.py` (22 tests, 4 classes):

#### `TestIsRecursiveNetlist` — Detection logic

| Test | Scenario |
|------|----------|
| `test_flat_netlist` | Dict with `instances` key → not recursive |
| `test_recursive_netlist` | Dict of named netlists → recursive |
| `test_empty_dict` | Empty dict → not recursive |

#### `TestFlattenRecursiveNetlist` — Flattening algorithm

| Test | Scenario | Verification |
|------|----------|-------------|
| `test_no_subcircuits` | Flat netlist passed to flattener | Returned unchanged |
| `test_basic_flattening` | Single subcircuit level | Child instances prefixed with `~` |
| `test_connection_rewriting` | Parent connections reference subcircuit ports | Rewritten to internal refs |
| `test_tuple_targets` | Circulax tuple-target connections | Tuples rewritten correctly |
| `test_nested_subcircuits` | 3-level nesting | Instance: `outer~inner~R1` |
| `test_6_level_deep_nesting` | 6-level deep hierarchy | Instance: `a~x~x~x~x~x~x` with component `Resistor` |
| `test_6_level_deep_nesting_dc_solve` | 6-level deep with DC solve | Correct voltage at deepest node |
| `test_ground_not_prefixed` | Subcircuit with internal GND | `GND` not prefixed, global |
| `test_multiple_instances_of_same_subcircuit` | Two instances of same subcircuit | `RP1~R1`, `RP2~R1` both present |
| `test_parent_ports_rewritten` | Parent ports reference subcircuit | Ports rewritten through mapping |
| `test_nets_list_rewriting` | `nets` list format connections | Refs prefixed in net dicts |

#### `TestCompileCircuitRecursive` — End-to-end compilation

| Test | Scenario | Verification |
|------|----------|-------------|
| `test_parallel_resistors_dc` | Two R in parallel via RecursiveNetlist | DC solve: I = V / R_parallel |
| `test_subcircuit_with_state_transient` | Subcircuit with Inductor (stateful) | Transient analysis works |
| `test_ports_property` | `circuit.ports` on compiled circuit | Returns declared port names |
| `test_ports_property_empty` | Circuit without ports | Returns empty tuple |

#### `TestCircuitInModelsMap` — Circuit-as-subcircuit composition

| Test | Scenario | Verification |
|------|----------|-------------|
| `test_circuit_as_subcircuit` | Compiled Circuit in parent's models_map | DC solve matches expected |
| `test_circuit_without_source_raises` | Circuit with no stored source netlist | Raises `ValueError` |
| `test_model_name_collision_same_object` | Same leaf model under same key | No error (deduped) |
| `test_model_name_collision_different_object_raises` | Different models under same key | Raises `ValueError` |

### Verification

All 22 subcircuit tests pass. Full test suite (243 existing + 22 new) passes with no
regressions. Linting clean under ruff.

```bash
pytest tests/test_subcircuit.py -v        # 22 passed
pytest tests/ -v                          # 265 passed
ruff check circulax/ tests/
ruff format circulax/ tests/
```

---

## V2 Roadmap

### Parameterized subcircuits

**Problem**: V1 subcircuit settings are fixed at definition time. There is no way to
override internal instance settings from the parent — e.g., using the same subcircuit
topology with different resistance values in different instances.

**Gap from V1**: `flatten_recursive_netlist` does not propagate parent instance `settings`
to subcircuit internals. The flattening step blindly copies internal settings as-is.

**Proposed approach**: Add a `params` field to subcircuit netlists that declares exposed
parameters with defaults. Internal instance settings can reference these parameters by
name. During flattening, the parent instance's `settings` override the subcircuit's
`params` defaults, and the resolved values are substituted into internal instance settings.

```python
recnet = {
    "top": {
        "instances": {
            "RP1": {"component": "param_R", "settings": {"R_val": 200.0}},
            "RP2": {"component": "param_R"},  # uses default R_val=100
        },
        ...
    },
    "param_R": {
        "instances": {
            "R1": {"component": "Resistor", "settings": {"R": "$R_val"}},
            "R2": {"component": "Resistor", "settings": {"R": "$R_val"}},
        },
        "params": {"R_val": 100.0},
        "ports": {"p1": "R1,p1", "p2": "R1,p2"},
        "connections": {"R1,p1": "R2,p1", "R1,p2": "R2,p2"},
    },
}
```

**Implementation notes**:
- String-valued settings prefixed with `$` are parameter references.
- During flattening: resolve `$R_val` → parent's `settings["R_val"]` or subcircuit default.
- Affects `flatten_recursive_netlist` only — no compiler/solver changes.
- Validation: error if a referenced parameter is not declared in `params` and not provided
  by the parent.

### kfnetlist-native hierarchy

**Problem**: V1 operates on SAX-format dicts. `kfnetlist.Netlist` inputs are round-tripped
through `.to_dict()` when subcircuits are involved.

**Gap from V1**: `flatten_recursive_netlist` does not accept `kfnetlist.Netlist` objects.
The `kfnetlist` package has no `RecursiveNetlist` type or flatten method.

**Two paths**:

1. **Upstream in kfnetlist** (Rust): Add a `RecursiveNetlist` type and flatten method to
   kfnetlist. This integrates directly with `kfnetlist.extract()` which returns
   `dict[str, Netlist]` — structurally identical to a recursive netlist. Avoids all
   round-trips.

2. **Python-side adapter**: Convert `dict[str, kfnl.Netlist]` to SAX-format
   `dict[str, dict]`, flatten with `flatten_recursive_netlist`, convert back. Double
   round-trip but no Rust changes.

### Bottom-up composition (SAX `circuit()` style)

**Problem**: V1 always flattens — the solver sees every primitive instance globally. For
large subcircuits used many times, this duplicates work. SAX's `circuit()` instead builds
each subcircuit into a model function bottom-up through a dependency DAG, composing
S-parameter models without flattening.

**Gap from V1**: Circulax cannot evaluate a subcircuit as a black-box `(f, q)` function
inside the parent's assembly loop, because:
- Internal state variables must be part of the global system vector.
- The Jacobian sparsity pattern must include cross-terms between subcircuit internals and
  parent connection nodes.
- Nested Newton solves (true black-box) would be expensive and break JAX tracing.

**Feasibility**: Low priority. Flattening is exact (preserves nonlinearity and state) and
the performance cost is manageable for realistic circuit sizes. Bottom-up composition is
primarily valuable for linear subcircuits (S-parameter domain), which SAX already handles.

### Circuit.to_netlist() method

**Trivial now that V1 has landed**: Return `self.source_netlist`. The public
`source_netlist` property already exists — `to_netlist()` would be a named alias.
Useful for serialization, inspection, and programmatic composition.
