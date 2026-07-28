# Hierarchical Subcircuit Composition

## Overview

| Property | Value |
|----------|-------|
| Description | Define reusable subcircuits from compositions of primitive components |
| SPICE Equivalent | `.subckt` / `.ends` |
| Data Model | SAX `RecursiveNetlist` (`dict[str, Netlist]`) |
| Compilation Strategy | Netlist-level flattening (pre-compilation) |
| Status | V1 specified, not yet implemented |

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
| `circulax/netlist.py` | Add `flatten_recursive_netlist()` |
| `circulax/circuit.py` | Extend `compile_circuit` to detect RecursiveNetlist and Circuit-in-models_map; store source data on `Circuit`; add `ports` property |
| `circulax/__init__.py` | Export `flatten_recursive_netlist` |
| `tests/test_subcircuit.py` | New test file |

#### `flatten_recursive_netlist` signature

```python
def flatten_recursive_netlist(
    recnet: dict[str, dict],
    sep: str = "~",
) -> dict:
    """Flatten a RecursiveNetlist into a single SAX-format netlist.

    Extends SAX's flatten_netlist to handle circulax connection
    extensions (tuple targets, nets lists).
    """
```

#### `Circuit.__init__` additions

```python
self._source_netlist = _source_netlist   # original netlist (for reuse as subcircuit)
self._source_models = _source_models     # leaf models used (for reuse as subcircuit)
```

#### `Circuit.ports` property

```python
@property
def ports(self) -> tuple[str, ...]:
    """External port names from the source netlist."""
    return tuple((self._source_netlist or {}).get("ports", {}).keys())
```

#### RecursiveNetlist detection

A `dict` is a `RecursiveNetlist` if:
- It does NOT have an `"instances"` key (which would make it a flat Netlist).
- At least one value is a `dict` with an `"instances"` key.

### Test Matrix

| Test | Scenario | Verification |
|------|----------|-------------|
| `test_parallel_resistors_recnet` | Two R in parallel via RecursiveNetlist | DC solve: I = V / R_parallel |
| `test_multiple_subcircuit_instances` | Two instances of same subcircuit | Correct prefixing: `RP1~R1`, `RP2~R1` |
| `test_nested_subcircuits` | 3-level nesting | Instance name: `outer~inner~R1` |
| `test_circuit_in_models_map` | Compiled Circuit passed in models_map | DC solve matches standalone |
| `test_stateful_subcircuit` | Subcircuit with Inductor (state `i_L`) | Transient analysis works |
| `test_ground_sharing` | Subcircuit with internal GND | GND not prefixed, shared with parent |
| `test_tuple_target_connections` | Circulax tuple extension in subcircuit | Correct flattening |
| `test_subcircuit_ports_property` | `circuit.ports` on compiled circuit | Returns declared port names |

### Verification

```bash
cd /path/to/circulax
pytest tests/test_subcircuit.py -v        # new tests
pytest tests/ -v                          # no regressions
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

**Trivial once V1 lands**: Return `self._source_netlist`. Useful for serialization,
inspection, and programmatic composition.
