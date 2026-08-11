"""Tests for the per-group device-map strategy in DC assembly.

Validates §6 of ``circulax_unvmap_assembly.md``: every strategy ("vmap",
"scan", "chunked") computes the same residual/Jacobian stamp (A), the switch
is inert by default so the existing suite is unaffected (D), and a gradient
through the assembly stays exact and finite on "scan" (C).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.solvers.assembly import (
    _map_devices,
    assemble_gc_real,
    assemble_residual_only_real,
    assemble_system_real,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@pytest.fixture
def diode_netlist() -> tuple[dict, dict]:
    """Three parallel diodes + a series resistor from a voltage source.

    Three instances land in a single ``diode`` group so the device-map path
    (``_primal_and_jac_real`` over N devices) is exercised, with N=3 giving a
    ragged tail for chunk sizes that don't divide evenly.
    """
    from circulax.components.electronic import Diode, Resistor, VoltageSource

    models_map = {
        "diode": Diode,
        "resistor": Resistor,
        "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "source_voltage", "settings": {"V": 0.7}},
            "R1": {"component": "resistor", "settings": {"R": 100.0}},
            "D1": {"component": "diode", "settings": {"Is": 1e-12}},
            "D2": {"component": "diode", "settings": {"Is": 2e-12}},
            "D3": {"component": "diode", "settings": {"Is": 3e-12}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "D1,p2", "D2,p2", "D3,p2"),
            "V1,p2": "R1,p1",
            "R1,p2": ("D1,p1", "D2,p1", "D3,p1"),
        },
    }
    return net_dict, models_map


def _compile(net_dict: dict, models_map: dict, strategy: str | None) -> tuple[dict, int]:
    override = None if strategy is None else {"diode": strategy}
    groups, sys_size, _ = compile_netlist(net_dict, models_map, assembly_strategy=override)
    return groups, sys_size


def _compile_chunked(net_dict: dict, models_map: dict, chunk_size: int) -> tuple[dict, int]:
    groups, sys_size, _ = compile_netlist(net_dict, models_map, assembly_strategy={"diode": ("chunked", chunk_size)})
    return groups, sys_size


def test_map_devices_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown device-map strategy"):
        _map_devices(lambda x: x, jnp.arange(3.0), strategy="bogus")


def test_default_strategy_is_vmap(diode_netlist: tuple[dict, dict]) -> None:
    groups, _ = _compile(*diode_netlist, strategy=None)
    assert groups["diode"].assembly_strategy == "vmap"
    assert groups["diode"].assembly_chunk_size == 1


def test_strategy_equivalence_full_stamp(diode_netlist: tuple[dict, dict]) -> None:
    net_dict, models_map = diode_netlist
    groups_v, sys_size = _compile(net_dict, models_map, "vmap")
    groups_s, _ = _compile(net_dict, models_map, "scan")

    y = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (sys_size,))

    def run(groups: dict) -> tuple:
        return assemble_system_real(y, groups, t1=0.0, dt=1e-9)

    f_v, q_v, j_v = run(groups_v)
    f_s, q_s, j_s = run(groups_s)

    for a, b in ((f_v, f_s), (q_v, q_s), (j_v, j_s)):
        assert jnp.allclose(a, b, rtol=1e-9, atol=1e-15)

    # chunked with sizes 1, 2 (ragged tail for N=3), 3 (exact)
    for c in (1, 2, 3):
        gc, _ = _compile_chunked(net_dict, models_map, c)
        assert gc["diode"].assembly_strategy == "chunked"
        assert gc["diode"].assembly_chunk_size == c
        f_c, q_c, j_c = run(gc)
        for a, b in ((f_v, f_c), (q_v, q_c), (j_v, j_c)):
            assert jnp.allclose(a, b, rtol=1e-9, atol=1e-15), f"chunk={c}"


def test_strategy_equivalence_residual_only(diode_netlist: tuple[dict, dict]) -> None:
    net_dict, models_map = diode_netlist
    groups_v, sys_size = _compile(net_dict, models_map, "vmap")
    groups_s, _ = _compile(net_dict, models_map, "scan")

    y = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (sys_size,))

    f_v, q_v = assemble_residual_only_real(y, groups_v, t1=0.0, dt=1e-9)
    f_s, q_s = assemble_residual_only_real(y, groups_s, t1=0.0, dt=1e-9)
    assert jnp.allclose(f_v, f_s, rtol=1e-9, atol=1e-15)
    assert jnp.allclose(q_v, q_s, rtol=1e-9, atol=1e-15)


def test_strategy_equivalence_gc(diode_netlist: tuple[dict, dict]) -> None:
    net_dict, models_map = diode_netlist
    groups_v, sys_size = _compile(net_dict, models_map, "vmap")
    groups_s, _ = _compile(net_dict, models_map, "scan")

    y = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (sys_size,))

    g_v, c_v = assemble_gc_real(y, groups_v)
    g_s, c_s = assemble_gc_real(y, groups_s)
    assert jnp.allclose(g_v, g_s, rtol=1e-9, atol=1e-15)
    assert jnp.allclose(c_v, c_s, rtol=1e-9, atol=1e-15)


def test_component_class_hint(diode_netlist: tuple[dict, dict]) -> None:
    """A class-level ``_assembly_strategy`` hint is honoured without an override."""
    net_dict, models_map = diode_netlist
    diode_cls = models_map["diode"]
    diode_cls._assembly_strategy = "scan"  # noqa: SLF001
    try:
        groups, _, _ = compile_netlist(net_dict, models_map)
        assert groups["diode"].assembly_strategy == "scan"
    finally:
        del diode_cls._assembly_strategy  # noqa: SLF001


def test_override_beats_class_hint(diode_netlist: tuple[dict, dict]) -> None:
    net_dict, models_map = diode_netlist
    diode_cls = models_map["diode"]
    diode_cls._assembly_strategy = "scan"  # noqa: SLF001
    try:
        groups, _, _ = compile_netlist(net_dict, models_map, assembly_strategy={"diode": "vmap"})
        assert groups["diode"].assembly_strategy == "vmap"
    finally:
        del diode_cls._assembly_strategy  # noqa: SLF001


def test_gradient_through_scan_matches_vmap(diode_netlist: tuple[dict, dict]) -> None:
    """Gradient of a residual scalar w.r.t. a device param agrees across strategies."""
    net_dict, models_map = diode_netlist
    groups_v, sys_size = _compile(net_dict, models_map, "vmap")
    groups_s, _ = _compile(net_dict, models_map, "scan")

    y = 0.1 * jax.random.normal(jax.random.PRNGKey(3), (sys_size,))

    def scalar(groups: dict, scale: float) -> jax.Array:
        g = groups["diode"]
        params = eqx.tree_at(lambda p: p.Is, g.params, g.params.Is * scale)
        g2 = eqx.tree_at(lambda x: x.params, g, params)
        gg = dict(groups)
        gg["diode"] = g2
        f, _, _ = assemble_system_real(y, gg, t1=0.0, dt=1e-9)
        return jnp.sum(f**2)

    gv = jax.grad(lambda s: scalar(groups_v, s))(1.0)
    gs = jax.grad(lambda s: scalar(groups_s, s))(1.0)
    assert jnp.isfinite(gs)
    assert jnp.allclose(gv, gs, rtol=1e-9, atol=1e-15)
