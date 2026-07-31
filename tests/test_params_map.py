"""Tests for params_map parameter renaming in compile_netlist / compile_circuit."""

import jax
import jax.numpy as jnp

from circulax.circuit import compile_circuit
from circulax.compiler import compile_netlist
from circulax.components.electronic import Capacitor, Resistor, VoltageSource

jax.config.update("jax_enable_x64", True)


def _make_rc_netlist(*, r_setting_name="R", r_value=100.0):
    """Build a simple V-R-C netlist where the resistor setting key is configurable."""
    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "source_voltage", "settings": {"V": 5.0}},
            "R1": {"component": "resistor", "settings": {r_setting_name: r_value}},
            "C1": {"component": "capacitor", "settings": {"C": 1e-12}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "C1,p2"),
            "V1,p2": "R1,p1",
            "R1,p2": "C1,p1",
        },
    }
    return net_dict, models_map


def test_params_map_renames_setting():
    """params_map renames a netlist setting key to the model field name."""
    net_dict, models_map = _make_rc_netlist(r_setting_name="resistance", r_value=200.0)
    params_map = {"resistor": {"resistance": "R"}}
    groups, _sys_size, _port_map = compile_netlist(net_dict, models_map, params_map=params_map)
    assert "resistor" in groups
    assert float(groups["resistor"].params.R[0]) == 200.0


def test_params_map_none_is_noop():
    """Passing params_map=None (the default) doesn't change behavior."""
    net_dict, models_map = _make_rc_netlist(r_setting_name="R", r_value=100.0)
    groups, _, _ = compile_netlist(net_dict, models_map, params_map=None)
    assert float(groups["resistor"].params.R[0]) == 100.0


def test_params_map_unmapped_keys_pass_through():
    """Settings not in the rename dict pass through unchanged."""
    net_dict, models_map = _make_rc_netlist(r_setting_name="R", r_value=50.0)
    params_map = {"resistor": {"something_else": "other_field"}}
    groups, _, _ = compile_netlist(net_dict, models_map, params_map=params_map)
    assert float(groups["resistor"].params.R[0]) == 50.0


def test_params_map_multiple_renames():
    """Multiple keys can be renamed in a single component."""
    models_map = {
        "source_voltage": VoltageSource,
        "resistor": Resistor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "source_voltage", "settings": {"voltage": 3.3, "rise_delay": 0.1e-8}},
            "R1": {"component": "resistor", "settings": {"R": 10.0}},
        },
        "connections": {
            "GND,p1": "V1,p1",
            "V1,p2": "R1,p1",
            "R1,p2": "GND,p1",
        },
    }
    params_map = {"source_voltage": {"voltage": "V", "rise_delay": "delay"}}
    groups, _, _ = compile_netlist(net_dict, models_map, params_map=params_map)
    assert float(groups["source_voltage"].params.V[0]) == 3.3
    assert float(groups["source_voltage"].params.delay[0]) == 0.1e-8


def test_params_map_through_compile_circuit():
    """params_map threads through compile_circuit and produces a working Circuit."""
    net_dict, models_map = _make_rc_netlist(r_setting_name="resistance", r_value=100.0)
    params_map = {"resistor": {"resistance": "R"}}
    circuit = compile_circuit(net_dict, models_map, params_map=params_map)
    y = circuit.dc()
    assert y.shape[0] > 0
    assert jnp.isfinite(y).all()


def test_params_map_dc_matches_direct():
    """DC solve with params_map matches the same circuit compiled without mapping."""
    r_value = 47.0
    net_direct, models = _make_rc_netlist(r_setting_name="R", r_value=r_value)
    net_mapped, _ = _make_rc_netlist(r_setting_name="resistance", r_value=r_value)
    params_map = {"resistor": {"resistance": "R"}}

    circuit_direct = compile_circuit(net_direct, models)
    circuit_mapped = compile_circuit(net_mapped, models, params_map=params_map)

    y_direct = circuit_direct.dc()
    y_mapped = circuit_mapped.dc()
    assert jnp.allclose(y_direct, y_mapped, atol=1e-10)
