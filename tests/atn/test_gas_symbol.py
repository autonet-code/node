"""The gas symbol follows the chain when the registry names none."""
from __future__ import annotations

from atn.config import RPBConfig, gas_symbol_for_chain


def test_known_chains():
    assert gas_symbol_for_chain(31337) == "ETH"
    assert gas_symbol_for_chain(127823) == "XTZ"
    assert gas_symbol_for_chain(42793) == "XTZ"
    assert gas_symbol_for_chain(137) == "POL"
    assert gas_symbol_for_chain(42161) == "ETH"


def test_unknown_chain_is_gas():
    assert gas_symbol_for_chain(999) == "gas"
    assert gas_symbol_for_chain(0) == "gas"
    assert gas_symbol_for_chain(None) == "gas"


def test_config_default_is_derived_not_xtz():
    assert RPBConfig().gas_symbol == ""
    assert RPBConfig().gas_decimals == 18
