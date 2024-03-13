import pytest
import main.py


def test_pauli_expectation():
    assert main.expectation_value()
