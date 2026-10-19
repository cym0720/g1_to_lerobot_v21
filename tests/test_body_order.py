"""Protect numeric body-order hints while consolidating the two legacy copies."""
import numpy as np
import pytest

from packages.body_order import detect_body_order


@pytest.mark.parametrize(
    "tail, expected",
    [
        ([0.22, 0.349], "lift_first"),
        ([0.349, 0.22], "waist_first"),
        ([0.22, 0.22], None),
        ([0.0, 0.349], None),
    ],
)
def test_numeric_hint_and_ambiguity(tail, expected):
    values = np.zeros((3, 20))
    values[:, 18:20] = tail
    before = values.copy()
    assert detect_body_order(values) == expected
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("values", [[], [0.22] * 20, np.zeros((3, 19))])
def test_unusable_shape_is_ambiguous(values):
    assert detect_body_order(values) is None
