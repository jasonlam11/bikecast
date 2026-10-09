import math

import pytest

from bikecast.evaluation.metrics import coverage, mae, rmse, skill, wape


def test_hand_computed_values():
    y, yhat = [0, 2, 4, 10], [1, 2, 2, 6]
    # errors: 1, 0, 2, 4
    assert mae(y, yhat) == pytest.approx(7 / 4)
    assert rmse(y, yhat) == pytest.approx(math.sqrt((1 + 0 + 4 + 16) / 4))
    assert wape(y, yhat) == pytest.approx(7 / 16)


def test_wape_handles_zero_actuals_that_break_mape():
    # MAPE would divide by zero on the first two hours; WAPE only needs the total.
    assert wape([0, 0, 5], [1, 0, 5]) == pytest.approx(1 / 5)


def test_wape_is_nan_when_all_actuals_are_zero():
    assert math.isnan(wape([0, 0], [1, 0]))


def test_skill_sign():
    assert skill(1.5, 2.0) == pytest.approx(0.25)
    assert skill(2.0, 2.0) == 0
    assert skill(3.0, 2.0) < 0


def test_coverage():
    assert coverage([1, 5, 10, 3], [0, 0, 0, 4], [2, 6, 8, 9]) == pytest.approx(0.5)


def test_metrics_reject_nan_and_shape_mismatch():
    with pytest.raises(ValueError):
        mae([1, float("nan")], [1, 2])
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])
