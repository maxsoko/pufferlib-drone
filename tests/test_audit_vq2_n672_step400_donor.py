import pytest

from scripts.audit_vq2_n672_step400_donor import _history_numeric_differences


def test_history_difference_compares_nested_numeric_values() -> None:
    left = [{"step": 50, "metric": {"correlation": 0.25}, "passed": True}]
    right = [{"step": 50, "metric": {"correlation": 0.25000001}, "passed": True}]
    result = _history_numeric_differences(left, right)
    assert result["numeric_values_compared"] == 2
    assert result["maximum_absolute_error"] == pytest.approx(1e-8)


def test_history_difference_rejects_structure_or_boolean_change() -> None:
    with pytest.raises(RuntimeError, match="structure"):
        _history_numeric_differences([{"step": 50}], [{"other": 50}])
    with pytest.raises(RuntimeError, match="boolean"):
        _history_numeric_differences([{"passed": True}], [{"passed": False}])
