import numpy as np

from ainwp_ptype_original.vertical_profiles import interp_to_heights_numba


def test_original_endpoint_filling_is_preserved():
    values = np.array([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32)
    heights = np.array([[[[100.0]], [[1000.0]], [[2000.0]]]], dtype=np.float32)
    targets = np.array([0.0, 500.0, 5000.0], dtype=np.float32)
    result = interp_to_heights_numba(values, heights, targets)
    assert result[0, 0, 0, 0] == 1.0
    assert result[0, 2, 0, 0] == 3.0
