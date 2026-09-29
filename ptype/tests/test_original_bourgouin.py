import numpy as np

from ainwp_ptype_original.bourgouin import calc_bourgouin_ptype


HEIGHTS = np.arange(0, 5001, 250)


def test_original_all_warm_and_all_cold_classes():
    assert calc_bourgouin_ptype(np.full(HEIGHTS.shape, 2.0), HEIGHTS)[0] == 0
    assert calc_bourgouin_ptype(np.full(HEIGHTS.shape, -2.0), HEIGHTS)[0] == 1
