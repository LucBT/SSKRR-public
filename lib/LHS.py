import numpy as np
from smt.sampling_methods import LHS
def get_LHS_sampling(_xlimits, _num, _criterion, seed = None):
    """Produce the LHS sampling in _xlimits with _num points following the _criterion

    Args:
        _xlimits (ndarray): [[x0, x1], [y0, y1], [z0, z1], ...]
        _num (int) : number of obs
        _criterion : among [‘center’, ‘maximin’, ‘centermaximin’, ‘correlation’, ‘c’, ‘m’, ‘cm’, ‘corr’, ‘ese’]
    """
    sampling = LHS(xlimits = _xlimits, criterion = _criterion, random_state = seed)
    num = _num
    _x_sampling = sampling(num)

    return _x_sampling