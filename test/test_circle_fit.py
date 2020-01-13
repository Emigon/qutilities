import pytest

import numpy as np
import matplotlib.pyplot as plt

from qutilities import *
from qutilities.circle import *

@pytest.fixture
def notch_resonator():
    nch = ideal_notch()
    s21 = nch(np.linspace(4.5*ureg('GHz'), 5.5*ureg('GHz'), 10000))
    return s21

@pytest.mark.plot
def test_good_circle_fit(notch_resonator):
    circle, err = circle_fit(notch_resonator)
    fig, axes = plt.subplots()
    notch_resonator.plotz()
    circle.add_to(axes)

    plt.show()

    assert circle.r == pytest.approx(0.25)
