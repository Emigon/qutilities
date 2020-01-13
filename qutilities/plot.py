""" plot.py

author: daniel parker
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_s21(s21, axes = None, xunits = None):
    """ plot s21 across three panels (magnitude, phase, cplx scatter) """
    if axes is None:
        fig, axes = plt.subplots(ncols = 3)

    axes[0].set_title('$|S_{21}|$')
    axes[1].set_title('$Arg (S_{21})$')
    axes[2].set_title('$S_{21}$ - complex')

    plt.sca(axes[0])
    s21.plot(style = 'dBm', xunits = xunits)
    plt.sca(axes[1])
    s21.plot(style = 'deg', xunits = xunits)
    plt.sca(axes[2])
    s21.plotz()

    axes[2].scatter(1, 0, marker = 'x', color = 'r', s = 3)

    return axes
