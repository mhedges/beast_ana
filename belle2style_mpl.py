# Author: M. T. Hedges


"""
Belle2-like style for matplotlib. Implement with:

(example)
import matplotlib.pyplot as plt
import belle2style_mpl
style = belle2style_mpl.b2_style_mpl()
plt.style.use(style)

Note that color of scatter plot points must be set *manually*!
"""
__all__ = [
    'style_mpl',
    ]

def b2_style_mpl():

    STYLE = {}


    STYLE['lines.linewidth'] = 1
    STYLE['errorbar.capsize'] = 0.0

    # font
    STYLE['font.family'] = 'sans-serif'
    STYLE['mathtext.fontset'] = 'stixsans'
    STYLE['mathtext.default'] = 'rm'
    # helvetica usually not present on linux
    STYLE['font.sans-serif'] = 'helvetica, Helvetica, Nimbus Sans L, Mukti Narrow, FreeSans'

    # figure layout
    STYLE['figure.figsize'] = 8.75, 5.92
    #   atlasStyle->SetPaperSize(20,26); # in cm
    # STYLE['figure.figsize'] =  10.2362205, 7.874015 # in inc, not working
    STYLE['figure.facecolor'] = 'white'
    STYLE['figure.subplot.bottom'] = 0.16
    STYLE['figure.subplot.top'] = 0.95
    STYLE['figure.subplot.left'] = 0.16
    STYLE['figure.subplot.right'] = 0.95

    # axes
    STYLE['axes.labelsize'] = 20
    STYLE['axes.grid'] = False
    STYLE['xtick.labelsize'] = 19
    STYLE['xtick.major.size'] = 12
    STYLE['xtick.minor.visible'] = True
    STYLE['xtick.minor.size'] = 6
    STYLE['ytick.labelsize'] = 19
    STYLE['ytick.major.size'] = 14
    STYLE['ytick.minor.visible'] = True
    STYLE['ytick.minor.size'] = 7
    STYLE['lines.markersize'] = 9.6
    STYLE['lines.color'] = 'black'

    # marker colors

    # STYLE['lines.markeredgewidth'] = 0. # not working, it changes other stuff

    # legend
    STYLE['legend.numpoints'] = 1
    STYLE['legend.fontsize'] = 19
    STYLE['legend.labelspacing'] = 0.3
    STYLE['legend.frameon'] = False

    # what cannot be set with rcParams:
    # * markeredgewidth
    # * axis-label alignment
    # * axis-label offset
    # * axis-ticks

    return STYLE

