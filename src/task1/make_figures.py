import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Path to which figures will be saved
PATH = '../../report/task1/figures'
# Width of the text area in our document. Given in inches
TEXT_WIDTH = 5.78851
# Multiplicators for the figure's width
TEXT_WIDTH_MUL = 0.7
TEXT_WIDTH_MUL_WIDE = 0.9
# Aspect ratio of figures
ASPECT = 3. / 4.

FIG_WITDH = TEXT_WIDTH * TEXT_WIDTH_MUL
FIG_HEIGHT = FIG_WITDH * ASPECT
FIG_HEIGHT_FLAT = FIG_HEIGHT * 0.45

# Font sizes
SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 12


def setup_matplotlib():
    # Use LaTeX to typeset all text in the figure
    # This obviously needs a working LaTeX installation on the system
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': SMALL_SIZE,
        'axes.titlesize': MEDIUM_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': MEDIUM_SIZE,
        'text.usetex': True,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        'text.latex.preamble': ['\\usepackage{amsmath}\n'
                                '\\usepackage{amssymb}']
    })


def make_example_plot(path, n=4, left=0, right=3, res=50):
    width = TEXT_WIDTH * TEXT_WIDTH_MUL
    fig = plt.figure(figsize=(width, width*ASPECT))
    ax = plt.gca()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.grid(linestyle='dashed')

    x = np.linspace(left, right, num=res)
    for i in range(1, n+1):
        y = x ** i
        ax.plot(x, y, label=f'$y = x^{i}$')

    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(path, 'example_plot.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    setup_matplotlib()
    make_example_plot(PATH)

