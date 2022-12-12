import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import colorsys


# put in a Colormap, plot the RGB value and the colormap itself
def plotLSColormap(cmp):
    rgba = cmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3))
    col = ['r', 'g', 'b']
    for i in range(3):
        ax.plot(np.arange(256) / 256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('RGB')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cax = plt.axes([0.1, 0.05, 0.8, 0.04])
    cax.set_axis_off()
    cax.imshow(gradient, aspect='auto', cmap=cmp)
    plt.subplots_adjust(bottom=0.1, top=0.85, left=0.1, right=0.9)
    plt.show()


def createLightnessMap(rbga):
    # Colormap as a function returns RGBA values (A for alpha for transparency)
    # need to drop a
    (r, g, b) = rbga[:3]
    hls = colorsys.rgb_to_hls(r, g, b)
    l_list = [0.9, 0.3]
    rgbs = [colorsys.hls_to_rgb(hls[0], l, hls[2]) for l in l_list]
    rgbs = np.array(rgbs).T.tolist()
    rgbs = [[[0, rgb[0], rgb[0]], [1, rgb[1], rgb[1]]] for rgb in rgbs]

    cdict = {'red': rgbs[0], 'green': rgbs[1], 'blue': rgbs[2]}

    return colors.LinearSegmentedColormap('', segmentdata=cdict, N=256)  # arg '' for name (not used)
