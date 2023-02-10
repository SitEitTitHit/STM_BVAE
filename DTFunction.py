import numpy as np
from scipy import optimize, interpolate, misc
import matplotlib.pyplot as plt

grid_set_path = 'grids/04_08_11_13_16_18_19_22_23/'
bias_interp = np.arange(-100e-3, 100e-3 + 1e-3, 1e-3)

# mrange_change is for smoothing out the measure range change
mrange_change = {'08': [25, 40], '11': [30, 50], '13': [30, 40]}


# accept the path and open a .curves file (from R-STM)
# return curve: curve[0] = Xs, curve[1] = Ys
def open_curve(path):
    f = open(path)
    flag_data = 0
    curve = []
    for line in f.readlines():
        if line == 'data\n':
            flag_data = 1
        elif flag_data:
            curve.append([float(i) for i in line.split()])
    return np.array(curve).T


def gaussianFilter(grid_path, sig=0.5):
    # grid_path is the npz file to be processed
    dIdV = np.load(grid_path[:-4]+'.twisted.npz')['twisted_dIdV']
    # this path is editable
    GFed_dIdV = np.zeros(dIdV.shape)
    for i in range(dIdV.shape[2]):
        GFed_dIdV[:, :, i] = ndimage.gaussian_filter(dIdV[:, :, i], sigma=sig)
        print(dIdV[:, :, i].shape)
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(dIdV[:, :, i])
        # axs[1].imshow(GFed_dIdV[:, :, i])
        # plt.show()
    return GFed_dIdV


# calculate gap for each point
def gap_map(f, lower_bound=-0.04, upper_bound=0.04):  # in eVolt
    # find maximum in 笨办法
    # 都怪不可以直接写optimize.fminbound(-f)，f不让直接加负号
    x_left = np.arange(lower_bound, 0, 1e-5)
    x_right = np.arange(0, upper_bound, 1e-5)
    gap_left = lower_bound + np.argmax(f(x_left)) * 1e-5
    gap_right = 0 + np.argmax(f(x_right)) * 1e-5
    if gap_left < lower_bound + 1e-5:
        if gap_right > upper_bound - 1e-5:
            return float('nan')
        else:
            return gap_right
    else:
        return -gap_left


# negative second derivative
def ng_scd_deriv(f):
    d2f = np.zeros(len(bias_interp))
    for i, x in enumerate(bias_interp):
        d2f[i] = - misc.derivative(f, x, dx=5e-3, n=2)
    return d2f


# 需要重新整理一下抹平换量程的函数
# def fit_mrc(tck, name):
#     if name <= '13':
#         def f(x): return interpolate.splev(x, tck)
#         x = np.concatenate((bias_ie[:mrange_change[name][0]],
#                             bias_ie[mrange_change[name][1]:]))
#         tck_new = interpolate.splrep(x, f(x))
#         return tck_new
#     else: return tck