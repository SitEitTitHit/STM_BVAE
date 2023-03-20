import numpy as np
from scipy import optimize, interpolate, misc, ndimage
import matplotlib.pyplot as plt

grid_set_path = 'grids/04_08_11_13_16_18_19_22_23/'
bias_interp = np.arange(-100e-3, 100e-3 + 1e-3, 1e-3)


# 这段整个改了一下fit的横轴
a0, a2 = 1e-3, 2e-2
def f_ext(x, a0, a2): return a0 * x ** 3 + a2 * x
b0 = np.real(np.roots([a0, 0, a2, -100e-3])[2])
a = np.linspace(-b0, b0, 201, endpoint=True)
bias_ie = f_ext(a, a0, a2)
# plt.plot(f_ext(a, a0, a2), a)
# plt.scatter(f_ext(a, a0, a2), np.zeros(201)-b0, s=3)
# plt.show()
# print(bias_ie[1]-bias_ie[0], bias_ie[101]-bias_ie[100])
bias_interp = bias_ie


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


# 处理掉多余的点
def detele_extra_points(name, grid_dict, bias, y, length):
    if name == '16' or name == 'OP32K':  # 就因为OP32K的最后多了几个点
        y = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[:-5]
        y = y.swapaxes(0, 2)
        length = y.shape
        bias = grid_dict['Bias (V)'][0, 0][:-5]
        y = y.reshape(length[0]*length[1], length[2]).T

    if name == '19':  # 就因为OD28K的最后多了几个点
        y = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[1:]
        y = y.swapaxes(0, 2)
        length = y.shape
        bias = grid_dict['Bias (V)'][0, 0][1:]
        y = y.reshape(length[0]*length[1], length[2]).T
    return length, bias, y


# mrange_change is for smoothing out the measure range change
# python取list左闭右开
mrange_changes = {'04': [21, 23], '08': [11, 13], '11': [16, 18], '13': [10, 12]}


# 需要重新整理一下抹平换量程的函数
# 思路直接drop掉附近的点呢？反正本身也要插值
def mrange_change(name, bias, y, length):
    if name <= '13':
        y = np.concatenate((y[:mrange_changes[name][0]], y[mrange_changes[name][1]:]))
        bias = np.concatenate((bias[:mrange_changes[name][0]], bias[mrange_changes[name][1]:]))
        length[2] = y.shape[0]
    return length, bias, y