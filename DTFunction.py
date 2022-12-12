import numpy as np
from scipy import optimize, interpolate, misc
import matplotlib.pyplot as plt

grid_set_path = 'grids/08_11_13_16_18_19_21_22_23/'
bias_interp = np.arange(-100e-3, 100e-3 + 1e-3, 1e-3)
# bias_ie = np.arange(-200e-3, 200e-3 + 1e-3, 1e-3)

bm0_tune = {'08': [0, 0], '11': [0, 0], '13': [0, 0], '16': [9e-12, 8.5e-13],
            '18': [0, 0], '19': [1e-12, 0e-13], '22': [-5e-12, 0], '23': [1e-12, -3e-13]}
bm1_tune = {'08': [0, 0], '11': [0, 0], '13': [0, 0], '16': [-2.5e-12, 0.5e-13],
            '18': [0, 0], '19': [6e-12, -3.5e-13], '22': [2e-12, 0], '23': [1e-12, 0]}
# mrange_change is for smoothing out the measure range change
# mrange_change = {'08': [25, 40], '11': [30, 50], '13': [30, 50]} for ext3
mrange_change = {'08': [25, 40], '11': [30, 50], '13': [30, 40]}

# for ext2 & ext3
# a0, a2 = 1e-3, 1e-2
# def f_ext(x, a0, a2): return a0 * x ** 3 + a2 * x
# b0 = np.real(np.roots([a0, 0, a2, -200e-3])[2])
# a = np.linspace(-b0, b0, 201, endpoint=True)
# bias_ie = f_ext(a, 1e-3, 1e-2)

a0, a2 = 1e-3, 2e-2
def f_ext(x, a0, a2): return a0 * x ** 3 + a2 * x
b0 = np.real(np.roots([a0, 0, a2, -200e-3])[2])
a = np.linspace(-b0, b0, 151, endpoint=True)
bias_ie = f_ext(a, a0, a2)
# plt.plot(f_ext(a, a0, a2), a)
# plt.scatter(f_ext(a, a0, a2), np.zeros(151)-b0, s=3)
# plt.show()

print(bias_ie[1]-bias_ie[0], bias_ie[76]-bias_ie[75])


# accept the path and open a .curves file (by R-STM)
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


# left lim < min(bias) < ... < max(bias) < right lim
def give_interp_lim(bias):
    if len(np.argwhere(bias_ie < min(bias))) > 0:
        left_lim = max(np.argwhere(bias_ie < min(bias)).flatten())
    else:
        left_lim = 0
    if len(np.argwhere(bias_ie > max(bias))) > 0:
        right_lim = min(np.argwhere(bias_ie > max(bias)).flatten())
    else:
        right_lim = -1
    return left_lim, right_lim


def fit_mrc(tck, name):
    if name <= '13':
        def f(x): return interpolate.splev(x, tck)
        x = np.concatenate((bias_ie[:mrange_change[name][0]],
                            bias_ie[mrange_change[name][1]:]))
        tck_new = interpolate.splrep(x, f(x))
        return tck_new
    else: return tck


# at the inner side = p_m2*f + (1-p_m2)*f_mean
# at the outer side = p_m1*f + (1-p_m1)*f_mean
# left_flag is used to denote whether this mixing is for the LEFT or RIGHT side
def linear_mix(p_m1, p_m2, f_bkgd, fm_bkgd, left, right, left_flag):
    def f_mixed(t):
        if left_flag:
            return (p_m1 + (p_m2 - p_m1)*(t - left)/(right - left)) * f_bkgd(t) \
                   + ((1 - p_m1) + (p_m1 - p_m2)*(t - left)/(right - left)) * fm_bkgd(t)
        else:
            return (p_m2 - (p_m2 - p_m1)*(t - left)/(right - left)) * f_bkgd(t) \
                   + ((1 - p_m2) + (p_m2 - p_m1)*(t - left)/(right - left)) * fm_bkgd(t)
    return f_mixed


# do the extrapolation, points besides L/R lims will be replaced by linear fit
# lim are indices
def interp_extrap(f, bias, left_lim, right_lim):
    padding = int(len(bias)/8)  # /8 here is arbitrary
    bias_cubic_interp = np.concatenate((bias[0:padding], bias[-padding:-1]))
    # f_extrap is a cubic polynomial fit for background (both ends)
    f_extrap = np.poly1d(np.polyfit(bias_cubic_interp, f(bias_cubic_interp), 1))

    if not right_lim == -1:
        return np.concatenate((f_extrap(bias_ie[0:left_lim]),
                               f(bias_ie[left_lim:right_lim]),
                               f_extrap(bias_ie[right_lim:])))
    else:
        return np.concatenate((f_extrap(bias_ie[0:left_lim]),
                               f(bias_ie[left_lim:])))


def interp_extrap1(tck, bias, left_lim, right_lim, return_all=False):

    def f(x): return interpolate.splev(x, tck)

    if right_lim == -1:
        return f(bias_ie)
    else:

        padding = int(len(bias)/4)  # /8 here is arbitrary
        cutoff = int(len(bias)/4)
        # f_bckgnd(s) is a linear fit for background (both ends)
        bckgnd0 = np.polyfit(bias[0:padding], f(bias[0:padding]), 1)
        f_bckgnd0 = np.poly1d(bckgnd0)
        bckgnd1 = np.polyfit(bias[-padding:], f(bias[-padding:]), 1)
        f_bckgnd1 = np.poly1d(bckgnd1)

        # connect two sides
        x0 = [bias_ie[left_lim], bias_ie[left_lim + cutoff]]
        y0 = [f_bckgnd0(x0[0]), f(x0[1])]
        bc0 = [[1, bckgnd0[0]], [1, interpolate.splev(x0[1], tck, der=1)]]
        f0 = interpolate.CubicSpline(x0, y0, bc_type=bc0)

        x1 = [bias_ie[right_lim - cutoff], bias_ie[right_lim]]
        y1 = [f(x1[0]), f_bckgnd1(x1[1])]
        bc1 = [[1, interpolate.splev(x1[0], tck, der=1)], [1, bckgnd1[0]]]
        f1 = interpolate.CubicSpline(x1, y1, bc_type=bc1)

        if return_all:
            return f_bckgnd0, f0, f1, f_bckgnd1, padding, cutoff
        else:
            return np.concatenate((
                f_bckgnd0(bias_ie[0:left_lim]),
                f0(bias_ie[left_lim:left_lim + cutoff]),
                f(bias_ie[left_lim + cutoff:right_lim - cutoff]),
                f1(bias_ie[right_lim - cutoff:right_lim]),
                f_bckgnd1(bias_ie[right_lim:])))


def interp_extrap2(tck, bias, name, left_lim, right_lim, f_mean, p_m1=0.2, p_m2=0.7, return_all=False):

    if name <= '13': tck = fit_mrc(tck, name)
    def f(x): return interpolate.splev(x, tck)
    if right_lim == -1: return f(bias_ie)
    else:

        padding = int(len(bias)/4)  # /8 here is arbitrary
        cutoff = int(len(bias)/6)
        # f_bckgnd(s) is a linear fit for background (both ends)
        bckgnd0 = np.polyfit(bias[0:padding], f(bias[0:padding]), 1)
        bm0 = np.polyfit(bias[0:padding], f_mean(bias[0:padding]), 1) + bm0_tune[name]
        def f_bckgnd0(x):
            left = bias_ie[0]
            right = bias_ie[left_lim]
            return (p_m1 + (p_m2 - p_m1) * (x - left) / (right - left)) * np.poly1d(bckgnd0)(x) \
                   + ((1 - p_m1) + (p_m1 - p_m2) * (x - left) / (right - left)) * np.poly1d(bm0)(x)

        bckgnd1 = np.polyfit(bias[-padding:], f(bias[-padding:]), 1)
        bm1 = np.polyfit(bias[-padding:], f_mean(bias[-padding:]), 1) + bm1_tune[name]
        def f_bckgnd1(x):
            left = bias_ie[right_lim]
            right = bias_ie[-1]
            return (p_m2 - (p_m2 - p_m1) * (x - left) / (right - left)) * np.poly1d(bckgnd1)(x) \
                   + ((1-p_m2) + (p_m2 - p_m1) * (x - left) / (right - left)) * np.poly1d(bm1)(x)

        # connect two sides
        x0 = [bias_ie[left_lim], bias_ie[left_lim + cutoff]]
        y0 = [f_bckgnd0(x0[0]), f(x0[1])]
        bc0 = [[1, bckgnd0[0]], [1, interpolate.splev(x0[1], tck, der=1)]]
        f0 = interpolate.CubicSpline(x0, y0, bc_type=bc0)

        x1 = [bias_ie[right_lim - cutoff], bias_ie[right_lim]]
        y1 = [f(x1[0]), f_bckgnd1(x1[1])]
        bc1 = [[1, interpolate.splev(x1[0], tck, der=1)], [1, bckgnd1[0]]]
        f1 = interpolate.CubicSpline(x1, y1, bc_type=bc1)

        if return_all:
            return f_bckgnd0, f0, f1, f_bckgnd1, padding, cutoff
        else:
            return np.concatenate((
                f_bckgnd0(bias_ie[0:left_lim]),
                f0(bias_ie[left_lim:left_lim + cutoff]),
                f(bias_ie[left_lim + cutoff:right_lim - cutoff]),
                f1(bias_ie[right_lim - cutoff:right_lim]),
                f_bckgnd1(bias_ie[right_lim:])))


def interp_extrap4(tck, bias, name, left_lim, right_lim, f_mean, p_m1=0.2, p_m2=0.7, return_all=False):

    def f(x): return interpolate.splev(x, tck)
    # def f(x): return interpolate.splev(x, fit_mrc(tck, name))

    if right_lim == -1: return f(bias_ie)
    else:

        padding = int(len(bias)/4)  # /8 here is arbitrary
        cutoff = int(len(bias)/5)
        # f_bckgnd(s) is a linear fit for background (both ends)
        bckgnd0 = np.polyfit(bias[0:padding], f(bias[0:padding]), 1)
        bm0 = np.polyfit(bias[0:padding], f_mean(bias[0:padding]), 1) + bm0_tune[name]
        f_bckgnd0 = linear_mix(p_m1, p_m2, np.poly1d(bckgnd0), np.poly1d(bm0),
                               bias_ie[0], bias_ie[left_lim], left_flag=True)

        bckgnd1 = np.polyfit(bias[-padding:], f(bias[-padding:]), 1)
        bm1 = np.polyfit(bias[-padding:], f_mean(bias[-padding:]), 1) + bm1_tune[name]
        f_bckgnd1 = linear_mix(p_m1, p_m2, np.poly1d(bckgnd1), np.poly1d(bm1),
                               bias_ie[right_lim], bias_ie[-1], left_flag=False)

        # connect two sides
        f0 = linear_mix(0, 1, f, f_bckgnd0, bias_ie[left_lim], bias_ie[left_lim + cutoff], left_flag=True)
        f1 = linear_mix(0, 1, f, f_bckgnd1, bias_ie[right_lim - cutoff], bias_ie[right_lim], left_flag=False)

        if return_all:
            return f_bckgnd0, f0, f1, f_bckgnd1, padding, cutoff, bckgnd0, bm0, bckgnd1, bm1
        else:
            return np.concatenate((
                f_bckgnd0(bias_ie[0:left_lim]),
                f0(bias_ie[left_lim:left_lim + cutoff]),
                f(bias_ie[left_lim + cutoff:right_lim - cutoff]),
                f1(bias_ie[right_lim - cutoff:right_lim]),
                f_bckgnd1(bias_ie[right_lim:])))
