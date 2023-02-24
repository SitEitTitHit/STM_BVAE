import nanonispy
from scipy import optimize, interpolate, misc, ndimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import random
import DTFunction as DTF


# load, calculate mean curve
class GridProcess:

    def __init__(self, grid_path, load_modifier='None'):
        self.grid_path = grid_path
        self.name = grid_path[-6:-4]
        self.load_modifier = load_modifier
        grid_dict = nanonispy.read.Grid(grid_path)._load_data()
        print(f"Grid 0.{self.name} loaded. The keys are", list(grid_dict.keys()))

        # 虽然看起来暴力，但是应该不会侵入性地写入原始数据（3ds文件）
        if load_modifier == 'twisted':
            grid_dict['LI Demod 1 X (A)'] = np.load(grid_path[:-4]+'.twisted.npz')['twisted_dIdV']

        if load_modifier == 'GF0.5':
            grid_dict['LI Demod 1 X (A)'] = np.load(grid_path[:-4]+'.twisted_GF0.5.npz')['GFed_dIdV']

        # length: a tuple () of grid_width, grid_height, & curve.
        self.length = grid_dict['LI Demod 1 X (A)'].shape
        self.bias = grid_dict['Bias (V)'][0, 0]
        self.y = grid_dict['LI Demod 1 X (A)'].reshape(self.length[0] * self.length[1], self.length[2]).T

        # processing specific grids
        if self.name == '16' or self.name == 'OP32K':  # 就因为OP32K的最后多了几个点
            self.y = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[:-5]
            self.y = self.y.swapaxes(0, 2)
            self.length = self.y.shape
            self.bias = grid_dict['Bias (V)'][0, 0][:-5]
            self.y = self.y.reshape(self.length[0] * self.length[1], self.length[2]).T

        if self.name == '19':  # 就因为OD28K的最后多了几个点
            self.y = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[1:]
            self.y = self.y.swapaxes(0, 2)
            self.length = self.y.shape
            self.bias = grid_dict['Bias (V)'][0, 0][1:]
            self.y = self.y.reshape(self.length[0] * self.length[1], self.length[2]).T

        # get standard deviation & mean curve
        # (0105)标准差并不是一个好做法，剥离出normalization后续单独定义吧
        self.mean_curve = np.zeros(self.length[2])
        for i in range(self.length[2]):
            self.mean_curve[i] = np.mean(self.y[i])

        # interpolate的插值参数一般叫tck
        self.tck_mean = interpolate.splrep(self.bias, self.mean_curve)
        def f_mean_temp(x): return interpolate.splev(x, self.tck_mean)
        self.offset = optimize.fminbound(f_mean_temp, -0.01, 0.01)  # here 0.01 is arbitrary

        # ODNSC不修offset
        if self.name == '23':
            self.offset = 0

        print('offset = {:.8f}'.format(self.offset))

        self.y = self.y.T

    def f_mean(self, x): return interpolate.splev(x+self.offset, self.tck_mean)

    # offset plot
    def plot_offset(self):
        fig = plt.plot(DTF.bias_interp, self.f_mean(DTF.bias_interp))
        plt.scatter(self.offset, self.f_mean(self.offset), c='r', s=20)
        plt.savefig(self.grid_path[:-4] + '_offset.svg')
        plt.close('all')

    # 各种类型的数剧处理都给删了……看一下后续怎么做对不同数据处理方式拓展性好一些
    # grid interpolation, subtract mean curve, save to "_interp.npz"
    def interp(self, save_itp=False, map_gap=False, negative_second_derivative=False):

        mean_curve_interp = self.f_mean(DTF.bias_interp)

        if save_itp:
            dIdV_itp = np.zeros((self.length[0] * self.length[1], len(DTF.bias_interp)))
        if map_gap:
            gap = np.zeros(self.length[0] * self.length[1])
        if negative_second_derivative:
            max_d2f = np.zeros(self.length[0] * self.length[1])

        for i in range(self.length[0] * self.length[1]):

            tck = interpolate.splrep(self.bias - self.offset, self.y[i])
            def f(x): return interpolate.splev(x, tck)

            if save_itp:
                dIdV_itp[i] = f(DTF.bias_interp)
            if map_gap:
                gap[i] = DTF.gap_map(f)
            if negative_second_derivative:
                # 粗暴地全域取Max
                max_d2f[i] = max(DTF.ng_scd_deriv(f))

        # save. And do data type transform for pyTorch
        if save_itp:
            # dIdV_itp = dIdV_itp.astype(np.float32) / 1e-12  # normalize!
            # np.savez(self.grid_path[:-4] + '.interp.npz', bias=DTF.bias_interp, dIdV=dIdV_itp,
            #          mean_curve=mean_curve_interp / 1e-12, std=self.std, offset=self.offset)
            norm = np.mean(np.concatenate((self.mean_curve[:5], self.mean_curve[-5:])))
            dIdV_itp = dIdV_itp.astype(np.float32)
            if self.load_modifier == 'GF0.5':
                # np.savez(self.grid_path[:-4] + '.interp_GF05.npz', bias=DTF.bias_interp, dIdV=dIdV_itp / norm,
                #          mean_curve=mean_curve_interp / norm, std=norm, offset=self.offset)
                np.savez(self.grid_path[:-4] + '.interp_GF05_1.npz', bias=DTF.bias_interp, dIdV=dIdV_itp / norm,
                         mean_curve=mean_curve_interp / norm, std=norm, offset=self.offset)

        if map_gap:
            gap = gap.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.gap.npz', gap=gap)

        if negative_second_derivative:
            max_d2f = max_d2f.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.d2f.npz', d2f=max_d2f)


    # plot all the treatment we've done to the curve
    def all_plot(self):
        sample = self.y[random.randint(0, self.length[0] * self.length[1])]
        # sample = self.mean_curve
        # f = interpolate.interp1d(self.bias - self.offset, sample, kind='quadratic',
        #                          bounds_error=False, fill_value='extrapolate')
        tck = interpolate.splrep(self.bias - self.offset, sample)
        def f(x): return interpolate.splev(x, tck)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.xlabel('bias(mV)')
        ax1.plot(self.bias, sample, 'b--')
        ax1.plot(DTF.bias_interp, f(DTF.bias_interp), 'g')

        left_lim, right_lim = DTF.give_interp_lim(self.bias)
        ax1.plot(DTF.bias_ie, DTF.interp_extrap2(tck, self.bias, self.name, left_lim, right_lim), 'r')

        ax1.set_ylabel('original dI/dV')
        # sam_gap = gap_map(f)
        # ax1.axvline(-sam_gap, alpha=0.5, c='c', ls='--')
        # ax1.axvline(+sam_gap, alpha=0.5, c='c', ls='--')

        # sam_d2f = ng_scd_deriv(f)
        # ax2 = ax1.twinx()
        # ax2.plot(DTF.bias_interp, sam_d2f, c='orange', alpha=0.8)
        # ax2.scatter(DTF.bias_interp[np.argmax(sam_d2f)], max(sam_d2f), c='orange', s=50, marker='x')
        # ax2.set_ylim(min(sam_d2f) - (max(sam_d2f) - min(sam_d2f)) * 0.3,
        #              max(sam_d2f) + (max(sam_d2f) - min(sam_d2f)) * 0.3)
        # ax2.set_ylabel('negative second derivative')

        plt.suptitle(self.name)
        for i in range(100):
            if not os.path.exists(self.grid_path[:-4] + '_all_plot_' + str(i) + '.svg'):
                plt.savefig(self.grid_path[:-4] + '_all_plot_' + str(i) + '.svg')
                break
        plt.show()


def plot_gap_map(grid_path):
    gap = np.load(grid_path[:-4] + '.gap.npz')['gap']
    grid_size = int(np.sqrt(len(gap)))
    a = gap.reshape((grid_size, -1))
    current_cmap = matplotlib.cm.get_cmap("viridis").copy()
    current_cmap.set_bad(color='grey')
    plt.imshow(a)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig(grid_path[:-4] + '_gapmap.svg')


def plot_d2f_map(grid_path):
    d2f = np.load(grid_path[:-4] + '.d2f.npz')['d2f']
    grid_size = int(np.sqrt(len(d2f)))
    a = d2f.reshape((grid_size, -1))
    plt.imshow(a)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig(grid_path[:-4] + '_d2fmap.svg')


# enumerate all files in the dir
Flist = os.listdir(DTF.grid_set_path)
Flist_split = [f.split('.') for f in Flist]


def mean_curve_plot():
    means, grid_names = [], []

    for i in range(len(Flist)):
        if Flist_split[i][1] == '3ds':
            grid_names.append(Flist[i])

    for i, grid_name in enumerate(grid_names):
        Grid = GridProcess(DTF.grid_set_path + grid_name)
        a = []
        for j in np.linspace(-0.10, -0.08, num=21, endpoint=True):
            a.append(Grid.f_mean(j + Grid.offset))
        for j in np.linspace(0.08, 0.10, num=21, endpoint=True):
            a.append(Grid.f_mean(j + Grid.offset))
        means.append(np.mean(a))
        plt.plot(DTF.bias_interp, Grid.f_mean(DTF.bias_interp + Grid.offset) / np.mean(a),
                 color=cm.get_cmap('hot')(i/len(grid_names)*0.8))
    plt.show()


# 这一段是批量处理grid的
for i in range(len(Flist)):
    if Flist_split[i][1] == '3ds':
        Grid = GridProcess(DTF.grid_set_path + Flist[i], load_modifier='GF0.5')
        Grid.interp(save_itp=True)
        # Grid.all_plot4()


# 这一段是使用gaussianFilter
# for i in range(len(Flist)):
#     if Flist_split[i][1] == '3ds':
#         GFed_dIdV = gaussianFilter(DTF.grid_set_path + Flist[i], sig=0.5)
#         np.savez(DTF.grid_set_path+Flist[i][:-4]+'.twisted_GF0.5.npz', GFed_dIdV=GFed_dIdV)

# GFed_dIdV = gaussianFilter('grids/04_08_11_13/04.3ds', sig=0.5)
# np.savez('grids/04_08_11_13/04.twisted_GF0.5.npz', GFed_dIdV=GFed_dIdV)
# Grid = GridProcess('grids/04_08_11_13/04.3ds', load_modifier='GF0.5')
# Grid.interp(save_itp=True)