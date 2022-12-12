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
        # self.grid_name = grid_path[-9:-4]
        self.name = grid_path[-6:-4]
        self.load_modifier = load_modifier
        grid_dict = nanonispy.read.Grid(grid_path)._load_data()
        print("Grid loaded. The keys are", list(grid_dict.keys()))

        # 好暴力啊这个操作
        if load_modifier == 'twisted':
            grid_dict['LI Demod 1 X (A)'] = np.load(grid_path[:-4]+'.twisted.npz')['twisted_dIdV']

        if load_modifier == 'GF0.5':
            grid_dict['LI Demod 1 X (A)'] = np.load(grid_path[:-4]+'.twisted_GF0.5.npz')['GFed_dIdV']

        # length: a tuple () of grid_width, grid_height, & curve.
        self.length = grid_dict['LI Demod 1 X (A)'].shape
        self.bias = grid_dict['Bias (V)'][0, 0]

        # get standard deviation & mean curve
        self.y = grid_dict['LI Demod 1 X (A)'].reshape(self.length[0] * self.length[1], self.length[2]).T

        if self.name == 'OP32K' or self.name == '16':  # 就因为OP32K的最后多了几个点
            self.y = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[:-5]
            self.y = self.y.swapaxes(0, 2)
            self.length = self.y.shape
            self.bias = grid_dict['Bias (V)'][0, 0][:-5]
            self.y = self.y.reshape(self.length[0] * self.length[1], self.length[2]).T

        if self.name == '19':  # 就因为OP32K的最后多了几个点
            self.y = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[1:]
            self.y = self.y.swapaxes(0, 2)
            self.length = self.y.shape
            self.bias = grid_dict['Bias (V)'][0, 0][1:]
            self.y = self.y.reshape(self.length[0] * self.length[1], self.length[2]).T

        self.std = np.std(self.y)
        self.mean_curve = np.zeros(self.length[2])
        for i in range(self.length[2]):
            self.mean_curve[i] = np.mean(self.y[i])

        # mean curve interpolation & offset calibration
        # self.f_mean = interpolate.interp1d(self.bias, self.mean_curve, kind='quadratic',
        #                                    bounds_error=False, fill_value='extrapolate')
        self.tck_mean = interpolate.splrep(self.bias, self.mean_curve)
        def f_mean_temp(x): return interpolate.splev(x, self.tck_mean)
        self.offset = optimize.fminbound(f_mean_temp, -0.01, 0.01)  # here 0.01 is arbitrary

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

    # grid interpolation, subtract mean curve, save to "_interp.npz"
    def interp(self, save_itp=False, map_gap=False, negative_second_derivative=False,
               save_itp_sub=False, save_itp_ext=False, linear_norm_flag=False,
               save_itp_ext1=False, save_itp_ext2=False, save_itp_ext4=False):

        mean_curve_interp = self.f_mean(DTF.bias_interp)

        if save_itp or save_itp_sub:
            dIdV_itp = np.zeros((self.length[0] * self.length[1], len(DTF.bias_interp)))
        if save_itp_ext:
            dIdV_itp_ext = np.zeros((self.length[0] * self.length[1], len(DTF.bias_ie)))
            left_lim, right_lim = DTF.give_interp_lim(self.bias)
        if save_itp_ext1:
            dIdV_itp_ext1 = np.zeros((self.length[0] * self.length[1], len(DTF.bias_ie)))
            left_lim, right_lim = DTF.give_interp_lim(self.bias)
        if save_itp_ext2:
            dIdV_itp_ext2 = np.zeros((self.length[0] * self.length[1], len(DTF.bias_ie)))
            left_lim, right_lim = DTF.give_interp_lim(self.bias)
        if save_itp_ext4:
            dIdV_itp_ext4 = np.zeros((self.length[0] * self.length[1], len(DTF.bias_ie)))
            left_lim, right_lim = DTF.give_interp_lim(self.bias)
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
                gap[i] = gap_map(f)
            if negative_second_derivative:
                # 粗暴地全域取Max
                max_d2f[i] = max(ng_scd_deriv(f))
            if save_itp_ext:
                dIdV_itp_ext[i] = interp_extrap(f, self.bias, left_lim, right_lim)
            if save_itp_ext1:
                dIdV_itp_ext1[i] = interp_extrap1(tck, self.bias, left_lim, right_lim)
            if save_itp_ext2:
                dIdV_itp_ext2[i] = DTF.interp_extrap2(tck, self.bias, self.name, left_lim, right_lim, self.f_mean)
            if save_itp_ext4:
                dIdV_itp_ext4[i] = DTF.interp_extrap4(tck, self.bias, self.name, left_lim, right_lim, self.f_mean)

        # subtracted with mean curve and std
        if save_itp_sub:
            dIdV_itp_sub = dIdV_itp.T
            for i in range(len(DTF.bias_interp)):
                dIdV_itp_sub[i] = (dIdV_itp_sub[i] - mean_curve_interp[i]) / self.std
            dIdV_itp_sub = dIdV_itp_sub.T

            dIdV_min = np.min(dIdV_itp_sub)
            dIdV_max = np.max(dIdV_itp_sub)

        # linear_norm (only when needed i.e. Cross Entropy Loss)
        def linear_norm(lower_lim=0.2, upper_lim=0.8):
            nonlocal dIdV_itp_sub
            dIdV_itp_sub = lower_lim + (dIdV_itp_sub - dIdV_min) / (dIdV_max - dIdV_min) * (upper_lim - lower_lim)

        # linear transform to (lower_limit, upper_limit)
        if linear_norm_flag:
            linear_norm()

        # data type transform for pyTorch
        # save
        if save_itp:
            # dIdV_itp = dIdV_itp.astype(np.float32) / 1e-12  # normalize!
            # np.savez(self.grid_path[:-4] + '.interp.npz', bias=DTF.bias_interp, dIdV=dIdV_itp,
            #          mean_curve=mean_curve_interp / 1e-12, std=self.std, offset=self.offset)
            norm = np.mean(np.concatenate((self.mean_curve[:5], self.mean_curve[-5:])))
            dIdV_itp = dIdV_itp.astype(np.float32)
            if self.load_modifier == 'GF0.5':
                np.savez(self.grid_path[:-4] + '.interp_GF05.npz', bias=DTF.bias_interp, dIdV=dIdV_itp / norm,
                         mean_curve=mean_curve_interp / norm, std=norm, offset=self.offset)

        if map_gap:
            gap = gap.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.gap.npz', gap=gap)

        if negative_second_derivative:
            max_d2f = max_d2f.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.d2f.npz', d2f=max_d2f)

        if save_itp_sub:
            dIdV_itp_sub = dIdV_itp_sub.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.interp_sub.npz', bias=DTF.bias_interp, dIdV=dIdV_itp_sub,
                     mean_curve=mean_curve_interp, std=self.std, offset=self.offset,
                     min=dIdV_min, max=dIdV_max)
            print(self.grid_path + ' processed and saved')

        if save_itp_ext:
            dIdV_itp_ext = dIdV_itp_ext.astype(np.float32) / 1e-12
            np.savez(self.grid_path[:-4] + '.interp_ext.npz', bias=DTF.bias_ie, dIdV=dIdV_itp_ext,
                     mean_curve=mean_curve_interp / 1e-12, std=self.std, offset=self.offset)

        if save_itp_ext1:
            a = interp_extrap1(self.tck_mean, self.bias, left_lim, right_lim)
            norm = np.mean(np.concatenate((a[:20], a[-20:])))
            dIdV_itp_ext1 = dIdV_itp_ext1.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.interp_ext1.npz', bias=DTF.bias_ie, dIdV=dIdV_itp_ext1 / norm,
                     mean_curve=mean_curve_interp/norm, std=norm, offset=self.offset)

        if save_itp_ext2:
            a = DTF.interp_extrap2(self.tck_mean, self.bias, self.name, left_lim, right_lim, self.f_mean)
            norm = np.mean(np.concatenate((a[:20], a[-20:])))
            dIdV_itp_ext2 = dIdV_itp_ext2.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.interp_ext3.npz', bias=DTF.bias_ie, dIdV=dIdV_itp_ext2 / norm,
                     mean_curve=mean_curve_interp/norm, std=norm, offset=self.offset)

        if save_itp_ext4:
            a = DTF.interp_extrap4(self.tck_mean, self.bias, self.name, left_lim, right_lim, self.f_mean)
            norm = np.mean(np.concatenate((a[:5], a[-5:])))
            dIdV_itp_ext4 = dIdV_itp_ext4.astype(np.float32)
            np.savez(self.grid_path[:-4] + '.interp_ext4.npz', bias=DTF.bias_ie, dIdV=dIdV_itp_ext4 / norm,
                     mean_curve=mean_curve_interp/norm, std=norm, offset=self.offset)

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

    def all_plot1(self):
        offset = 1.5e-12
        fig, ax = plt.subplots()

        for i in range(9):
            ax = plt.subplot(3, 3, i+1)

            sample = self.y[random.randint(0, self.length[0] * self.length[1])]
            tck = interpolate.splrep(self.bias - self.offset, sample)
            def f(x): return interpolate.splev(x, tck)

            ax.plot(self.bias, sample-offset, 'b--')
            ax.plot(DTF.bias_interp, f(DTF.bias_interp)-offset, 'g')

            left_lim, right_lim = DTF.give_interp_lim(self.bias)

            if not right_lim == -1:
                f_bckgnd0, f0, f1, f_bckgnd1, padding, cutoff = \
                    DTF.interp_extrap2(tck, self.bias, self.name, left_lim, right_lim, self.f_mean, return_all=True)
                ax.plot(DTF.bias_ie[left_lim + cutoff:right_lim - cutoff], f(DTF.bias_ie[left_lim + cutoff:right_lim - cutoff]), 'g')
                ax.plot(self.bias[0:padding], f_bckgnd0(self.bias[0:padding]), 'b')
                ax.plot(DTF.bias_ie[0:left_lim], f_bckgnd0(DTF.bias_ie[0:left_lim]), 'b')
                ax.plot(self.bias[-padding:], f_bckgnd1(self.bias[-padding:]), 'b')
                ax.plot(DTF.bias_ie[right_lim:], f_bckgnd1(DTF.bias_ie[right_lim:]), 'b')
                ax.plot(DTF.bias_ie[left_lim:left_lim + cutoff], f0(DTF.bias_ie[left_lim:left_lim + cutoff]), 'y')
                ax.plot(DTF.bias_ie[right_lim - cutoff:right_lim], f1(DTF.bias_ie[right_lim - cutoff:right_lim]), 'y')

                ax.plot(DTF.bias_ie, DTF.interp_extrap2(tck, self.bias, self.name, left_lim, right_lim, self.f_mean) + offset, 'r')

            if self.name <= '13':
                x = np.concatenate((DTF.bias_ie[left_lim:DTF.mrange_change[self.name][0]],
                                    DTF.bias_ie[DTF.mrange_change[self.name][1]:right_lim]))
                def f_prime(x): return interpolate.splev(x, DTF.fit_mrc(tck, self.name))
                ax.plot(x, f_prime(x), 'b')
                ax.plot(DTF.bias_ie[DTF.mrange_change[self.name]],
                        f_prime(DTF.bias_ie[DTF.mrange_change[self.name]]), 'y')

            ax.tick_params(axis='both', which='both', labelsize=8)

        plt.suptitle(self.name)
        fig.set_figheight(9)
        fig.set_figwidth(12)
        # for i in range(100):
        #     if not os.path.exists(self.grid_path[:-4] + '_all_plot_' + str(i) + '.svg'):
        #         plt.savefig(self.grid_path[:-4] + '_all_plot_' + str(i) + '.svg')
        #         break
        plt.show()

    def all_plot4(self):
        offset = 1.5e-12
        fig, ax = plt.subplots()

        for i in range(9):
            ax = plt.subplot(3, 3, i+1)

            sample = self.y[random.randint(0, self.length[0] * self.length[1])]
            tck = interpolate.splrep(self.bias - self.offset, sample)
            def f(x): return interpolate.splev(x, tck)

            ax.plot(self.bias, sample-offset, 'm--')
            ax.plot(self.bias, f(self.bias)-offset, 'g')

            left_lim, right_lim = DTF.give_interp_lim(self.bias)

            if right_lim == -1:
                ax.plot(DTF.bias_ie, f(DTF.bias_ie), 'g')
            else:
                f_bckgnd0, f0, f1, f_bckgnd1, padding, cutoff, bckgnd0, bm0, bckgnd1, bm1 = \
                    DTF.interp_extrap4(tck, self.bias, self.name, left_lim, right_lim, self.f_mean, return_all=True)
                ax.plot(DTF.bias_ie[left_lim + cutoff:right_lim - cutoff], f(DTF.bias_ie[left_lim + cutoff:right_lim - cutoff]), 'g')

                ax.plot(self.bias[:padding], f_bckgnd0(self.bias[:padding]), 'y')
                ax.plot(self.bias[:padding], np.poly1d(bckgnd0)(self.bias[:padding]), 'y--')
                ax.plot(DTF.bias_ie[:left_lim], f_bckgnd0(DTF.bias_ie[:left_lim]), 'y')
                ax.plot(DTF.bias_ie[:left_lim], np.poly1d(bm0)(DTF.bias_ie[:left_lim]), 'y--')
                ax.plot(self.bias[-padding:], f_bckgnd1(self.bias[-padding:]), 'y')
                ax.plot(self.bias[-padding:], np.poly1d(bckgnd1)(self.bias[-padding:]), 'y--')
                ax.plot(DTF.bias_ie[right_lim:], f_bckgnd1(DTF.bias_ie[right_lim:]), 'y')
                ax.plot(DTF.bias_ie[right_lim:], np.poly1d(bm1)(DTF.bias_ie[right_lim:]), 'y--')

                ax.plot(DTF.bias_ie[left_lim:left_lim+cutoff], f0(DTF.bias_ie[left_lim:left_lim+cutoff]), 'b')
                ax.plot(DTF.bias_ie[right_lim-cutoff:right_lim], f1(DTF.bias_ie[right_lim-cutoff:right_lim]), 'b')

            if self.name <= '13':
                def f_prime(x): return interpolate.splev(x, DTF.fit_mrc(tck, self.name))
                xrange = DTF.bias_ie[DTF.mrange_change[self.name][0]:DTF.mrange_change[self.name][1]]
                ax.plot(xrange, f_prime(xrange), 'cyan')

            ax.plot(DTF.bias_ie, DTF.interp_extrap4(tck, self.bias, self.name, left_lim, right_lim, self.f_mean) + offset, 'r')

            ax.tick_params(axis='both', which='both', labelsize=8)

        plt.suptitle(self.name)
        fig.set_figheight(9)
        fig.set_figwidth(12)
        for i in range(100):
            if not os.path.exists(self.grid_path[:-4] + '_all_plot_' + str(i) + '.svg'):
                plt.savefig(self.grid_path[:-4] + '_all_plot_' + str(i) + '.svg')
                break
        # plt.show()


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
        # Grid.plot_offset()
        # Grid.interp(save_interp=True, map_gap=False, save_interp_sub=True, negative_second_derivative=False)
        # plot_gap_map(grid_set_path + Flist[i])
        # plot_d2f_map(grid_set_path + Flist[i])

    # fig, ax = plt.subplots()
    # plt.plot(range(len(means)), means)
    plt.show()


def mean_curve_plot2():
    for i in range(len(Flist)):
        if Flist_split[i][1] == '3ds':
            Grid = GridProcess(grid_set_path + Flist[i])
            padding = int(len(Grid.bias) / 8)
            bias_cubic_interp = np.concatenate((Grid.bias[0:padding], Grid.bias[-padding:-1]))
            y_cubic_interp = np.concatenate((Grid.mean_curve[0:padding], Grid.mean_curve[-padding:-1]))
            f_cubic = np.poly1d(np.polyfit(bias_cubic_interp, y_cubic_interp, 3))
            y_normed = []
            for j in range(len(Grid.bias)):
                y_normed.append(Grid.mean_curve[j] / f_cubic(Grid.bias[j]))

            plt.plot(Grid.bias - Grid.offset, y_normed, color=cm.get_cmap('hsv')(i / len(Flist)))
    # fig, ax = plt.subplots()
    # plt.plot(range(len(means)), means)
    plt.show()


# norm_lim = [[40,50],[40,50],[40,50],[40,50],[40,50],[40,50],[40,50],[20,30]]
def mean_curve_plot3():
    grid_names = []
    for i in range(len(Flist)):
        if Flist_split[i][1] == '3ds':
            grid_names.append(Flist[i])
    for i, grid_name in enumerate(grid_names):
        Grid = GridProcess(DTF.grid_set_path + grid_name)
        left_lim, right_lim = DTF.give_interp_lim(Grid.bias)
        a = DTF.interp_extrap2(Grid.tck_mean, Grid.bias, Grid.name, left_lim, right_lim, Grid.f_mean)
        # norm = np.mean(np.concatenate((a[norm_lim[i][0]:norm_lim[i][1]], a[-norm_lim[i][1]:-norm_lim[i][0]])))
        norm = np.mean(np.concatenate((a[:20], a[-20:])))
        plt.plot(DTF.bias_ie, a / norm,
                 color=cm.get_cmap('hot')(i/len(grid_names)*0.8))
    plt.savefig(DTF.grid_set_path+'means3.svg')
    # plt.show()


def mean_curve_plot4():
    grid_names = []
    for i in range(len(Flist)):
        if Flist_split[i][1] == '3ds':
            grid_names.append(Flist[i])
    for i, grid_name in enumerate(grid_names):
        Grid = GridProcess(DTF.grid_set_path + grid_name)
        left_lim, right_lim = DTF.give_interp_lim(Grid.bias)
        a = DTF.interp_extrap2(Grid.tck_mean, Grid.bias, Grid.name, left_lim, right_lim, Grid.f_mean)
        # norm = np.mean(np.concatenate((a[norm_lim[i][0]:norm_lim[i][1]], a[-norm_lim[i][1]:-norm_lim[i][0]])))
        norm = np.mean(np.concatenate((a[:20], a[-20:])))
        plt.plot(DTF.bias_ie, a / norm,
                 color=cm.get_cmap('hot')(i/len(grid_names)*0.8))
    plt.savefig(DTF.grid_set_path+'means4.svg')
    # plt.show()


def mean_curve_plot5():
    means, grid_names = [], []

    for i in range(len(Flist)):
        if Flist_split[i][1] == '3ds':
            grid_names.append(Flist[i])

    for i, grid_name in enumerate(grid_names):
        Grid = GridProcess(DTF.grid_set_path + grid_name)
        a = []
        for j in np.linspace(-0.10, -0.09, num=11, endpoint=True):
            a.append(Grid.f_mean(j + Grid.offset))
        for j in np.linspace(0.09, 0.10, num=11, endpoint=True):
            a.append(Grid.f_mean(j + Grid.offset))
        means.append(np.mean(a))
        plt.plot(DTF.bias_interp, Grid.f_mean(DTF.bias_interp + Grid.offset) / np.mean(a),
                 color=cm.get_cmap('hot')(i/len(grid_names)*0.8))
        # Grid.plot_offset()
        # Grid.interp(save_interp=True, map_gap=False, save_interp_sub=True, negative_second_derivative=False)
        # plot_gap_map(grid_set_path + Flist[i])
        # plot_d2f_map(grid_set_path + Flist[i])

    # fig, ax = plt.subplots()
    # plt.plot(range(len(means)), means)
    plt.show()



# mean_curve_plot5()

# 这一段是批量处理grid的
for i in range(len(Flist)):
    if Flist_split[i][1] == '3ds':
        Grid = GridProcess(DTF.grid_set_path + Flist[i], load_modifier='GF0.5')
        Grid.interp(save_itp=True)
        # Grid.all_plot4()

# Grid = GridProcess(grid_set_path+'OD03K.3ds')
# Grid.interp(save_interp=False, map_gap=True, save_interp_sub=False, negative_second_derivative=False)
# plot_gap_map(grid_set_path+'OD03K.3ds')


# Grid.all_plot4()
# Grid.interp(save_interp=True, map_gap=False, save_interp_sub=True, negative_second_derivative=False)

# plot_gap_map(grid_set_path+'OP32K.3ds')
# plot_d2f_map(grid_set_path+'OP32K.3ds')


# fig, ax = plt.subplots()
# norm = [2, 1, 1, 1, 1, 0.8, 0.3]
# CList = os.listdir('grids/08_11_13_16_18_19_21_22_23/avg_curves/')
# for i in range(len(CList)):
#     [Xs, Ys] = open_curve('grids/08_11_13_16_18_19_21_22_23/avg_curves/' + CList[i])
#     ax.plot(Xs, Ys/np.mean(Ys)*norm[i], c=cm.get_cmap('hot')(0.8 - i/len(CList)*0.8), alpha=1)
# ax.set_xlim(-300, 300)
# plt.show()

# print(DTF.bias_interp_extrap[1]-DTF.bias_interp_extrap[0])
# print(DTF.bias_interp_extrap[101]-DTF.bias_interp_extrap[100])


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


# 这一段是使用gaussianFilter
# for i in range(len(Flist)):
#     if Flist_split[i][1] == '3ds':
#         GFed_dIdV = gaussianFilter(DTF.grid_set_path + Flist[i], sig=0.5)
#         np.savez(DTF.grid_set_path+Flist[i][:-4]+'.twisted_GF0.5.npz', GFed_dIdV=GFed_dIdV)

# GFed_dIdV = gaussianFilter('grids/04_08_11_13/04.3ds', sig=0.5)
# np.savez('grids/04_08_11_13/04.twisted_GF0.5.npz', GFed_dIdV=GFed_dIdV)
# Grid = GridProcess('grids/04_08_11_13/04.3ds', load_modifier='GF0.5')
# Grid.interp(save_itp=True)