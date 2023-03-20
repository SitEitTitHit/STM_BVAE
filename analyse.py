# 文件管理
import importlib
import argparse
import os
import re  # string split

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import math  # math.cell used
import random
import heapq  # sorting
from scipy.signal import correlate2d
from scipy import misc, interpolate

import matplotlib.pyplot as plt
from matplotlib.widgets import AxesWidget, Slider, Button, RadioButtons, CheckButtons
from matplotlib import cm, cbook
import createCmp


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


model_index_default = 0
save_index_default = 7
beta_default = 1e-3
grid_set = '9_doping'
grid_set_path = 'grids/04_08_11_13_16_18_19_22_23/'
dpk_default = 'interp_GF05_1'
no_label_model_list = [0]

flag_all = 0
flag_LS_retrieve = 0
flag_LS_filter = 0
flag_LS_profile = 1

flag_LS_distribution = 0
flag_plot_ae_outputs = 0
flag_LS_correlation = 0
flag_LS_waterfall = 0
flag_LS_mapping = 0
flag_map_AC = 0
flag_LS_1sMse = 0

# argument parsers
parser = argparse.ArgumentParser(description='Welcome!')
parser.add_argument('-m', '--mid', default=model_index_default, type=int, help='give models index')
parser.add_argument('-s', '--sid', default=save_index_default, type=int, help='give save index')
parser.add_argument('-k', '--key', default=dpk_default, type=str, help='give data process keyword')
args = parser.parse_args()

model_index = args.mid
save_index = args.sid
data_process_keyword = args.key

grid_names, grid_legends = [], []
Flist = os.listdir(grid_set_path)
Flist_split = [f.split('.') for f in Flist]
for i in range(len(Flist)):
    if Flist_split[i][1] == data_process_keyword:
        grid_names.append(Flist[i])
        grid_legends.append(Flist_split[i][0])


mdl = importlib.import_module('models.VAE_model_' + str(model_index))
dim = mdl.encoded_dim

save_name = 'VAE_model_' + str(model_index) + '_save_' + str(save_index)
save_path = 'saves/save_' + grid_set + '/' + save_name
fig_path = 'figs/fig_' + grid_set + '/' + save_name + '/' + save_name


class FullDataset(Dataset):

    def __init__(self, grid_names):
        super().__init__()
        self.grid_num = len(grid_names)
        self.dIdV = []
        self.lens = []
        self.mean_curve = []
        self.std = []
        self.offset = []
        for grid in grid_names:
            data = np.load(grid_set_path + grid)
            self.bias = data['bias']
            self.dIdV.append(data['dIdV'])
            self.lens.append(len(data['dIdV']))
            self.mean_curve.append(data['mean_curve'])
            self.std.append(data['std'])
            self.offset.append('offset')
            print('Grid ' + grid + ' Loaded')

    def __getitem__(self, index):
        label = 0
        while (True):
            if index < len(self.dIdV[label]):
                dIdV = self.dIdV[label][index]
                break
            else:
                index -= len(self.dIdV[label])
                label += 1
        return dIdV, label

    def __len__(self):
        return sum(self.lens)

    # reimpose mean curve AND normalize with (X-mean)/std
    def add_mean(self, curve, label):
        # label = int(label)  # avoid possible type error
        # # curve = curve + (self.mean_curve[label] - np.mean(self.mean_curve[label])) / self.std[label]
        # curve = curve / self.std[label] * 1e-12
        return curve


# instantiate (load grids) and load model status
data = FullDataset(grid_names)
grid_size = [int(np.sqrt(data.lens[i])) for i in range(data.grid_num)]

VAE = mdl.VAE(input_len=len(data.bias), encoded_space_dim=mdl.encoded_dim)
VAE.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model_dict = torch.load(save_path + '.pth', map_location=device)
VAE.load_state_dict(model_dict['VAE_state_dict'])
print(save_name + ' loaded')

# read encoded latent space data
enc_mu = pd.read_csv(f'./saves/save_{grid_set}/{save_name}_encoded_mu.csv')
# reshape the data into 'Panel' (Multi-index Dataframe)
enc_mu = enc_mu.rename(columns={'Unnamed: 0': 'index'})
enc_mu = enc_mu.set_index(['label', 'index'])
enc_mu_stats = []  # Log statistics of encoded mu

for i in range(data.grid_num):
    enc_stat = []
    for j in range(dim):
        a = np.array(enc_mu.loc[i, str(j)])  # Take the Parameter j of Grid i
        enc_stat.append({'max': max(a), 'min': min(a), 'len': len(a),
                         'mean': np.mean(a), 'std': np.std(a)})
    enc_mu_stats.append(enc_stat)

# Sort the index of encoded mu by its standard deviation
# index 0 for largest std
stds_dim = []
for j in range(dim):
    # stds_grid = []
    # for i in range(data.grid_num):
    #     stds_grid.append(enc_mu_stats[i][j]['std'])
    # stds_dim.append(np.mean(stds_grid))
    stds_dim.append(np.std(np.array(enc_mu[str(j)])))
std_index = np.argsort(stds_dim)[::-1]
# print(std_index)
# std_index 排序了所有这些输入，正向使用index[i]给出第i重要的参数应该给decoder的位置


# https://stackoverflow.com/questions/55095111/displaying-radio-buttons-horizontally-in-matplotlib
class MyRadioButtons(RadioButtons):

    def __init__(self, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="vertical", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([], [], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles: c.set_picker(5)
        self._observers = cbook.CallbackRegistry()
        self.connect_event('pick_event', self._clicked)

    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
                event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))


# input是np array
def decode(paras):
    return data.add_mean(VAE._decode(torch.reshape(torch.Tensor(paras), (1, -1))).detach().numpy().reshape(-1), paras[-1])


def give_mapping_lims(a, mapping_type):
    mean = np.mean(a)
    std = np.std(a)
    if mapping_type=='r':
        return mean-2*std, mean+2*std
    elif mapping_type=='k':
        return mean-0.5*std, mean+3*std
    elif mapping_type=='c':
        return mean-1*std, mean+4*std


def FFT(a):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(a))))


def AutoCorrelation(a):
    return correlate2d(a, a, 'same', 'wrap')


def symmetrize_4fold(a):
    return (a+a[:, ::-1]+a[::-1, :]+a[::-1, ::-1])/4


def PlotAeOutputs(m=data.grid_num, n=8):
    fig0, axs0 = plt.subplots(m, n, figsize=(2 * n, 1.5 * m), squeeze=False)
    fig1, axs1 = plt.subplots(m, n, figsize=(2 * n, 1.5 * m), squeeze=False)
    lens = 0
    for i in range(m):
        for j in range(n):
            # 随机选取编号为0-len(data)中的某一张图
            sample = data[random.randint(lens, lens + data.lens[i])]
            dIdV = torch.Tensor(sample[0])
            label = sample[1]
            with torch.no_grad():
                rec_dIdV = VAE._plot(torch.reshape(dIdV, (1, -1)), torch.reshape(torch.tensor(label), (1, -1)))
                rec_dIdV = rec_dIdV.reshape(-1)

            axs0[i, j].plot(data.bias, data.add_mean(dIdV, label), label='original', c='g')
            axs0[i, j].plot(data.bias, data.add_mean(rec_dIdV, label), label='reconstructed', c='b')
            axs1[i, j].plot(data.bias, dIdV, label='original')
            axs1[i, j].plot(data.bias, rec_dIdV, label='reconstructed')
            # plt.plot(full_dataset.bias, dIdV, label='original')
            # plt.plot(full_dataset.bias, rec_dIdV, label='reconstructed')
            axs0[i, j].get_xaxis().set_visible(False)
            axs0[i, j].get_yaxis().set_visible(False)
            axs1[i, j].get_xaxis().set_visible(False)
            axs1[i, j].get_yaxis().set_visible(False)

        lens += data.lens[i]

    axs0[m - 1, n - 1].legend()
    axs1[m - 1, n - 1].legend()
    fig0.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    fig1.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    fig0.savefig(fig_path + '_output.svg')
    fig1.savefig(fig_path + '_output_2.svg')
    plt.close('all')
    print('AE outputs plotted and saved!')


def LSDistribution(n_rows=2, dim_clipped=8):
    # 这里用subplot()比用subplots()实现来的方便一些哦，因为编号index是1维而不是2维
    n_cols = math.ceil(data.grid_num / n_rows)
    max_std = max([enc_mu_stats[i][std_index[0]]['std'] for i in range(data.grid_num)])

    for i in range(data.grid_num):
        ax = plt.subplot(n_rows, n_cols, i + 1)  # 注意要+1
        for j in range(dim)[::-1]:  # plot from 9 to 0
            # 直方图里切成500条
            j_prime = std_index[j]
            plt.hist(np.array(enc_mu.loc[i, str(j_prime)]), bins=1000, range=(-3,3), histtype='step', label=j,
                     color=cm.get_cmap('gist_rainbow')(j/dim), alpha=0.7)
            # print(f"G{i}L{j_prime} max {enc_mu_stats[i][j_prime]['max']:.6f} min {enc_mu_stats[i][j_prime]['min']:.6f} "
            #       f"std {enc_mu_stats[i][j_prime]['std']:.6f}")
        ax.set_title('Grid 0.'+grid_legends[i])
        ax.set_ylim(0, 1200)
        ax.set_xlim(-6*max_std, 6*max_std)
        ax.tick_params(axis='both', which='both', labelsize=8)
        # 最后一张图加图例
        if i == data.grid_num - 1:
            ax.legend(prop={'size': 8})
    # 为了美观调整一下子图间距
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    fig = plt.gcf()
    fig.set_figheight(3*n_rows)
    fig.set_figwidth(4*n_cols)
    plt.suptitle('Latent Space Histogram')
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    plt.savefig(fig_path + '_LS_distribution.svg')
    plt.close('all')

    for j in range(dim_clipped):
        ax = plt.subplot(n_rows, n_cols, j + 1)  # 注意要+1
        j_prime = std_index[j]
        for i in range(data.grid_num):
            plt.hist(np.array(enc_mu.loc[i, str(j_prime)]), bins=1000, range=(-3,3), histtype='step', label='0.'+grid_legends[i],
                     color=cm.get_cmap('hsv')(i/data.grid_num*0.8), alpha=0.7)
        ax.set_title(f'Para {j}')
        ax.set_ylim(0, 1200)
        ax.set_xlim(-6*max_std, 6*max_std)
        ax.tick_params(axis='both', which='both', labelsize=8)
        # 最后一张图加图例
        if j == data.grid_num - 1:
            ax.legend(prop={'size': 8})
    # 为了美观调整一下子图间距
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    fig = plt.gcf()
    fig.set_figheight(3*n_rows)
    fig.set_figwidth(4*n_cols)
    plt.suptitle('Latent Space Histogram')
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    plt.savefig(fig_path + '_LS_distribution1.svg')
    plt.close('all')

    print('Latent space distribution plotted and saved!')


# real_space_size = (480, 480, 360, 350, 440, 320, 300, 350)  # in Angstrom
# real_space_size = (480, 380, 400, 350, 440, 450, 300, 350)  # in Angstrom
# 不如直接给Qx的位置
Qx_pos = [200, 215, 209.5, 201.5, 205.5, 227.5, 217.5, 186, 193.5]  # the 0th for 0.04 is not accurate
original_grid_size = [200, 256, 256, 256, 256, 281, 256, 256, 256]
Qx = [Qx_pos[i]-original_grid_size[i]/2 for i in range(data.grid_num)]
zoom_in_factor = 1.3
FFT_lim = [[0.5*grid_size[i]*(1-zoom_in_factor*2*Qx[i]/original_grid_size[0])-0.5,
            0.5*grid_size[i]*(1+zoom_in_factor*2*Qx[i]/original_grid_size[0])-0.5] for i in range(data.grid_num)]


def LSMapping(dim_clipped=8, flag_map_AC=True):
    # 用subplots实现
    colour_gradient_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    fig0, axs0 = plt.subplots(data.grid_num, dim_clipped, figsize=(25, 25), squeeze=False)
    fig1, axs1 = plt.subplots(data.grid_num, dim_clipped, figsize=(25, 25), squeeze=False)
    if flag_map_AC:
        fig2, axs2 = plt.subplots(data.grid_num, dim_clipped, figsize=(25, 25), squeeze=False)

    def ax_adjust(ax, i, j):
        ax.title.set_size(8)
        ax.set_xlabel('para'+str(j), labelpad=0.03)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('0.'+grid_legends[i], labelpad=0.05)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.invert_yaxis()

    for i in range(data.grid_num):
        for j in range(dim_clipped):
            j_prime = std_index[j]
            topo = np.array(enc_mu.loc[i, str(j_prime)]).reshape((grid_size[i], -1))  # 取grid i的第j个参数
            vmin, vmax = give_mapping_lims(topo, 'r')
            axs0[i, j].imshow(topo, vmin=vmin, vmax=vmax)
            ax_adjust(axs0[i, j], i, j)

            f_shift = FFT(topo)
            if data_process_keyword == 'interp_twisted':
                f_shift = symmetrize_4fold(f_shift)
            vmin, vmax = give_mapping_lims(f_shift, 'k')
            axs1[i, j].imshow(f_shift, cmap='afmhot', vmin=vmin, vmax=vmax)
            axs1[i, j].set_xlim(FFT_lim[i][0], FFT_lim[i][1])
            axs1[i, j].set_ylim(FFT_lim[i][0], FFT_lim[i][1])
            ax_adjust(axs1[i, j], i, j)

            if flag_map_AC:
                ac = AutoCorrelation(topo)
                vmin, vmax = give_mapping_lims(ac, 'c')
                axs2[i, j].imshow(ac, cmap='viridis', vmin=vmin, vmax=vmax)
                ax_adjust(axs2[i, j], i, j)
                print(f'{i*dim_clipped+j+1}/{data.grid_num*dim_clipped} done!')

    if data_process_keyword == 'interp_twisted':
        print('FFT symmetrization done')
    fig0.suptitle('Latent Space Mapping')
    fig1.suptitle('Latent Space Mapping FFT')
    fig0.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, hspace=0.05, wspace=0.05)
    fig1.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, hspace=0.05, wspace=0.05)
    fig0.savefig(fig_path + '_LS_mapping.svg')
    fig1.savefig(fig_path + '_LS_mapping_FFT.svg')

    if flag_map_AC:
        fig2.suptitle('Latent Space Mapping AC')
        fig2.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, hspace=0.05, wspace=0.05)
        fig2.savefig(fig_path + '_LS_mapping_AC.svg')
    plt.close('all')
    print('Latent space mapping & FFT & AC plotted and saved!')


def LSWaterfall(var_index: int, wtf_offset=0, grid_offset=2, cutoff=15):

    fig, ax = plt.subplots()
    var_index_prime = std_index[var_index]
    with torch.no_grad():
        for i in range(data.grid_num):
            paras = []
            for j in range(dim):
                paras.append(enc_mu_stats[i][j]['mean'])
            if not model_index == 5:
                paras.append(i)
            legend_flag = False  # 画图例辅助flag

            cmp = createCmp.createLightnessMap(cm.get_cmap('hsv')(i/data.grid_num*0.8))

            stats = enc_mu_stats[i][var_index_prime]  # 仅仅为了方便调用
            lower_bound = stats['mean'] - 1.5 * stats['std']
            span = 3 * stats['std']

            for k in np.linspace(0, 1, 15, endpoint=True):
                paras[var_index_prime] = k * span + lower_bound
                curve = decode(paras) + (k*wtf_offset+i*grid_offset)*0.1  # 做图的offset
                # cutoff切掉边边比较好看
                # 比中间值大的那条线定颜色，利用legend_flag
                if legend_flag or (k <= 0.5):
                    # 为真直接画图
                    ax.plot(data.bias[cutoff:-cutoff], curve[cutoff:-cutoff],
                            c=cmp(k / 2 + 0.25), alpha=0.8)
                else:
                    # 否则>=0的第一个记legend
                    legend_flag = True
                    ax.plot(data.bias[cutoff:-cutoff], curve[cutoff:-cutoff],
                            c=cmp(k / 2 + 0.25), alpha=0.8,
                            label='0.'+grid_legends[i])
        ax.legend()
        plt.title(f'Latent Space Parameter {var_index}')
    plt.savefig(fig_path + f'_waterfall{var_index}.svg')
    plt.close('all')


def LSWaterfall_all(n_rows=2, dim_clipped=8, wtf_offset=0, grid_offset=2, cutoff=1):
    n_cols = math.ceil(dim_clipped / n_rows)
    with torch.no_grad():
        for var_index in range(dim_clipped):
            var_index_prime = std_index[var_index]
            ax = plt.subplot(n_rows, n_cols, var_index + 1)
            for i in range(data.grid_num):
                paras = []
                for j in range(dim):
                    paras.append(enc_mu_stats[i][j]['mean'])
                if not model_index in no_label_model_list:
                    paras.append(i)

                cmp = createCmp.createLightnessMap(cm.get_cmap('hsv')(i/data.grid_num*0.8))

                stats = enc_mu_stats[i][var_index_prime]
                lower_bound = stats['mean'] - 1.5 * stats['std']
                span = 3 * stats['std']

                for k in np.linspace(0, 1, 15, endpoint=True):
                    paras[var_index_prime] = k * span + lower_bound
                    curve = decode(paras) + (k*wtf_offset+i*grid_offset)*0.1  # 做图的offset

                    if not k == 1:
                        ax.plot(data.bias[cutoff:-cutoff], curve[cutoff:-cutoff],
                                c=cmp(k / 2 + 0.25), alpha=0.7)
                    else:
                        ax.plot(data.bias[cutoff:-cutoff], curve[cutoff:-cutoff],
                                c=cmp(k / 2 + 0.25), alpha=0.7, label='0.'+grid_legends[i])
                ax.set_title(f'Para {var_index}', fontsize=10)
                ax.tick_params(axis='both', which='both', labelsize=8)

        ax.legend(prop={'size': 8})
    fig = plt.gcf()
    fig.set_figheight(3.5*n_rows)
    fig.set_figwidth(4*n_cols)
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.01, right=0.99, hspace=0.15, wspace=0.1)
    plt.savefig(fig_path + f'_waterfall_all.svg')
    plt.close('all')
    print('Latent space waterfall plotted and saved!')


def LSCorrelation(g, n, limit=3):
    # n-1*n-1 triangle
    fig, axs = plt.subplots(n - 1, n - 1, figsize=(2.5 * (n - 1), 2.5 * (n - 1)))
    for pi in range(n - 1):
        for pj in range(n - 1 - pi):
            pi_prime = std_index[pi]
            pj_prime = std_index[pj + pi + 1]
            # note i for y and j for x
            ax = axs[pi, pj + pi]
            a, b = enc_mu.loc[g, str(pj_prime)], enc_mu.loc[g, str(pi_prime)]
            ax.plot(a, b, '.', ms=0.3, alpha=0.4, c=cm.get_cmap('hsv')(g/data.grid_num*0.8), rasterized=True)
            ax.text(0.1-limit, 0.1-limit, 'Corr = {:.3f}'.format(np.corrcoef(a, b)[0][1]))
            ax.set_ylim(-limit, limit)
            ax.set_xlim(-limit, limit)
            ax.axvline(0, alpha=0.8, c='grey', ls='--')
            ax.axhline(0, alpha=0.8, c='grey', ls='--')
            if pi == 0:
                ax.set(xlabel='para ' + str(pj + pi + 1))
                ax.xaxis.set_label_position('top')
            if pj == 0:
                ax.set(ylabel='para ' + str(pi))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    # hide the rest plot
    for pi in range(n - 2):
        for pj in range(n - 2 - pi):
            axs[-pi - 1, pj].axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(fig_path + '_G' + str(g) + '_' + 'LS_correlation.svg')
    plt.close('all')


def LS1sMse(offset_grid=0.025, offset_dim=2):
    fig, ax = plt.subplots()
    for j in range(dim):
        ax.axvline(j * offset_dim, alpha=0.8, c='gray', ls='--', lw=0.5)

    for i in range(data.grid_num):
        cmp = createCmp.createLightnessMap(cm.get_cmap('hsv')(i/data.grid_num*0.8))
        for j in range(dim):
            j_prime = std_index[j]
            m = enc_mu_stats[i][j_prime]['mean']
            s = enc_mu_stats[i][j_prime]['std']
            p = np.array([-s, 0, s])
            mse = []
            paras = []
            for j0 in range(dim):
                paras.append(enc_mu_stats[i][j0]['mean'])
            if not model_index in no_label_model_list:
                paras.append(i)
            y0 = decode(paras)
            for pi in p:
                paras[j_prime] = pi + m
                y = decode(paras)
                # copysign(x, y) return x with sign of y
                mse.append(math.copysign(np.square(np.subtract(y, y0)).mean(), pi))

            mse = np.array(mse)
            plt.plot(p + j * offset_dim + m, mse + i * offset_grid, '-o', c=cmp(1 - j / dim), ms='3')

    ax.set_ylabel('Grids')
    ax.set_xlabel('Paras')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    fig.savefig(fig_path + '_LS_1sMse.svg')
    plt.close('all')
    print('Latent space 1-sigma MSE plotted and saved!')


'''
def plot_mean_curve():
    fig, ax = plt.subplots()
    for i in grid_selection:
        plt.plot(data.mean_curve[i], label='grids '+str(i))
        ax.legend()
'''


def MultipleMap(g, p):
    p_prime = std_index[p]
    a = []
    bias = 20  # in meV
    a.append(np.array(enc_mu.loc[g, str(p_prime)]))
    a.append(np.load(grid_set_path + grid_legends[g] + '.d2f.npz')['d2f'])
    # a.append(np.load(grid_set_path + grid_legends[g] + '.interp.npz')['dIdV'].T[70 + bias])
    title = ['para' + str(p), 'd2f']
    # title = ['para' + str(p), 'd2f', 'bias' + str(bias)]

    cols = len(a)
    fig, axs = plt.subplots(1, cols, figsize=(2.5 * cols, 2.5 + 0.5))
    for i in range(len(a)):
        a[i] = a[i].reshape((grid_size[g], -1))
        axs[i].set_title(title[i])
        axs[i].invert_yaxis()
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    axs[0].imshow(a[0])
    axs[1].imshow(a[1], vmax=6e-7)
    # axs[2].imshow(a[2])
    # axs[2].imshow(a[2], vmin=-7e-12, vmax=0)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.8, hspace=0.05, wspace=0.05)
    fig.suptitle(grid_legends[g])
    fig.savefig(fig_path + '_G' + str(g) + '_' + '-'.join(title) + '.svg')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(a[0], a[1], '.', ms=0.3, alpha=0.4, c='b', rasterized=True)
    ax1.set_ylim(0, 6e-7)
    plt.xlabel(title[0])
    ax1.set_ylabel(title[1])

    # ax2 = ax1.twinx()
    # ax2.plot(a[0], a[2], '.', ms=0.3, alpha=0.4, c='r', rasterized=True)
    # # ax2.set_ylim(-7e-12, 0)
    # ax2.set_ylabel(title[2])

    fig.savefig(fig_path + '_G' + str(g) + '_' + '-'.join(title) + '_2.svg')
    plt.close('all')
    plt.show()


# 可以考虑在这里加上一段直接比较谱线MSE的
def data_index_retrieve(paras, process_batch=2**16, rtv_num: int=1):
    # retrieve the index of original data whose paras are most similar to the paras given
    # 似乎这句话没什么用
    with torch.no_grad():
        # error = []
        # paras_tile = (torch.Tensor(np.tile(paras, (batch_size, 1)))).to(device)
        # # 把paras也堆成对应的形状
        # for i in range(math.ceil(len(data)/batch_size)-1):
        #     batch = []
        #     for j in range(batch_size):
        #         batch.append(list(enc_mu.iloc[i*batch_size+j]))
        #     batch = (torch.Tensor(batch)).to(device)
        #     error.append(torch.mean(torch.square(torch.add(batch, paras_tile, alpha=-1)), 1).cpu())
        #
        # i = i + 1
        # batch = []
        # paras_tile = (torch.Tensor(np.tile(paras, (len(data)-i*batch_size, 1)))).to(device)
        # for j in range(len(data)-i*batch_size):
        #     batch.append(list(enc_mu.iloc[i*batch_size+j]))
        # batch = (torch.Tensor(batch)).to(device)
        # error.append(torch.mean(torch.square(torch.add(batch, paras_tile, alpha=-1)), 1).cpu())
        error = []
        paras_tile = (torch.Tensor(np.tile(paras, (process_batch, 1)))).to(device)
        # 把paras也堆成对应的形状
        for i in range(math.ceil(len(data)/process_batch)-1):
            batch = (torch.from_numpy(np.array(enc_mu.iloc[i*process_batch:(i+1)*process_batch]))).to(device)
            error.append(torch.mean(torch.square(torch.add(batch, paras_tile, alpha=-1)), 1).cpu())

        i = i + 1
        paras_tile = (torch.Tensor(np.tile(paras, (len(data)-i*process_batch, 1)))).to(device)
        batch = (torch.from_numpy(np.array(enc_mu.iloc[i*process_batch:]))).to(device)
        error.append(torch.mean(torch.square(torch.add(batch, paras_tile, alpha=-1)), 1).cpu())

    error = np.array(torch.cat(error))

    if rtv_num==1: return np.argmin(error)
    else: return heapq.nsmallest(rtv_num, range(len(error)), key=error.__getitem__)


def LSRetrieve():
    with torch.no_grad():
        paras = np.zeros(dim)
        fig, ax = plt.subplots()
        curve = decode(paras)
        line, = ax.plot(data.bias, curve, 'g')
        ax.set_xlabel('Bias(V)')
        ax.set_ylabel('y(a.u.)')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')

        # Adjust the subplots region to leave some space for the sliders and buttons
        fig.subplots_adjust(left=0.15, bottom=0.35)

        axis, slider = [], []
        for i in range(dim):
            # axis vessels: label, valmin, valmin, valinit, etc.
            axis.append(plt.axes([0.25, 0.30 - i * 0.03, 0.5, 0.03], label='l' + str(i)))
            # plt.axes：Add an axes to the current figure and make it the current axes.
            slider.append(Slider(axis[i], 'para' + str(i),
                                 -3*stds_dim[std_index[i]], 3*stds_dim[std_index[i]], 0, valfmt='% .2f'))

        # 用于放置raw data的para分量，这样可以整个ax clear刷新掉不会堆积
        ax_label = plt.axes([0,0,0,0])

        def sliders_update(val):
            for i in range(dim):
                paras[std_index[i]] = slider[i].val
            ax.clear()
            curve = decode(paras)
            ax.plot(data.bias, curve, 'g')
            ax_label.clear()

        def button_update(event):
            ax.clear()
            ax.plot(data.bias, decode(paras), 'g')
            index = data_index_retrieve(paras)
            curve_raw = data[index][0]
            ax.plot(data.bias, curve_raw, 'b')
            ax_label.clear()
            for i in range(dim):
                ax_label.annotate(round(enc_mu.iloc[index][std_index[i]], 2),
                                  (0.83, 0.307 - i * 0.03), xycoords='figure fraction')

        for i in range(dim):
            slider[i].on_changed(sliders_update)

        ax_retrieve = plt.axes([0.9, 0.05, 0.05, 0.05], label='RT')
        btn = Button(ax_retrieve, 'RT')
        btn.on_clicked(button_update)
        # ！！！如果有button plt.show()一定要紧跟在后面，我也不知道为啥
        plt.show()


def ls_channel_calcu(grid_index=5, channels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], batch_size=2048):

    VAE.to(device)
    channels = torch.Tensor(channels)
    with torch.no_grad():
        decoded_data = []
        start_index = sum(data.lens[:grid_index])
        for i in range(math.ceil(data.lens[grid_index]/batch_size)-1):
            batch = []
            for j in range(batch_size):
                batch.append(list(enc_mu.loc[grid_index, start_index+i*batch_size+j]))
            batch = torch.Tensor(batch) * channels
            batch = batch.to(device)
            decoded_data.append(VAE._decode(batch).cpu())

        i = i+1
        batch = []
        for j in range(data.lens[grid_index]-i*batch_size):
            batch.append(list(enc_mu.loc[grid_index, start_index+i*batch_size+j]))
        batch = torch.Tensor(batch) * channels
        # print(batch.cpu())
        batch = batch.to(device)
        decoded_data.append(VAE._decode(batch).cpu())

    decoded_data = torch.cat(decoded_data)
    print('Calculation Finished!!!')
    return np.array(decoded_data)


def LSFilter():
    paras = np.zeros(dim)
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 7))
    grid_index = 0
    channels = [0]*dim
    decoded_data = 0
    bias_index = 0

    def ax_adjust(ax):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()

    def init():
        for axi in axs:
            for axj in axi:
                axj.clear()
                ax_adjust(axj)

    def plot_refresh():
        init()

        a = np.array(np.load(grid_set_path + grid_legends[grid_index] + '.' + data_process_keyword + '.npz')['dIdV']
                     )[:, bias_index].reshape((grid_size[grid_index], -1))
        vmin, vmax = give_mapping_lims(a, 'r')
        axs[0,0].imshow(a, vmin=vmin, vmax=vmax)

        a = FFT(a)
        if data_process_keyword == 'interp_twisted':
            a = symmetrize_4fold(a)
        vmin, vmax = give_mapping_lims(a, 'k')
        axs[0,1].imshow(a, vmin=vmin, vmax=vmax, cmap='afmhot')

        a = np.array(decoded_data)[:, bias_index].reshape((grid_size[grid_index], -1))
        vmin, vmax = give_mapping_lims(a, 'r')
        axs[1,0].imshow(a, vmin=vmin, vmax=vmax)

        a = FFT(a)
        if data_process_keyword == 'interp_twisted':
            a = symmetrize_4fold(a)
        vmin, vmax = give_mapping_lims(a, 'k')
        axs[1,1].imshow(a, vmin=vmin, vmax=vmax, cmap='afmhot')

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(left=0.20, bottom=0.1, right=0.90, top=0.9, hspace=0.05, wspace=0.05)

    rax = fig.add_axes([0.05, 0.5, 0.1, 0.4])
    rlabels = ['0.' + grid_legends[i] for i in range(data.grid_num)]
    radio = RadioButtons(rax, rlabels)
    def grid_select(label):
        nonlocal grid_index
        grid_index = rlabels.index(label)
    radio.on_clicked(grid_select)

    cax = fig.add_axes([0.05, 0.1, 0.1, 0.4])
    clabels = ['p'+str(i) for i in range(dim)]
    check = CheckButtons(cax, clabels)
    def channel_select(label):
        # 取反打X的那个channel
        nonlocal channels
        channels[std_index[clabels.index(label)]] = int(not bool(channels[std_index[clabels.index(label)]]))
    check.on_clicked(channel_select)

    sax = fig.add_axes([0.2, 0.03, 0.32, 0.05])
    slider = Slider(sax, 'bias', valmin=data.bias[0], valmax=data.bias[-1], valinit=0, valfmt='% .3f')
    def sliders_update(val):
        nonlocal bias_index
        bias_index = (np.abs(data.bias-val)).argmin()
        plot_refresh()
    slider.on_changed(sliders_update)

    bax = fig.add_axes([0.05, 0.03, 0.1, 0.05])
    btn = Button(bax, 'Cal')
    def calcu(event):
        nonlocal decoded_data
        print(grid_index, channels)
        decoded_data = ls_channel_calcu(grid_index=grid_index, channels=channels)
        plot_refresh()
    btn.on_clicked(calcu)

    init()

    plt.show()


def LSProfile(dim_clipped=8, para_lim=3, wtf_count=50, wtf_offset=80):

    # profile切线颜色
    cmp_key = 'plasma'
    cl = cm.get_cmap(cmp_key)(0); cr = cm.get_cmap(cmp_key)(0.999)
    # 记录每个linecut起点终点的取值
    p0 = np.zeros(dim)
    p1 = np.zeros(dim)
    # 记录当前选中的两个index(排序之前，0不是最大！)
    x_idx = std_index[0]; y_idx = std_index[1]
    rtv_colors = [cm.get_cmap(cmp_key)(i/(wtf_count-1)) for i in range(wtf_count)]  # 提前算好这些xxx点的颜色

    def plot_correlation():
        ax = axes[0]
        ax.clear()
        # 画correlation
        for g in range(data.grid_num):
            a, b = enc_mu.loc[g, str(x_idx)], enc_mu.loc[g, str(y_idx)]
            ax.plot(a, b, '.', ms=0.2, alpha=0.2, c=cm.get_cmap('hsv')(g/data.grid_num*0.8), rasterized=True)
        # ax.text(0.1-para_lim, 0.1-para_lim, 'Corr = {:.3f}'.format(np.corrcoef(a, b)[0][1]))
        ax.axvline(0, alpha=0.8, c='grey', ls='--')
        ax.axhline(0, alpha=0.8, c='grey', ls='--')

        ax.set_ylim(-para_lim, para_lim)
        ax.set_xlim(-para_lim, para_lim)
        ax.set_aspect(1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    def plot_cutline():
        # 画linecut的起点和终点
        # 若parases_rtv不为空，画出对应的点xxx
        ax = axes[1]
        ax.clear()
        xs = [p0[x_idx], p1[x_idx]]; ys = [p0[y_idx], p1[y_idx]]
        ax.set_facecolor('none')
        ax.plot(xs, ys, c='black')
        ax.scatter(xs, ys, marker='o', c=[cl, cr], zorder=99)  # 用z_order强行抬升图层

        ax.set_ylim(-para_lim, para_lim)
        ax.set_xlim(-para_lim, para_lim)
        ax.set_aspect(1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        if len(parases_rtv) != 0:
            ax.scatter(parases_rtv[x_idx], parases_rtv[y_idx], marker='x', c=rtv_colors, zorder=100)

    def plot_wtf(clr='cl1', d2f_mark=False):
        # 画waterfall plot
        ax = axes[2]
        ax.clear()
        # paras的list，先用linespace列出来，再转置一下，为了代码简练
        parases = [np.linspace(p0[i], p1[i], wtf_count, endpoint=True) for i in range(dim)]
        parases = np.array(parases).T
        for i, paras in enumerate(parases):
            offset = (i/wtf_count*wtf_offset)*0.1
            curve = decode(paras)+offset
            if clr=='cl1':
                ax.plot(data.bias, curve, c='black', alpha=0.7)
            else:
                ax.plot(data.bias, curve, c=cm.get_cmap(cmp_key)(i/(wtf_count-1)), alpha=0.7)

            # 先不打包成函数了先塞在这里
            if d2f_mark:
                tck = interpolate.splrep(data.bias, curve)
                def f(x): return interpolate.splev(x, tck)
                d2f = np.zeros(len(data.bias))
                for j, x in enumerate(data.bias):
                    d2f[j] = - misc.derivative(f, x, dx=5e-3, n=2)
                # 不能直接全域取最大，还是得稍微处理一下最初最末的点
                d2f_range = 5
                middle = int(len(data.bias)/2)
                arg_l = np.argmax(d2f[d2f_range:middle])
                arg_r = np.argmax(d2f[middle:-d2f_range])+middle
                ax.scatter(data.bias[arg_l], f(data.bias[arg_l]), c='b', s=5)
                ax.scatter(data.bias[arg_r], f(data.bias[arg_r]), c='b', s=5)


    # 暂存indices和parases_rtv，再多写一个flag_recal节省运算
    indices, parases_rtv = [], []
    def rtv_wtf(flag_recal=True, clr='cl1'):
        ax = axes[3]
        ax.clear()
        # paras的list，先用linespace列出来，再转置一下，为了代码简练
        parases = [np.linspace(p0[i], p1[i], wtf_count, endpoint=True) for i in range(dim)]
        parases = np.array(parases).T
        nonlocal indices, parases_rtv

        if flag_recal:
            indices, parases_rtv = [], []
        for i, paras in enumerate(parases):
            # index = data_index_retrieve(paras)
            # curve_raw = data[index][0]+(i/wtf_count*wtf_offset)*0.1
            # parases_rtv.append(enc_mu.iloc[index])

            # 上面是之前一条线的版本，现在改成五/十条线一下
            # indices(dim, rtv_num)是原始线的序号，再以dim维度堆叠
            if flag_recal:
                indices.append(data_index_retrieve(paras, rtv_num=5))

                paras_rtv = [enc_mu.iloc[index] for index in indices[i]]
                paras_rtv = np.mean(paras_rtv, axis=0)
                parases_rtv.append(paras_rtv)

            curve_raw = [data[index][0] for index in indices[i]]
            curve_raw = np.mean(curve_raw, axis=0)
            curve_raw = curve_raw+(i/wtf_count*wtf_offset)*0.1

            # 取出是第几个grid的
            labels = [data[index][1] for index in indices[i]]

            if clr=='cl1':
                ax.plot(data.bias, curve_raw, c='black', alpha=0.7)
                ax.text(0.08, curve_raw[-5], str(labels), fontsize=5)
            else:
                ax.plot(data.bias, curve_raw, c=cm.get_cmap(cmp_key)(i/(wtf_count-1)), alpha=0.7)

        if flag_recal:
            parases_rtv = np.array(parases_rtv).T
        # 调用plot_cutline把这些点map回去
        plot_cutline()

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    plot_correlation()
    plot_cutline()
    plot_wtf()

    # Adjust the subplots region to leave some space for the sliders and buttons
    # fig.subplots_adjust(left=0.1, bottom=0.1)
    axes[0].set_position([0.046, 0.15, 0.233, 0.70])
    axes[1].set_position([0.046, 0.15, 0.233, 0.70])
    # ax0, ax1是两个位置重合的ax分别装correlation plot和linecut
    axes[2].set_position([0.333, 0.05, 0.313, 0.90])
    axes[3].set_position([0.666, 0.05, 0.313, 0.90])
    rax_x = fig.add_axes([0.046, 0.90, 0.233, 0.06])
    rax_y = fig.add_axes([0.006, 0.15, 0.026, 0.70])
    sax_x0 =fig.add_axes([0.046, 0.08, 0.233, 0.02])
    sax_x1 =fig.add_axes([0.046, 0.04, 0.233, 0.02])
    sax_y0 =fig.add_axes([0.293, 0.15, 0.006, 0.70])
    sax_y1 =fig.add_axes([0.306, 0.15, 0.006, 0.70])

    bax_cl0=fig.add_axes([0.333, 0.05, 0.020, 0.03])
    bax_cl1=fig.add_axes([0.353, 0.05, 0.020, 0.03])
    bax_rtv=fig.add_axes([0.666, 0.05, 0.020, 0.03])
    bax_contour = fig.add_axes([0.666, 0.92, 0.020, 0.03])
    bax_s = fig.add_axes([0.006, 0.90, 0.013, 0.06])
    bax_l = fig.add_axes([0.020, 0.90, 0.012, 0.06])

    rlabels = ['p' + str(i) for i in range(dim_clipped)]
    kwargs = {'handlelength': 0}
    radio_x = MyRadioButtons(rax_x, rlabels, size=20, active=0, orientation="horizontal", **kwargs)
    kwargs = {'handlelength': 0}
    radio_y = MyRadioButtons(rax_y, rlabels, size=20, active=1, **kwargs)

    def radio_x_update(label):
        nonlocal x_idx; x_idx = std_index[rlabels.index(label)]
        print('index_x =', np.argwhere(std_index==x_idx), 'index_y =', np.argwhere(std_index==y_idx))
        x0 = p0[x_idx]; x1 = p1[x_idx]
        slider_x0.eventson = False; slider_x1.eventson = False
        # 这一句是为了不让后面set_val调用slider_update影响速度
        slider_x0.set_val(x0)
        slider_x1.set_val(x1)
        # 这样写是有bug的，第2个update会晚，原因不知道
        # slider_x0.set_val(p0[x_idx])
        # slider_x1.set_val(p1[x_idx])
        slider_x0.eventson = True; slider_x1.eventson = True
        plot_correlation()
        plot_cutline()

    def radio_y_update(label):
        nonlocal y_idx; y_idx = std_index[rlabels.index(label)]
        print('index_x =', np.argwhere(std_index==x_idx), 'index_y =', np.argwhere(std_index==y_idx))
        y0 = p0[y_idx]; y1 = p1[y_idx]
        slider_y0.eventson = False; slider_y1.eventson = False
        slider_y0.set_val(y0)
        slider_y1.set_val(y1)
        slider_y0.eventson = True; slider_y1.eventson = True
        plot_correlation()
        plot_cutline()

    radio_x.on_clicked(radio_x_update)
    radio_y.on_clicked(radio_y_update)

    slider_x0 = Slider(sax_x0, 'x0', valmin=-para_lim, valmax=para_lim, valinit=0, valfmt='% .2f', color=cl)
    slider_x1 = Slider(sax_x1, 'x1', valmin=-para_lim, valmax=para_lim, valinit=0, valfmt='% .2f', color=cr)
    slider_y0 = Slider(sax_y0, 'y0', valmin=-para_lim, valmax=para_lim, valinit=0, valfmt='% .2f', color=cl, orientation='vertical')
    slider_y1 = Slider(sax_y1, 'y1', valmin=-para_lim, valmax=para_lim, valinit=0, valfmt='% .2f', color=cr, orientation='vertical')

    def sliders_update(val):
        p0[x_idx] = slider_x0.val; p1[x_idx] = slider_x1.val
        p0[y_idx] = slider_y0.val; p1[y_idx] = slider_y1.val
        print('sliders updated')
        plot_cutline()
        plot_wtf()

    slider_x0.on_changed(sliders_update)
    slider_x1.on_changed(sliders_update)
    slider_y0.on_changed(sliders_update)
    slider_y1.on_changed(sliders_update)

    button_rtv = Button(bax_rtv, 'rtv')
    def button_rtv_update(event): rtv_wtf()
    button_rtv.on_clicked(button_rtv_update)

    def ax_adjust(ax, i):
        ax.title.set_size(8)
        ax.set_xlabel('0.'+grid_legends[i], labelpad=0.05)
        ax.xaxis.set_label_position('top')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.invert_yaxis()

    button_ctr = Button(bax_contour, 'ctr')
    def button_ctr_update(event):

        # 计算得到(gridsize^2, dim)和(dim)的内积
        def value(i):
            return np.dot(enc_mu.loc[i], p1-p0)

        # 先在这里写3*3
        n_rows = 3
        n_cols = math.ceil(data.grid_num/n_rows)
        fig1 = plt.figure(figsize=(3*n_cols, 3*n_rows))
        for i in range(data.grid_num):
            ax = plt.subplot(n_rows, n_cols, i+1)  # 注意要+1
            topo = np.array(value(i)).reshape((grid_size[i], -1))  # 取grid i的第j个参数
            vmin, vmax = give_mapping_lims(topo, 'r')
            ax.imshow(topo, vmin=vmin, vmax=vmax)
            ax_adjust(ax, i)
        fig1.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, hspace=0.05, wspace=0.05)
        plt.show()
    button_ctr.on_clicked(button_ctr_update)

    button_s = Button(bax_s, 'sv')
    def button_s_update(event):
        print('before sorted', [list(p0), list(p1)])
        # 应该有更直接的函数
        sorted_p0 = [p0[std_index[i]] for i in range(dim)]
        sorted_p1 = [p1[std_index[i]] for i in range(dim)]
        print([sorted_p0, sorted_p1])
        # 现在输出的是排序过后的版本了
    button_s.on_clicked(button_s_update)

    button_l = Button(bax_l, 'ld')
    def button_l_update(event):
        string_p = input('input p0 & p1: ')
        string_p = list(filter(None, re.split(',|\s|\[|\]', string_p)))
        p0_temp, p1_temp = np.zeros(dim), np.zeros(dim)
        for i in range(dim):
            p0_temp[std_index[i]] = float(string_p[i])
            p1_temp[std_index[i]] = float(string_p[i+dim])
        print('reverse sorted', [list(p0_temp), list(p1_temp)])
        if len(p0_temp)==dim and len(p1_temp)==dim:
            nonlocal p0, p1
            p0 = p0_temp; p1 = p1_temp
            plot_cutline()
            plot_wtf()
            x0 = p0[x_idx]; x1 = p1[x_idx]; y0 = p0[y_idx]; y1 = p1[y_idx]
            slider_x0.eventson = False; slider_x1.eventson = False
            slider_y0.eventson = False; slider_y1.eventson = False
            slider_x0.set_val(x0)
            slider_x1.set_val(x1)
            slider_y0.set_val(y0)
            slider_y1.set_val(y1)
            slider_x0.eventson = True; slider_x1.eventson = True
            slider_y0.eventson = True; slider_y1.eventson = True
        else:
            print('input failed!')
    button_l.on_clicked(button_l_update)

    button_cl0 = Button(bax_cl0, 'cl0')
    def button_cl0_update(event):
        plot_wtf(clr='cl0')
        rtv_wtf(flag_recal=False, clr='cl0')
    button_cl0.on_clicked(button_cl0_update)

    button_cl1 = Button(bax_cl1, 'cl1')
    def button_cl1_update(event):
        plot_wtf(clr='cl1')
        rtv_wtf(flag_recal=False, clr='cl1')
    button_cl1.on_clicked(button_cl1_update)

    # ！！！如果有button plt.show()一定要紧跟在后面，我也不知道为啥
    plt.show()


if flag_plot_ae_outputs or flag_all: PlotAeOutputs()

if flag_LS_distribution or flag_all: LSDistribution()

if flag_LS_waterfall or flag_all:
    # for i in range(dim):
    #     LSWaterfall(i)
    # print('Latent space waterfall plotted and saved!')
    LSWaterfall_all()

if flag_LS_mapping or flag_all: LSMapping(flag_map_AC=(flag_map_AC or flag_all))

if flag_LS_correlation or flag_all:
    for i in range(data.grid_num):
        LSCorrelation(i, 8)
    print('Latent correlations plotted and saved!')

if flag_LS_1sMse or flag_all: LS1sMse()

if flag_LS_retrieve: LSRetrieve()

if flag_LS_filter: LSFilter()

if flag_LS_profile: LSProfile()
