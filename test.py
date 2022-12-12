import numpy as np
from numpy import fft
from scipy import interpolate
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider, Button
from matplotlib import cm, patches
import os
import pandas as pd
import random
import torch
import math  # 用到cell以取整
from torch.utils.data import random_split
from torch.utils.data import Dataset
import importlib
import argparse
import nanonispy
import re

# grid_set_path = 'grids/08_11_13_16_18_19_21_22_23/'
# x = np.load(grid_set_path + '08.interp.npz')['dIdV']
# print(x[0])


# def plotLSColormap(cmp):
#     rgba = cmp(np.linspace(0, 1, 256))
#     fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
#     col = ['r', 'g', 'b']
#     for i in range(3):
#         ax.plot(np.arange(256) / 256, rgba[:, i], color=col[i])
#     ax.set_xlabel('index')
#     ax.xaxis.set_label_position('top')
#     ax.set_ylabel('RGB')
#     plt.setp(ax.get_xticklabels(), visible=False)
#     plt.setp(ax.get_yticklabels(), visible=False)
#     ax.tick_params(axis='both', which='both', length=0)
#
#     gradient = np.linspace(0, 1, 256)
#     gradient = np.vstack((gradient, gradient))
#     cax = plt.axes([0.1, 0.05, 0.8, 0.04])
#     cax.set_axis_off()
#     cax.imshow(gradient, aspect='auto', cmap=cmp)
#     plt.subplots_adjust(bottom=0.1, top=0.85, left=0.1, right=0.9)
#     plt.show()
#
#
# cdict0 = {'red':   [[0.0,  0.0, 0.0],
#                    [0.5,  1.0, 1.0],
#                    [1.0,  1.0, 1.0]],
#           'green': [[0.0,  0.0, 0.0],
#                    [0.25, 0.0, 0.0],
#                    [0.75, 1.0, 1.0],
#                    [1.0,  1.0, 1.0]],
#           'blue':  [[0.0,  0.0, 0.0],
#                    [0.5,  0.0, 0.0],
#                    [1.0,  1.0, 1.0]]}
#
#
# newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict0, N=256)
# plotLSColormap(newcmp)

# a0 = 1e-3
# a2 = 1e-2
#
# def f(x): return a0 * x ** 3 + a2 * x
#
# b0 = np.real(np.roots([a0, 0, a2, -200e-3])[2])
# a = np.linspace(-b0, b0, 201, endpoint=True)
# plt.plot(f(a), a)
# plt.scatter(f(a), np.zeros(201)-b0, s=3)
# plt.show()
#
# print(f(a))

# path = 'saves/save_15_24_28_32/'
# Flist = os.listdir(path)
# Flist_split = [f.split('_') for f in Flist]
#
# for i in range(len(Flist)):
#     if Flist_split[i][-1] == 'logvar.csv':
#         print(Flist[i])
#         os.remove(path+Flist[i])

# def subtract_bkgd(topo, n):
#     for i in range(n):
#         paras = np.polyfit(np.arange(n), topo[i], 1)
#         topo[i] = topo[i]-np.poly1d(paras)(np.arange(n))
#     return topo
#
#
# def FFT(a):
#     return 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(a))))
#
#
def give_mapping_lims(a, mapping_type):
    mean = np.mean(a)
    std = np.std(a)
    if mapping_type=='r':
        return mean-2*std, mean+2*std
    elif mapping_type=='k':
        return mean-0.5*std, mean+3*std
#
# # reB 是一个how many lines need be removed from B&L before correction?
# # lengthNamely = lengthNamely / size(topo,1) * (size(topo,1) - reB);
# # topo = topo(1:end-reB, 1+reB:end);
# # gridData = gridData(1:end-reB,1+reB:end,:);
# # lengthperpixel = lengthNamely / size(topo,1);
#
#
# def index2wavevector(n, x1, y1, x2, y2, length):
#     # 用到的length = length_per_pixel*size(topo,1)
#     # x1, y1对应第一个Q(Qx), y类似
#     center = math.floor(n/2)+1
#     x1 = x1 - center
#     y1 = y1 - center
#     x2 = x2 - center
#     y2 = y2 - center
#     FFT_unit = 2*math.pi/length
#     wv = {'qxx': FFT_unit*x1, 'qxy': FFT_unit*y1, 'qyx': FFT_unit*x2, 'qyy': FFT_unit*y2}
#     return wv
#
#
# def adjustwv(wv):
#     # 应该是定死了波矢的方向和大小？
#     if (wv['qxx']*wv['qxy'] > 0) & (wv['qyx']*wv['qyy'] < 0):
#         temp0 = wv['qxx']
#         temp1 = wv['qxy']
#         wv['qxx'] = wv['qyx']
#         wv['qxy'] = wv['qyy']
#         wv['qyx'] = temp0
#         wv['qyy'] = temp1  # Qx, Qy互换了一下
#     wv['qxx'] = abs(wv['qxx'])
#     wv['qxy'] = -abs(wv['qxy'])
#     wv['qyx'] = abs(wv['qyx'])
#     wv['qyy'] = abs(wv['qyy'])
#     return wv
#
#
# def kSpaceMap(n, length_per_pixel):
#     kx = np.arange(n) * 2*math.pi/(length_per_pixel*n)
#     # length_per_pixel*n是整张图一条边对应的物理长度，kx是对应的波长大小。
#     # kx = np.tile(kx, (len(kx), 1))
#     # # 还是对meshgrid理解不到位哦！！！这里要转置处理！
#     # # 傻逼……人家np直接有meshgrid，还是你太菜
#     # ky = kx.T
#     kx, ky = np.meshgrid(kx, kx)
#     kx = np.fft.fftshift(kx)
#     ky = np.fft.fftshift(ky)
#     flr = math.floor(n/2)
#     kx[:, 0:flr] = kx[:, 0:flr]-2*math.pi/length_per_pixel
#     # matlab和python数组索引的微妙区别！一定要三思！
#     # 挪到负半轴
#     ky[0:flr, :] = ky[0:flr, :]-2*math.pi/length_per_pixel
#
#     return kx, ky
#
#
# # 算法表现严重依赖于解卷绕的效果
# def LawlerFujita(topo, length_per_pixel, wv, L):
#     # wavevector来自于前面确认过的Qxx，Qxy等等等等，但是需要确认一下
#     L = L * a0 / length_per_pixel
#     kx, ky = kSpaceMap(n, length_per_pixel)
#     Lambda = 1/L
#     # 输入的L是总的原子，还是设置漂移几乎不变区域的原子，In Lawlyer-Fujita, L = ? atoms
#     # 前面还有L = L * a0 / lengthperpixel
#     F = np.exp(-(np.power(kx, 2)+np.power(ky, 2))/2/Lambda**2)
#     # 这应该是一个（高斯？）平均场
#     x = np.arange(n)*length_per_pixel
#     # x = np.tile(x, (len(x), 1))
#     # y = x.T
#     # # 和上面一样，创造X, Y的格点，步长是length_per_pixel
#     x, y = np.meshgrid(x, x)
#
#
#     planeWaveQx = np.exp(-1j*(wv['qxx']*x+wv['qxy']*y))
#     # 应该整出来还是一个二维数组吧，直接就是x，y上Qx的振幅？
#     # qxx是单个的数字.
#     planeWaveQy = np.exp(-1j*(wv['qyx']*x+wv['qyy']*y))
#     # 应该整出来还是一个二维数组吧
#
#     def process(planewave):
#         topo_with_wave = np.multiply(topo, planewave)
#         ftopo = fft.fftshift(fft.fft2(topo_with_wave))
#         T = np.multiply(ftopo, F)
#         T = np.angle(fft.ifft2(fft.ifftshift(T)))
#         T = np.unwrap(np.unwrap(T, axis=1), axis=0)
#         return T
#
#     # topoQx = np.multiply(topo, planeWaveQx)
#     # topoQy = np.multiply(topo, planeWaveQy)
#     #
#     # ftopoQx = np.fft.fftshift(np.fft.fft2(topoQx))
#     # ftopoQy = np.fft.fftshift(np.fft.fft2(topoQy))
#     #
#     # T1 = np.multiply(ftopoQx, F)
#     # T2 = np.multiply(ftopoQy, F)
#     #
#     # T1 = np.angle(np.fft.ifft2(np.fft.ifftshift(T1)))
#     # T2 = np.angle(np.fft.ifft2(np.fft.ifftshift(T2)))
#     # # 得到T1, T2幅角
#     # T1 = np.unwrap(np.unwrap(T1, axis=0), axis=1)
#     # T2 = np.unwrap(np.unwrap(T2, axis=0), axis=1)
#     # T1, T2应当是各点上对应的相位偏移
#     # 注意要对两个维度解卷绕
#
#     return process(planeWaveQx), process(planeWaveQy)
#
#
# plt.ion()

# grid_path = 'grids/08_11_13_16_18_19_21_22_23/16.3ds'
# grid_dict = nanonispy.read.Grid(grid_path)._load_data()
# print("Grid loaded. The keys are", list(grid_dict.keys()))
#
# length_namely = 320  # 单位为A
# grid_size = grid_dict['params'].shape[0]
# a0 = 3.8  #晶格常数，单位为A
# n = grid_size
# length_per_pixel = length_namely / n
# L = 5  #超参数！！！！
# rotate_to_45 = True
#
# # the z information is stored in the 'paras' ndarray (shape 256*256*10), z is in the 4th column.
# # topo = grid_dict['params'][:, :, 4].reshape((grid_size, -1))
#
# dIdV = grid_dict['LI Demod 1 X (A)'][:, :, 44].reshape((grid_size, -1))
# # vmin, vmax = give_mapping_lims(dIdV, 'r')
# # plt.imshow(dIdV, vmin=vmin, vmax=vmax)
# # plt.gca().invert_yaxis()
# # plt.show()
#
# AC = correlate2d(dIdV, dIdV, 'same', 'wrap')
# plt.imshow(AC)
# plt.gca().invert_yaxis()
# plt.show()

# string_p = input('input p0 & p1: ')
# # np.zeros(dim)
# string_p = list(filter(None, re.split(',|\s|\[|\]', string_p)))
# print(string_p)
# p0 = [float(string_p[i]) for i in range(10)]
# p1 = string_p[10:]
# print([np.array(p0), np.array(p1)])



k = 3
a = [3, 7, 1, 9, 5]

indices =
print(indices)  # 输出：[0, 2, 4]