import numpy as np
from numpy import fft
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import patches
import math
import nanonispy


def subtract_bkgd_lbl(topo, n):
    # Linear by line
    for i in range(n):
        paras = np.polyfit(np.arange(n), topo[i], 1)
        topo[i] = topo[i]-np.poly1d(paras)(np.arange(n))
    return topo


def FFT(a):
    return 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(a))))


def give_mapping_lims(a, mapping_type):
    mean = np.mean(a)
    std = np.std(a)
    if mapping_type == 'r':
        return mean-2*std, mean+2*std
    elif mapping_type == 'k':
        return mean-0.5*std, mean+3*std

# reB 是一个how many lines need be removed from B&L before correction?
# lengthNamely = lengthNamely / size(topo,1) * (size(topo,1) - reB);
# topo = topo(1:end-reB, 1+reB:end);
# gridData = gridData(1:end-reB,1+reB:end,:);
# lengthperpixel = lengthNamely / size(topo,1);


def index2wavevector(n, x1, y1, x2, y2, length):
    # 用到的length = length_per_pixel*size(topo,1)
    # x1, y1对应第一个Q(Qx), y类似
    center = math.floor(n/2)+1
    x1 = x1 - center
    y1 = y1 - center
    x2 = x2 - center
    y2 = y2 - center
    FFT_unit = 2*math.pi/length
    wv = {'qxx': FFT_unit*x1, 'qxy': FFT_unit*y1, 'qyx': FFT_unit*x2, 'qyy': FFT_unit*y2}
    return wv


def adjustwv(wv):
    # 应该是定死了波矢的方向和大小？
    if (wv['qxx']*wv['qxy'] > 0) & (wv['qyx']*wv['qyy'] < 0):
        temp0 = wv['qxx']
        temp1 = wv['qxy']
        wv['qxx'] = wv['qyx']
        wv['qxy'] = wv['qyy']
        wv['qyx'] = temp0
        wv['qyy'] = temp1  # Qx, Qy互换了一下
    wv['qxx'] = abs(wv['qxx'])
    wv['qxy'] = -abs(wv['qxy'])
    wv['qyx'] = abs(wv['qyx'])
    wv['qyy'] = abs(wv['qyy'])
    return wv


def kSpaceMap(n, length_per_pixel):
    kx = np.arange(n) * 2*math.pi/(length_per_pixel*n)
    # length_per_pixel*n是整张图一条边对应的物理长度，kx是对应的波长大小。
    # kx = np.tile(kx, (len(kx), 1))
    # # 还是对meshgrid理解不到位哦！！！这里要转置处理！
    # # 傻逼……人家np直接有meshgrid，还是你太菜
    # ky = kx.T
    kx, ky = np.meshgrid(kx, kx)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    flr = math.floor(n/2)
    kx[:, 0:flr] = kx[:, 0:flr]-2*math.pi/length_per_pixel
    # matlab和python数组索引的微妙区别！一定要三思！
    # 挪到负半轴
    ky[0:flr, :] = ky[0:flr, :]-2*math.pi/length_per_pixel

    return kx, ky


# 算法表现严重依赖于解卷绕的效果
def LawlerFujita(topo, length_per_pixel, wv, L, a0=3.8, unwrap_para=(0, 0)):
    n = topo.shape[0]
    # wavevector来自于前面确认过的Qxx，Qxy等等等等，但是需要确认一下
    L = L * a0 / length_per_pixel
    kx, ky = kSpaceMap(n, length_per_pixel)
    Lambda = 1/L
    # 输入的L是总的原子，还是设置漂移几乎不变区域的原子，In Lawlyer-Fujita, L = ? atoms
    # 前面还有L = L * a0 / lengthperpixel
    F = np.exp(-(np.power(kx, 2)+np.power(ky, 2))/2/Lambda**2)
    # 这应该是一个（高斯？）平均场
    x = np.arange(n)*length_per_pixel
    # x = np.tile(x, (len(x), 1))
    # y = x.T
    # # 和上面一样，创造X, Y的格点，步长是length_per_pixel
    x, y = np.meshgrid(x, x)


    planeWaveQx = np.exp(-1j*(wv['qxx']*x+wv['qxy']*y))
    # 应该整出来还是一个二维数组吧，直接就是x，y上Qx的振幅？
    # qxx是单个的数字.
    planeWaveQy = np.exp(-1j*(wv['qyx']*x+wv['qyy']*y))
    # 应该整出来还是一个二维数组吧

    def process(planewave, order=0):
        topo_with_wave = np.multiply(topo, planewave)
        ftopo = fft.fftshift(fft.fft2(topo_with_wave))
        T = np.multiply(ftopo, F)
        T = np.angle(fft.ifft2(fft.ifftshift(T)))
        if not order:
            T = np.unwrap(np.unwrap(T, axis=0), axis=1)
        else:
            T = np.unwrap(np.unwrap(T, axis=1), axis=0)
        # 顺序有莫名其妙的影响
        return T

    # topoQx = np.multiply(topo, planeWaveQx)
    # topoQy = np.multiply(topo, planeWaveQy)
    #
    # ftopoQx = np.fft.fftshift(np.fft.fft2(topoQx))
    # ftopoQy = np.fft.fftshift(np.fft.fft2(topoQy))
    #
    # T1 = np.multiply(ftopoQx, F)
    # T2 = np.multiply(ftopoQy, F)
    #
    # T1 = np.angle(np.fft.ifft2(np.fft.ifftshift(T1)))
    # T2 = np.angle(np.fft.ifft2(np.fft.ifftshift(T2)))
    # # 得到T1, T2幅角
    # T1 = np.unwrap(np.unwrap(T1, axis=0), axis=1)
    # T2 = np.unwrap(np.unwrap(T2, axis=0), axis=1)
    # T1, T2应当是各点上对应的相位偏移
    # 注意要对两个维度解卷绕

    return process(planeWaveQx, unwrap_para[0]), process(planeWaveQy, unwrap_para[1])


def LawlerFujita_lattice_correction(grid_path, length_namely, L, Q_idx, cutoff,
                                    unwrap_para=(0, 0), a0=3.8, rotate_to_45=True, twist_dIdV=False, skip_show=False):
    # L 几个原子附近u不变？
    # fitQarea格式：起点(x,y), 边长(a, b), 写成[[a, b], c, d]
    # a0 = 3.8  # 晶格常数，单位为A
    # rotate_to_45 = True
    if skip_show: plt.ion()

    [l_cutoff, r_cutoff] = cutoff
    grid_dict = nanonispy.read.Grid(grid_path)._load_data()
    print("Grid loaded. The keys are", list(grid_dict.keys()))

    # paras needs manual entering

    n = grid_dict['params'].shape[0]
    length_per_pixel = length_namely / n
    print('length_per_pixel =', length_per_pixel)

    # the z information is stored in the 'paras' ndarray (shape 256*256*10), z is in the 4th column.
    topo = grid_dict['params'][:, :, 4].reshape((n, -1))

    # subtract background
    topo = subtract_bkgd_lbl(topo, n)

    fig, ax = plt.subplots()
    plt.imshow(topo)
    ax.invert_yaxis()

    fig, ax = plt.subplots()
    vmin, vmax = give_mapping_lims(FFT(topo), 'k')
    ax.imshow(FFT(topo), vmin=vmin, vmax=vmax)
    ax.invert_yaxis()

    '''
    这段是手写的四个corner，感觉效果不好，不如直接在前面要求给死两个corner四个点
        fit_range = [70, 15]    # 框框起点，边长
        middle = (grid_size-1)/2
        corners = [[middle+i*fit_range[0]+(i-1)/2*fit_range[1], middle+j*fit_range[0]+(j-1)/2*fit_range[1]
                    ] for j in (1, -1) for i in (1, -1)]  # 双for嵌套定义可行，corners数组指向4个角角
        print(corners)
    
        rect_style = {'edgecolor':'black', 'fill':False}
        Q_idx = []
        for corner in corners:
            ax.add_patch(patches.Rectangle(corner, fit_range[1], fit_range[1], **rect_style))
            b= FFT(topo)[int(corner[0] - 1 / 2):int(corner[0] - 1 / 2 + fit_range[1]), int(corner[0] - 1 / 2):int(corner[0] - 1 / 2 + fit_range[1])]
            print(np.unravel_index(np.argmax(b), b.shape))
            (d, c) = np.unravel_index(np.argmax(b), b.shape)  # 注意顺序
            Q_idx.append([int(corner[0]+c-1/2), int(corner[1]+d-1/2)])
            plt.scatter(corner[0]+c, corner[1]+d, c='w', s=10)
            print((int(corner[0]+c-1/2), int(corner[1]+d-1/2)))
    
        ax.invert_yaxis()
        plt.show()
    '''
    # 还是不好用
    # rect_style = {'edgecolor': 'black', 'fill': False}
    # Q_idx = []
    # for i in range(2):
    #     ax.add_patch(patches.Rectangle(fitQarea[i][0], fitQarea[i][1], fitQarea[i][2], **rect_style))
    #     b = FFT(topo)[fitQarea[i][0][0]:fitQarea[i][0][0]+fitQarea[i][1],
    #                   fitQarea[i][0][1]:fitQarea[i][0][1]+fitQarea[i][2]]
    #     print(np.unravel_index(np.argmax(b), b.shape))
    #     (d, c) = np.unravel_index(np.argmax(b), b.shape)  # 注意顺序
    #     Q_idx.append([fitQarea[i][0][0]+c, fitQarea[i][0][1]+d])
    #     plt.scatter(fitQarea[i][0][0]+c, fitQarea[i][0][1]+d, c='w', s=7)
    #     print(fitQarea[i][0][0]+c, fitQarea[i][0][1]+d)

    wv = index2wavevector(n, Q_idx[1][0], Q_idx[1][1], Q_idx[0][0], Q_idx[0][1], length_namely)
    print(wv)
    wv = adjustwv(wv)
    print(wv)

    T1, T2 = LawlerFujita(topo, length_per_pixel, wv, L, a0, unwrap_para)

    fig, ax = plt.subplots()
    plt.imshow(T1)

    fig, ax = plt.subplots()
    plt.imshow(T2)

    # Find Drift field % Convert T1,2 to drift field
    Tx = np.zeros((n, n)); Ty = np.zeros((n, n))
    wv_matrix = [[wv['qxx'], wv['qxy']],
                 [wv['qyx'], wv['qyy']]]
    for i in range(n):
        for j in range(n):
            r = np.dot(np.linalg.inv(wv_matrix), [[T1[i, j]], [T2[i, j]]])  # in Matlab A\B=inv(A)*B
            [Tx[i, j], Ty[i, j]] = r

    print('Finished Convert T1_2 to drift field')
    # plt.imshow(Tx)
    # plt.show()
    # plt.imshow(Ty)
    # plt.show()

    # Correct Drift
    Q_1 = wv
    Q_0 = {}

    if rotate_to_45:
        theta = 0
    else:
        theta = np.arctan((wv['qxy']+wv['qyy'])/(wv['qyx']+wv['qxx']))

    qx_theta = theta - math.pi/4
    qy_theta = theta + math.pi/4
    a0 = 2*math.pi/np.sqrt(max([wv['qxx']**2+wv['qxy']**2, wv['qyx']**2+wv['qyy']**2]))
    Q_0['qxx'] = 2*math.pi / a0 * np.cos(qx_theta)
    Q_0['qxy'] = 2*math.pi / a0 * np.sin(qx_theta)
    Q_0['qyx'] = 2*math.pi / a0 * np.cos(qy_theta)
    Q_0['qyy'] = 2*math.pi / a0 * np.sin(qy_theta)
    wv_correct = Q_0
    Q_1 = [[Q_1['qxx'],Q_1['qyx']], [Q_1['qxy'], Q_1['qyy']]]
    Q_0 = [[Q_0['qxx'],Q_0['qyx']], [Q_0['qxy'], Q_0['qyy']]]
    TransferMatrix = np.dot(Q_1, np.linalg.inv(Q_0)).T

    x = np.arange(n) * length_per_pixel
    x, y = np.meshgrid(x, x)
    xcor = x+Tx
    ycor = y+Ty
    for i in range(n):
        for j in range(n):
            temp = [[xcor[i, j]], [ycor[i, j]]]
            temp = np.dot(TransferMatrix, temp)
            [xcor[i, j], ycor[i, j]] = temp
            # TransferMatrix和xcor是什么意思
            # 感觉是带单位的真实坐标
    print('find real location of pixels\n')

    fig, ax = plt.subplots()
    plt.pcolormesh(xcor, ycor, topo)
    # y轴会有点问题，这里y轴不需要再翻过来一次了
    ax = plt.gca()
    ax.set_aspect(1)

    # ax.set_xlim(l_cutoff*length_per_pixel, length_namely-r_cutoff*length_per_pixel)
    # ax.set_ylim(l_cutoff*length_per_pixel, length_namely-r_cutoff*length_per_pixel)


    # 我们先给死length perpixel不变好了
    fig, ax = plt.subplots()
    x_new = length_per_pixel*np.arange(l_cutoff, n-r_cutoff)  # 带单位de!
    x_new, y_new = np.meshgrid(x_new, x_new)
    points = np.stack((xcor.flatten(), ycor.flatten())).T  # points要求的奇怪格式
    topo_new = interpolate.griddata(points, topo.flatten(), (x_new, y_new), method='cubic')
    plt.imshow(topo_new)
    plt.gca().invert_yaxis()

    fig, ax = plt.subplots()
    vmin, vmax = give_mapping_lims(FFT(topo_new), 'k')
    plt.imshow(FFT(topo_new), vmin=vmin, vmax=vmax)
    plt.show()

    if skip_show: plt.ioff()

    if twist_dIdV:
        # save twist_paras
        twist_paras = {'l_cutoff': l_cutoff, 'r_cutoff': r_cutoff, 'wv': wv, 'L': L,
                       'rotate_to_45': rotate_to_45, 'unwrap_para': unwrap_para}
        '''
        好像不必改
        name = grid_path[-6:-4]
        if name == 'OP32K' or name == '16':  # 就因为OP32K的最后多了几个点
            grid_dict['LI Demod 1 X (A)'] = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[:-5]
            grid_dict['LI Demod 1 X (A)'] = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)

        if name == '19':  # 就因为OP32K的最后多了几个点
            grid_dict['LI Demod 1 X (A)'] = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)[1:]
            grid_dict['LI Demod 1 X (A)'] = grid_dict['LI Demod 1 X (A)'].swapaxes(0, 2)
        '''

        n_new = n-r_cutoff-l_cutoff
        length_new = [n_new, n_new, grid_dict['LI Demod 1 X (A)'].shape[2]]
        twisted_dIdV = np.zeros(length_new)
        for i in range(length_new[2]):
            temp = grid_dict['LI Demod 1 X (A)'][:, :, i]
            # temp = subtract_bkgd_lbl(temp, n) 自作主张！！！这是不对的！
            twisted_dIdV[:, :, i] = interpolate.griddata(points, temp.flatten(), (x_new, y_new), method='cubic')
            print(f'{i+1}/{length_new[2]} twisted')
        twisted_dIdV = twisted_dIdV.astype(np.float32)
        np.savez(grid_path[:-4]+'.twisted.npz', twisted_dIdV=twisted_dIdV, twist_paras=twist_paras)


#
# grid_path = 'grids/08_11_13_16_18_19_21_22_23/16.3ds'
# length_namely = 320
# L = 5
# Q_idx = [[205, 206], [45, 206]]
# cutoff = [20, 22]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=0, skip_show=False, unwrap_para=(0, 1))

# grid_path = 'grids/08_11_13_16_18_19_21_22_23/08.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[213, 217], [44, 217]]
# cutoff = [5, 10]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=1, skip_show=0, unwrap_para=(0, 1))

# grid_path = 'grids/08_11_13_16_18_19_21_22_23/11.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[208, 211], [49, 210]]
# cutoff = [10, 10]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=1, skip_show=0, unwrap_para=(1, 1))

# twist = 1; skip = 1
#
# grid_path = 'grids/08_11_13_16_18_19_21_22_23/13.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[203, 200], [61, 201]]
# cutoff = [13, 10]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=twist, skip_show=skip, unwrap_para=(0, 1))
#
# grid_path = 'grids/08_11_13_16_18_19_21_22_23/18.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[226, 229], [47, 230]]
# cutoff = [20, 20]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=0, skip_show=0, unwrap_para=(0, 1))
#
# grid_path = 'grids/08_11_13_16_18_19_21_22_23/19.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[215, 220], [41, 213]]
# cutoff = [33, 12]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=0, skip_show=0, unwrap_para=(1, 1))
#
# grid_path = 'grids/08_11_13_16_18_19_21_22_23/22.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[186, 186], [71, 184]]
# cutoff = [5, 5]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=0, skip_show=0, unwrap_para=(0, 1))
#
# grid_path = 'grids/08_11_13_16_18_19_21_22_23/23.3ds'
# length_namely = 320  # 其实不用改
# L = 5
# Q_idx = [[193, 194], [63, 192]]
# cutoff = [5, 5]
# LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=twist, skip_show=skip, unwrap_para=(0, 1))

grid_path = 'grids/04_08_11_13/04.3ds'
length_namely = 320  # 其实不用改
L = 5
Q_idx = [[138, 141], [70, 140]]
cutoff = [10, 15]
LawlerFujita_lattice_correction(grid_path, 320, 5, Q_idx, cutoff, twist_dIdV=1, skip_show=1, unwrap_para=(0, 0))


# twisted_dIdV = np.load(grid_path[:-4]+'.twisted.npz')['twisted_dIdV']
# plt.imshow(twisted_dIdV[:, :, 50])
# plt.show()
