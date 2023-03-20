import os
import subprocess

# Use list instead of tuple
model_index = [0]
save_index = [[6, 7, 8]]
beta = [[1e-3, 1e-3, 1e-3]]
patience = 5
dpk = 'interp_GF05_2'

flag_train = 1
flag_plot = 1


def exe_cmd(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        result = p.stdout.readline()  # 默认获取到的是二进制内容
        if result != b'':  # 获取内容不为空时
            try:
                print(result.decode('gbk').strip('\r\n'))  # 处理GBK编码的输出，去掉结尾换行
            except:
                print(result.decode('utf-8').strip('\r\n'))  # 如果GBK解码失败再尝试UTF-8解码
        else:
            break


d = os.popen("activate cuprate")
print(d.read())

d = os.popen("cd C:\\Users\\SethC\\STM_VAE")
print(d.read())

for i in range(len(model_index)):
    for j in range(len(save_index[i])):
        if flag_train:
            exe_cmd(f"python train.py -m {model_index[i]} -s {save_index[i][j]} -p {patience} -b {beta[i][j]} -k {dpk}")
        if flag_plot:
            exe_cmd(f"python analyse.py -m {model_index[i]} -s {save_index[i][j]} -k {dpk}")
