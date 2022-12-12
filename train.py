import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import importlib
import argparse
import os
import shutil  # to rename the file


model_index_default = 7
save_index_default = 0
beta_default = 1e-3
grid_set = 'UD150'
grid_set_path = 'grids/04_08_11_13/'
dpk_default = 'interp_twisted'

# grid_set = '00_10_22'
# grid_set_path = 'grids/00_10_22/'

# argument parsers
parser = argparse.ArgumentParser(description='Welcome!')
parser.add_argument('-m', '--mid', default=model_index_default, type=int, help='give models index')
parser.add_argument('-s', '--sid', default=save_index_default, type=int, help='give save index')
parser.add_argument('-p', '--pat', default=5, type=int, help='give patience')
parser.add_argument('-b', '--bet', default=beta_default, type=float, help='give beta')
parser.add_argument('-k', '--key', default=dpk_default, type=str, help='give data process keyword')
args = parser.parse_args()

model_index = args.mid
save_index = args.sid
data_process_keyword = args.key

# 预测grid
flag_generalize = 0

# hyperparameter
patience = args.pat
beta = args.bet
batch_size = 256
lr = 5e-05
wd = 1e-06
noise_factor = 0.2

log_file = 'ModelTrainingLog.txt'

grid_names = []
Flist = os.listdir(grid_set_path)
Flist_split = [f.split('.') for f in Flist]
for i in range(len(Flist)):
    if Flist_split[i][1] == data_process_keyword:
        grid_names.append(Flist[i])

if flag_generalize:
    print('Generalization!')
    Tcby10_label = np.array([1.5, 2.4, 3.2]).astype(np.float32)
    grid_set = '3+1'
    grid_names = ['OD15K.interp_sub.npz', 'OD24K.interp_sub.npz', 'OP32K.interp_sub.npz']

# import model
mdl = importlib.import_module('models.VAE_model_'+str(model_index))
print(f'VAE_model_{model_index} Save {save_index}, patience = {patience}')

# for concision
save_name = 'VAE_model_'+str(model_index)+'_save_'+str(save_index)
save_path = 'saves/save_'+grid_set+'/'+save_name
fig_path = 'figs/fig_'+grid_set+'/'+save_name+'/'+save_name

# check path existence
if not os.path.exists('saves/save_'+grid_set):
    os.makedirs('saves/save_'+grid_set)
if not os.path.exists('figs/fig_'+grid_set+'/'+save_name):
    os.makedirs('figs/fig_'+grid_set+'/'+save_name)


# log the losses, a dictionary of list of different losses (MSE, KLD, total)
loss_keys = ["MSE_loss", "b*KLD_loss", "loss"]
train_losses = {key: [] for key in loss_keys}
test_losses = {key: [] for key in loss_keys}


# Load and encapsulate data as torch.Dataset
class FullDataset(Dataset):

    # all grids jointed together
    def __init__(self, grid_names):
        super().__init__()
        self.grid_num = len(grid_names)
        self.dIdV = []
        self.lens = []
        self.mean_curve = []
        self.offset = []
        for grid in grid_names:
            data = np.load(grid_set_path + grid)
            self.bias = data['bias']
            self.dIdV.append(data['dIdV'])
            self.lens.append(len(data['dIdV']))
            self.mean_curve.append(data['mean_curve'])
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
        if flag_generalize:
            return dIdV, Tcby10_label[label]
        else:
            return dIdV, label

    def __len__(self):
        return sum(self.lens)


# instantiate (load grids)
full_dataset = FullDataset(grid_names)
print('Input dimension:', len(full_dataset)*len(full_dataset.bias))

# split train and test set
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

VAE = mdl.VAE(input_len=len(full_dataset.bias), encoded_space_dim=mdl.encoded_dim)
para_num = sum(param.numel() for param in VAE.parameters())
print('VAE parameters: ', para_num)
optimizer = torch.optim.Adam(VAE.parameters(), lr=lr, weight_decay=wd)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')
# Move VAE to the selected device
VAE.to(device)
# Need, as well, to move both the grids and the labels to device


def add_noise(inputs, noise_factor):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    return noisy


def loss_function(X, X_hat, mu, logvar):
    MSE_loss = torch.nn.functional.mse_loss(X, X_hat)

    """
    KL divergence
    ref: https://arxiv.org/pdf/1606.05908.pdf Equ.(7)
    D_KL [N(mu(X), Sigma(X)) || N(0, I)] = (1/2) * [ tr(Sigma) + mu^T mu - k - log[det(Sigma)] ]
    k is the distribution dimension
    where Sigma is the covariance matrix
    NOTE we are using log_var
    """
    # sum over encoded dimensions and average over
    KLD_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + torch.square(mu) - 1 - logvar, dim=1), dim=0)

    loss = MSE_loss + beta * KLD_loss

    losses = {"MSE_loss": MSE_loss, "b*KLD_loss": beta*KLD_loss, "loss": loss}
    return losses


def train_epoch(VAE, dataloader, optimizer):
    # set train mode for VAE
    VAE.train()
    # also dict list
    batch_train_loss = {key: [] for key in loss_keys}

    for dIdV, label in dataloader:

        # add some noise
        dIdV_noisy = add_noise(dIdV, noise_factor).to(device)
        dIdV = dIdV.to(device)
        label = label.to(device)

        decoded_output, mu, logvar = VAE(dIdV_noisy, label)

        losses = loss_function(dIdV, decoded_output, mu, logvar)

        loss = losses['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in loss_keys:
            batch_train_loss[key].append(losses[key].cpu().detach().numpy())

    for key in loss_keys:
        train_losses[key].append(np.mean(batch_train_loss[key]))


def test_epoch(VAE, dataloader):
    # set evaluation mode for VAE
    VAE.eval()
    with torch.no_grad():  # no need to track the gradients

        batch_test_loss = {key: [] for key in loss_keys}
        for dIdV, label in dataloader:

            dIdV_noisy = add_noise(dIdV, noise_factor).to(device)
            dIdV = dIdV.to(device)
            label = label.to(device)

            decoded_output, mu, logvar = VAE(dIdV_noisy, label)

            losses = loss_function(dIdV, decoded_output, mu, logvar)

            for key in loss_keys:
                batch_test_loss[key].append(losses[key].cpu().detach().numpy())

        for key in loss_keys:
            test_losses[key].append(np.mean(batch_test_loss[key]))


def SaveEncodedSamples():
    enc_mus = []  # list of dictionary
    # enc_logvars = []
    for sample in full_dataset:
        dIdV = torch.reshape(torch.Tensor(sample[0]), (1, -1)).to(device)
        label = sample[1]
        # encode data
        VAE.eval()
        with torch.no_grad():
            mu, logvar = VAE._encode(dIdV)
        # Append to list
        mu = mu.flatten().cpu().numpy()
        enc_mu = {f'{i}': paras for i, paras in enumerate(mu)}  # 用字典储存, 'i'->paras[i]
        enc_mu['label'] = label
        enc_mus.append(enc_mu)

        # logvar = logvar.flatten().cpu().numpy()
        # enc_logvar = {f'{i}': paras for i, paras in enumerate(logvar)}  # 用字典储存, 'i'->paras[i]
        # enc_logvar['label'] = label
        # enc_logvars.append(enc_logvar)

    enc_mus = pd.DataFrame(enc_mus)
    enc_mus.to_csv(path_or_buf=save_path+'_encoded_mu.csv')

    # enc_logvars = pd.DataFrame(enc_logvars)
    # enc_logvars.to_csv(path_or_buf=save_path+'_encoded_logvar.csv')
    # print('encoded means & log covariances saved as csv')
    print('encoded means saved as csv')


epoch = 0
min_test_loss = 0
date = time.asctime(time.localtime(time.time()))
start_time = time.time()


while True:

    train_epoch(VAE, train_loader, optimizer)
    test_epoch(VAE, test_loader)
    print('Epoch {}\ttrain loss {:.6f}\ttest loss {:.6f}\tMSE loss {:.6f}\tbeta*KLD loss {:.6f}'
          .format(epoch, train_losses['loss'][epoch],
                  test_losses['loss'][epoch],
                  test_losses['MSE_loss'][epoch],
                  test_losses['b*KLD_loss'][epoch]))

    # Early Stopping
    if epoch == 0:
        min_test_loss = test_losses['loss'][epoch]

    if test_losses['loss'][epoch] <= min_test_loss:
        min_test_loss = test_losses['loss'][epoch]
        early_stopping = 0

        torch.save({'epoch': epoch, 'VAE_state_dict': VAE.state_dict(),
                    'optimizer': optimizer.state_dict(), 'train_losses': train_losses, 'test_losses': test_losses,
                    'min_test_loss': min_test_loss},
                   save_path + '_temp.pth')

    else:
        early_stopping += 1
        print("Early stopping count {} of {}".format(early_stopping, patience))
        if early_stopping >= patience:
            print("Early stopped with best test loss: {:.6f}, and test loss for this epoch: {:.6f}."
                  .format(min_test_loss, test_losses['loss'][epoch]))
            break

    epoch += 1


end_time = time.time()

# losses plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(epoch+1), test_losses['loss'], label='loss')
ax1.plot(range(epoch+1), test_losses['MSE_loss'], label='MSE_loss')
plt.xlabel('Epoch')
ax1.set_ylabel('Test Loss')
plt.legend()
# ax1.set_yscale('log')

ax2 = ax1.twinx()
ax2.plot(range(epoch+1), test_losses['b*KLD_loss'], label='b*KLD_loss', c='r')
plt.legend()
plt.savefig(fig_path+'_test_loss.svg')

# rename temp_save as save
shutil.move(save_path+'_temp.pth', save_path+'.pth')

print('Losses and model saved.')

# log train information
f = open(log_file, 'a')
if flag_generalize:
    f.write('GENERALIZATION: yes')
f.write(f'''
--------------------------------
    Grid set: {grid_set}
    Grid name: {grid_names},
    {save_name}, {date},
    Train time: {end_time-start_time:.2f}s,
    Train device: {device}
    encoded dimension = {mdl.encoded_dim},
    VAE parameter numbers = {para_num},
    Input dimension = {len(full_dataset)*len(full_dataset.bias)},
    patience = {patience},
    beta for beta-VAE = {beta}
    batch size = {batch_size},
    learning rate = {lr},
    noise factor = {noise_factor}
    adam weight decay = {wd}.
    {epoch} epochs trained.
    Early stopped with best test loss {min_test_loss:.6f} at epoch {epoch - patience}.
    Losses logged and models saved.
--------------------------------
''')
f.close()

print('Train info logged.')

# save latent space data
SaveEncodedSamples()

print('Encoded data saved.')
