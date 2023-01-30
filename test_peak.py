
import torch
import random
import time
import numpy as np
import math
import sys
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from model_peak import Conformer
from scipy import signal
from audtorch.metrics.functional import pearsonr
import seaborn as sns

############################# random seed ###################################
seed = 77
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

############################# check GPU ###################################

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

######################### batch_size, epoch, LR ############################

input_sequence_length = 1000
output_sequence_length = 1000

################################ load data ##################################

data = np.load('train_all_seg_cc.npz', allow_pickle=True)

X_train = data["X_train"].astype(float)
Y_train = data["Y_train"].astype(float)


X_test = data["X_test"].astype(float)
Y_test = data["Y_test"].astype(float)


X_train = X_train.reshape(-1, 1, input_sequence_length, 5)
Y_train = Y_train.reshape(-1, 1, output_sequence_length, 6)


X_test = X_test.reshape(-1, 1, input_sequence_length, 5)
Y_test = Y_test.reshape(-1, 1, output_sequence_length, 6)

train_ecg = Y_train[:, :, :, 0]
test_ecg   =  Y_test[:, :, :, 0]

Y_train = Y_train[:, :, :, 1:]
Y_test  =  Y_test[:, :, :, 1:]

X_train = torch.from_numpy(X_train).type(torch.float32)
Y_train = torch.from_numpy(Y_train).type(torch.float32)

X_test = torch.from_numpy(X_test).type(torch.float32)
Y_test = torch.from_numpy(Y_test).type(torch.float32)

print('###################################')
print("X_train: ", X_train.shape)
print("Y_train: ", Y_train.shape)

print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)
print('###################################')


train = torch.utils.data.TensorDataset(X_train, Y_train)
val = torch.utils.data.TensorDataset(X_test, Y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True,
                                           generator=g)
val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True,
                                         generator=g)

############################# model ###################################

model = Conformer(num_classes=1000, input_dim=10,
                      encoder_dim=16,
                      num_encoder_layers=2,
                      num_attention_heads=16,
                      input_dropout_p=0.5,
                      feed_forward_dropout_p=0.5,
                      attention_dropout_p=0.5,
                      conv_dropout_p=0.5,
                      conv_kernel_size=31).to(device)
                      
model.load_state_dict(torch.load("model_peak.pt"))
model.eval()
print(model.training)

############################# TEST  ############################

MAE = nn.L1Loss()
MSE = nn.MSELoss()

def pearsonr_correlation(input, target):
    output = pearsonr(input, target)
    #output = np.corrcoef(input, target)
    
    return abs(output)

def to_np(x):
    x = x.detach()
    x = np.array(x.cpu())
    return x


print("############################# Training  ############################")
fig, ax = plt.subplots(3, 4, figsize=(18,9))
plt.suptitle('training')

for i in range(4):
    n = random.randrange(0, len(X_train))

    pred = model(X_train[n].unsqueeze(0).to(device))
    print(n,"-loss: ", MSE(pred.squeeze(0), Y_train[n].to(device)))

    ax[0, i].plot(X_train[n].squeeze(0))
    ax[0, i].plot(train_ecg[n].squeeze(0))
    
    ax[1, i].plot(train_ecg[n].squeeze(0))
    ax[1, i].plot(Y_train[n].squeeze(0))
    
    ax[2, i].plot(train_ecg[n].squeeze(0))
    ax[2, i].plot(to_np(pred).squeeze(0).squeeze(0))


print("############################# Validation  ############################")
fig, ax = plt.subplots(3, 4 ,figsize=(18,9))
plt.suptitle('Validation')

for i in range(4):
    n = random.randrange(0, len(X_test))

    pred = model(X_test[n].unsqueeze(0).to(device))    
    print(n,"-loss: ", MSE(pred.squeeze(0), Y_test[n].to(device)))
    
    ax[0, i].plot(X_test[n].squeeze(0))
    ax[0, i].plot(test_ecg[n].squeeze(0))
    
    ax[1, i].plot(test_ecg[n].squeeze(0))
    ax[1, i].plot(Y_test[n].squeeze(0))
    
    ax[2, i].plot(test_ecg[n].squeeze(0))
    ax[2, i].plot(to_np(pred).squeeze(0).squeeze(0))

    

############################# MSE & RMSE ############################

'''print("############################# MSE & RMSE ###########################")

mae_total_loss = 0
mse_total_loss = 0
correlation_loss = 0

for train_step, (train, ECG) in enumerate(train_loader):
      sys.stdout.write("\r %d/%d  " % (train_step+1,len(train_loader)))
      train = train.to(device)
      ECG = ECG.to(device)
      
      #output = model1(train)
      output = model(train)
      
      mae_loss = MAE(output, ECG)
      mse_loss = MSE(output, ECG)
      corr_loss = pearsonr_correlation(output, ECG)
      
      mae_total_loss = mae_total_loss + to_np(mae_loss)
      mse_total_loss = mse_total_loss + to_np(mse_loss)
      correlation_loss = correlation_loss + to_np(corr_loss)
      
print("train_MAE",mae_total_loss/len(X_train))    
print("train_MSE",mse_total_loss/len(X_train))
print("train_correlation",correlation_loss/len(X_train))

mae_total_loss = 0
mse_total_loss = 0
correlation_loss = 0

print() 
for val_step, (val, ECG) in enumerate(val_loader):
      sys.stdout.write("\r %d/%d  " % (val_step+1,len(val_loader)))
      val = val.to(device)
      ECG = ECG.to(device)
      
      
      #output = model1(val)
      output = model(val)
      
      mae_loss = MAE(output, ECG)
      mse_loss = MSE(output, ECG)
      corr_loss = pearsonr_correlation(output, ECG)
      
      mae_total_loss = mae_total_loss + to_np(mae_loss)
      mse_total_loss = mse_total_loss + to_np(mse_loss)
      correlation_loss = correlation_loss + to_np(corr_loss)
      
print("test_MAE",mae_total_loss/len(X_test))
print("test_MSE",mse_total_loss/len(X_test))        
print("test_correlation",correlation_loss/len(X_test))

mae_total_loss = []
mse_total_loss = []
correlation_loss = []

print() '''



loss = np.load('training_loss_peak.npz',allow_pickle=True)
training_loss = loss['training_loss']
validation_loss = loss['validation_loss']

num_epochs = len(training_loss)

plt.figure(figsize=(9,5))
plt.plot(range(num_epochs), training_loss, color="blue", label='Training_loss')
plt.plot(range(num_epochs), validation_loss, color="green", label='validation_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()