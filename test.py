
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
from model_embedding import Conformer_em
from model_peak import Conformer 
from Decoder import Conformer2
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


Y_train = Y_train[:, :, :, 0]
Y_test  =  Y_test[:, :, :, 0]

#Y_train = Y_train[:, :, 100:-100]
#Y_test  =  Y_test[:, :, 100:-100]

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

model1 = Conformer(num_classes=1000, input_dim=10,
                      encoder_dim=16,
                      num_encoder_layers=2,
                      num_attention_heads=16,
                      input_dropout_p=0.5,
                      feed_forward_dropout_p=0.5,
                      attention_dropout_p=0.5,
                      conv_dropout_p=0.5,
                      conv_kernel_size=31).to(device)
model1.load_state_dict(torch.load("model1.pt"))


model2 = Conformer_em(num_classes=1000, input_dim=10,
                      encoder_dim=32,
                      num_encoder_layers=4,
                      num_attention_heads=32,
                      input_dropout_p=0.5,
                      feed_forward_dropout_p=0.5,
                      attention_dropout_p=0.5,
                      conv_dropout_p=0.5,
                      conv_kernel_size=31).to(device)
model2.load_state_dict(torch.load("model2.pt")) 


model3 = Conformer2(num_classes=1000, input_dim=10,
                      encoder_dim=32,
                      num_encoder_layers=2,
                      num_attention_heads=32,
                      input_dropout_p=0.5,
                      feed_forward_dropout_p=0.5,
                      attention_dropout_p=0.5,
                      conv_dropout_p=0.5,
                      conv_kernel_size=31).to(device)
model3.load_state_dict(torch.load("model3.pt"))               
     
     
model1.eval()                 
model2.eval()
model3.eval()                 

print(model1.training, model2.training, model3.training)

############################# TEST  ############################

MAE = nn.L1Loss()
MSE = nn.MSELoss()
from scipy import stats

def to_np(x):
    x = x.detach()
    x = np.array(x.cpu())
    return x

def pearsonr_correlation(input, target):
    output = pearsonr(input, target)
    #output = stats.pearsonr(to_np(input), to_np(target))
    #output = np.corrcoef(to_np(input), to_np(target))
    
    return output




print("############################# Training  ############################")


save_pred = np.zeros((4, 1000))
save_targ = np.zeros((4, 1000))

fig, ax = plt.subplots(2, 4, figsize=(18,6))
plt.suptitle('training')

for i in range(4):
    n = random.randrange(0, len(X_train)) + 1

    pred = model1(X_train[n].unsqueeze(0).to(device))
    pred = model2(X_train[n].unsqueeze(0).to(device), pred)
    pred = model3(pred)
    
    #pred = pred[:,:, 100:-100]
    

    
    print(n,"-loss: ", MSE(pred.squeeze(0), Y_train[n].to(device)))
    print(n,"-loss: ", pearsonr_correlation(pred.squeeze(0), Y_train[n].to(device)))
    
    save_pred[i] = to_np(pred.squeeze(0).squeeze(0))
    save_targ[i] = to_np(Y_train[n][-1])
    
    ax[0, i].plot(X_train[n].squeeze(0))
    ax[0, i].plot(Y_train[n].squeeze(0))
    
    ax[1, i].plot(Y_train[n].squeeze(0))
    ax[1, i].plot(to_np(pred).squeeze(0).T)

plt.figure()
plt.plot(save_pred[0])
plt.plot(save_targ[0])
plt.show()

import csv

with open('example.csv', 'w') as file:

    for i in range(4):
    
      writer = csv.writer(file)
      writer.writerow(save_pred[i])
      writer.writerow(save_targ[i])
    
print("############################# Validation  ############################")
fig, ax = plt.subplots(2, 4 ,figsize=(18,6))
plt.suptitle('Validation')

for i in range(4):
    n = random.randrange(0, len(X_test)) + 1
    
    pred = model1(X_test[n].unsqueeze(0).to(device))
    pred = model2(X_test[n].unsqueeze(0).to(device), pred) 
    pred = model3(pred)
    #pred = pred[:,:, 100:-100]
    print(n,"-loss: ", MSE(pred.squeeze(0), Y_test[n].to(device)))
    print(n,"-loss: ", pearsonr_correlation(pred.squeeze(0), Y_test[n].to(device)))
    ax[0, i].plot(X_test[n].squeeze(0))
    ax[0, i].plot(Y_test[n].squeeze(0))
    
    ax[1, i].plot(Y_test[n].squeeze(0))
    ax[1, i].plot(to_np(pred).squeeze(0).T)

plt.show()   
############################# MSE & RMSE ############################

print("############################# MSE & RMSE ###########################")

mae_total_loss = 0
mse_total_loss = 0
correlation_loss = 0

for train_step, (train, ECG) in enumerate(train_loader):
      sys.stdout.write("\r %d/%d  " % (train_step+1,len(train_loader)))
      train = train.to(device)
      ECG = ECG.to(device)
      
      output = model1(train)
      output = model2(train, output)
      output = model3(output)
      
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
      
      
      output = model1(val)
      output = model2(val, output)
      output = model3(output)
      
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

print() 



loss = np.load('training_loss.npz',allow_pickle=True)
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