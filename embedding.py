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
from Encoder import Conformer
from Decoder import Conformer2
from scipy import signal
from audtorch.metrics.functional import pearsonr
from torchsummary import summary

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

data = np.load('train_all_seg.npz', allow_pickle=True)

X_train = data["X_train"].astype(float)
Y_train = data["Y_train"].astype(float)


X_test = data["X_test"].astype(float)
Y_test = data["Y_test"].astype(float)


X_train = X_train.reshape(-1, 1, input_sequence_length)
Y_train = Y_train.reshape(-1, 1, output_sequence_length, 6)


X_test = X_test.reshape(-1, 1, input_sequence_length)
Y_test = Y_test.reshape(-1, 1, output_sequence_length, 6)


Y_train = torch.from_numpy(Y_train[:, :, :, 0]).type(torch.float32)
Y_test = torch.from_numpy(Y_test[:, :, :, 0]).type(torch.float32)

print('###################################')
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)
print('###################################')

############################# model ###################################

model1 = Conformer(num_classes=1000, input_dim=10,
                      encoder_dim=32,
                      num_encoder_layers=2,
                      num_attention_heads=32,
                      input_dropout_p=0.5,
                      feed_forward_dropout_p=0.5,
                      attention_dropout_p=0.5,
                      conv_dropout_p=0.5,
                      conv_kernel_size=31).to(device)

model1.load_state_dict(torch.load("model1_ecg.pt"))

model1.eval()
print(model1.training)

############################# Embedding  ############################

def to_np(x):
    x = x.detach()
    x = np.array(x.cpu())
    return x

New_Y_train = np.zeros((len(Y_train), 120, 32))
New_Y_test = np.zeros((len(Y_test), 120, 32))

print("############################# Training  ############################")

for i in range(len(Y_train)):
    sys.stdout.write("\r %d/%d  " % (i+1,len(Y_train)))
    pred = model1(Y_train[i].unsqueeze(0).to(device))
    New_Y_train[i] = to_np(pred)
print() 
    
print("############################# Validation  ############################")


for i in range(len(Y_test)):
    sys.stdout.write("\r %d/%d  " % (i+1,len(Y_test)))
    pred = model1(Y_test[i].unsqueeze(0).to(device))
    New_Y_test[i] = to_np(pred)
print()

print('###################################')
print("X_train: ", X_train.shape)
print("Y_train: ", New_Y_train.shape)

print("X_test: ", X_test.shape)
print("Y_test: ", New_Y_test.shape)
print('###################################')


n = random.randrange(0, len(New_Y_train))
plt.figure()
plt.imshow(New_Y_train[n].T, aspect='auto')

plt.figure()
plt.plot(Y_train[n].T)


n = random.randrange(0, len(New_Y_test))
plt.figure()
plt.imshow(New_Y_test[n].T, aspect='auto')

plt.figure()
plt.plot(Y_test[n].T)

plt.show()    

np.savez('train_all_seg_em.npz', X_train=X_train, X_test=X_test, Y_train=New_Y_train, Y_test=New_Y_test)
