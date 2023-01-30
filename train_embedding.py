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
from scipy import signal
from audtorch.metrics.functional import pearsonr
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import segmentation_models_pytorch as smp
from torch.nn.functional import normalize



def to_np(x):
    x = x.detach()
    x = np.array(x.cpu())
    return x
    
############################# random seed ###################################

seed = 772
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
#device = torch.device('cpu')

######################### batch_size, epoch, LR ############################
input_sequence_length = 1000
output_sequence_length = 1000

batch_size = 16
num_epochs = 300
learning_rate = 0.0001

################################ load data ##################################

data = np.load('train_all_seg_em.npz', allow_pickle=True)

X_train = data["X_train"].astype(float)
Y_train = data["Y_train"].astype(float)
X_test = data["X_test"].astype(float)
Y_test = data["Y_test"].astype(float)

print('################ load data ################')
print('X_train',X_train.shape)
print('Y_train',Y_train.shape)
print('X_test',X_test.shape)
print('Y_test',Y_test.shape)

X_train = X_train.reshape(-1, 1, input_sequence_length)
Y_train = Y_train.reshape(-1, 120, 32)

X_test = X_test.reshape(-1, 1, input_sequence_length)
Y_test = Y_test.reshape(-1, 120, 32)


############################# split data ###################################

X_train = torch.from_numpy(X_train).type(torch.float32)
Y_train = torch.from_numpy(Y_train).type(torch.float32)

X_test = torch.from_numpy(X_test).type(torch.float32)
Y_test = torch.from_numpy(Y_test).type(torch.float32)

print('###################################')
print('X_train',X_train.shape)
print('Y_train',Y_train.shape)
print('X_test',X_test.shape)
print('Y_test',Y_test.shape)
print('###################################')

train = torch.utils.data.TensorDataset(X_train, Y_train)
val = torch.utils.data.TensorDataset(X_test, Y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

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

model1.load_state_dict(torch.load("model_peak.pt"))

model2 = Conformer_em(num_classes=1000, input_dim=10,
                      encoder_dim=32,
                      num_encoder_layers=4,
                      num_attention_heads=32,
                      input_dropout_p=0.5,
                      feed_forward_dropout_p=0.5,
                      attention_dropout_p=0.5,
                      conv_dropout_p=0.5,
                      conv_kernel_size=31).to(device)
                      
############################# loss & optimizer ###################################

#loss = nn.MSELoss()
loss = nn.L1Loss()
optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

############################# training ###################################
TT_start = time.time()  ### total training time

def NormalizeData(data):
    return (data - np.min(data, axis=-2)) / (np.max(data, axis=-2) - np.min(data, axis=-2))
    
def Normalize(outmap):
    outmap_min, _ = torch.min(outmap, dim=-2, keepdim=True)
    outmap_max, _ = torch.max(outmap, dim=-2, keepdim=True)
    outmap = (outmap - outmap_min) / (outmap_max - outmap_min)
    return outmap


def to_np(x):
    x = x.detach()
    x = np.array(x.cpu())
    return x

def train(num_epochs, model1, model2, train_loader, val_loader):
    # Train the model
    total_step = len(train_loader)

    training_loss_list = []
    validation_loss_list = []

    for epoch in range(num_epochs):
        start = time.time()

        training_loss = 0
        
        for training_step, (train, labels) in enumerate(train_loader):

            train = train.to(device)
            label = labels.to(device)
            
            output = model1(train)
            output = Normalize(output)
            output = model2(train, output)
            
            train_loss = loss(output, label)
            #print(train_loss)
           
            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            training_loss = training_loss + train_loss.item()

            if (training_step + 1) % 10 == 0:
                sys.stdout.write("\r Train Epoch: %d/%d \tLoss: %.4f  " % (epoch + 1,
                                                                           num_epochs,
                                                                           training_loss / (training_step + 1)))
        ############################## validation ######################################

        validation_loss = 0
        
        
        with torch.no_grad():
            for validation_step, (val, val_labels) in enumerate(val_loader):

                val = val.to(device)
                label = val_labels.to(device)

            
                output = model1(val)
                output = Normalize(output)
                output = model2(val, output)
                
                val_loss = loss(output, label)


                optimizer.zero_grad()
                torch.cuda.empty_cache()

                validation_loss = validation_loss + val_loss.item()
                
                if (validation_step + 1) % 10 == 0:
                    sys.stdout.write("\r Train Epoch: %d/%d \tLoss: %.4f \tVal_Loss: %.4f" % (epoch + 1,
                                                                                              num_epochs,
                                                                                              training_loss / (training_step + 1),
                                                                                              validation_loss / ( validation_step + 1)))

        end = time.time()

        print("\r Train Epoch: %d/%d \tLoss: %.4f \tVal_Loss: %.4f \ttime: %.2f" % (epoch + 1,
                                                                                    num_epochs,
                                                                                    training_loss / (training_step + 1),
                                                                                    validation_loss / (validation_step + 1),
                                                                                    end - start))

        training_loss_list.append(training_loss / (training_step + 1))
        validation_loss_list.append( validation_loss / (validation_step + 1))

        np.savez('training_loss_em', training_loss=training_loss_list, validation_loss=validation_loss_list)
        if epoch > 10 and (epoch + 1) % 50 == 0:
          torch.save(model2.state_dict(), "model_em.pt")
          lowest_val_loss = val_loss.item()
          print("Model save", "epoch: ", epoch + 1, "val_loss:", val_loss.item())
          
    ############################# save model ###########################
    
    np.savez('training_loss_em', training_loss=training_loss_list, validation_loss=validation_loss_list)
    torch.save(model2.state_dict(), "model_em.pt")
    lowest_val_loss = val_loss.item()
    print("Model save", "epoch: ", epoch + 1, "val_loss:", validation_loss / (validation_step + 1))
    pass

    return training_loss_list, validation_loss_list


training_loss, validation_loss = train(num_epochs, model1, model2, train_loader, val_loader)

############################# total time ############################

TT_end = time.time()
print("\rTotal training time: %.2f s" % (TT_end - TT_start))

############################# TEST  ############################
model.eval()

def to_np(x):
    x = x.detach()
    x = np.array(x.cpu())
    return x

print("############################# Training  ############################")
fig, ax = plt.subplots(2, 4, figsize=(18,6))
plt.suptitle('training')

for i in range(4):
    n = random.randrange(0, len(X_train))
    print(n)
    pred = model(X_train[n].unsqueeze(0).to(device))
    
    
    ax[0, i].plot(Y_train[n].squeeze(0))
    ax[0, i].plot(to_np(pred).squeeze(0).squeeze(0))
    
for i in range(4):
    n = random.randrange(0, len(X_train))
    print(n)
    pred = model(X_train[n].unsqueeze(0).to(device))
    
   
    ax[1, i].plot(Y_train[n].squeeze(0))
    ax[1, i].plot(to_np(pred).squeeze(0).squeeze(0))

print("############################# Validation  ############################")
fig, ax = plt.subplots(2,4 ,figsize=(18,6))
plt.suptitle('Validation')
for i in range(4):
    n = random.randrange(0, len(X_test))
    print(n)
   
    
    pred = model(X_test[n].unsqueeze(0).to(device))

    
    ax[0, i].plot(Y_test[n].squeeze(0))
    ax[0, i].plot(to_np(pred).squeeze(0).squeeze(0))
    

for i in range(4):
    n = random.randrange(0, len(X_test))
    print(n)
   
  
    pred = model(X_test[n].unsqueeze(0).to(device))
    
    
    ax[1, i].plot(Y_test[n].squeeze(0))
    ax[1, i].plot(to_np(pred).squeeze(0).squeeze(0))
    


plt.figure()
plt.plot(range(num_epochs), training_loss, color="blue", label='Training_loss')
plt.plot(range(num_epochs), validation_loss, color="green", label='validation_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()