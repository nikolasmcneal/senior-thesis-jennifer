import os
import librosa   
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from prednet import PredNet
from debug import info
import torchvision


num_epochs = 50
batch_size = 10
A_channels = (1, 8, 16, 32)
R_channels = (1, 8, 16, 32)
lr = 0.001 # if epoch < 75 else 0.0001
nt = 27 #5 # num of time steps

layer_loss_weights = Variable(torch.FloatTensor([[0.7], [0.1], [0.1], [0.1]]).cuda())
time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cuda())

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X):
    self.X = X

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    return self.X[index]

# Load the training data and perform preprocessing
data_prefix = "../../../../user_data/jjhuang1/train_data"
train_data = []
num_files = 0
for filename in os.listdir(data_prefix):
  if (filename.endswith(".wav")): 
    num_files += 1
    file_location = "./" + data_prefix + "/" + filename
    y, sr = librosa.load(file_location, sr=None)
    complete_melSpec = librosa.feature.melspectrogram(y=y, sr=sr)
    complete_melSpec_db = librosa.power_to_db(complete_melSpec, ref=np.max)
    complete_melSpec_db_norm = (complete_melSpec_db * (255.0/80.0)) + 255.0
    complete_melSpec_db_norm = np.rot90(complete_melSpec_db_norm.copy(),2)
    for j in range(1): #11
      curr = []
      curr_x = 0
      WINDOW_SIZE = 44
      SHIFT = 8
      for i in range(nt):#5):
        melSpec_db_norm = complete_melSpec_db_norm[:,(curr_x):(curr_x+WINDOW_SIZE)]
        curr.append(melSpec_db_norm)
        curr_x += SHIFT
      if (len(curr) == nt): #5):
        train_data.append(np.asarray(curr))
print("num files:", num_files)
train_dataset = MyDataset(train_data)
train_loader_args = dict(shuffle = True, batch_size = 16, num_workers = 4, pin_memory = True)
train_loader = DataLoader(train_dataset, **train_loader_args)

model = PredNet(R_channels, A_channels, output_mode='error')
if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs //2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer



for epoch in range(num_epochs):
    optimizer = lr_scheduler(optimizer, epoch)
    for i, inputs in enumerate(train_loader):
        inputs = Variable(inputs.cuda())
        errors = model(inputs) # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(errors)

        optimizer.zero_grad()

        errors.backward()

        optimizer.step()
        if i%200 == 0:
            print('Epoch: {}/{}, step: {}/{}, errors: {}'.format(epoch, num_epochs, i, len(train_dataset)//batch_size, errors.item()))

torch.save(model.state_dict(), 'training_upperlayerwts.pt')


