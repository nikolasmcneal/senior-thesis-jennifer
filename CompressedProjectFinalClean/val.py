import torch
import os
import numpy as np
import librosa
import torchvision
import soundfile as sf
import scipy

from torch.utils.data import DataLoader
from torch.autograd import Variable
from prednet import PredNet
from sklearn import manifold

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    #print(tensor.numpy().shape)
    im = Image.fromarray(tensor.numpy()[0].astype(np.uint8))
    im.save(filename)
#from scipy.misc import imshow, imsave

batch_size = 10
A_channels = (1, 8, 16, 32)
R_channels = (1, 8, 16, 32)
nt = 27 #5

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X):
    self.X = X

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    return self.X[index]

# Load the testing data and perform preprocessing
data_prefix = "../../../../user_data/jjhuang1/val_data"
test_data = []
num_files = 0
for filename in os.listdir(data_prefix):
  if filename.endswith(".wav"): 
    file_location = "./" + data_prefix + "/" + filename
    num_files += 1
    y, sr = librosa.load(file_location, sr=None)
    complete_melSpec = librosa.feature.melspectrogram(y=y, sr=sr, window=scipy.signal.hanning)
    complete_melSpec_db = librosa.power_to_db(complete_melSpec, ref=np.max)
    complete_melSpec_db_norm = (complete_melSpec_db * (255.0/80.0)) + 255.0
    complete_melSpec_db_norm_rot = np.rot90(complete_melSpec_db_norm.copy(),2)
    complete_melSpec_db_norm = torch.unsqueeze(torch.from_numpy(complete_melSpec_db_norm_rot.copy()),0)

    for j in range(1): #20
      curr = []
      curr_x = 0
      WINDOW_SIZE = 44
      SHIFT = 8
      for i in range(nt):#5):
        melSpec_db_norm = complete_melSpec_db_norm[0,:,(curr_x):(curr_x+WINDOW_SIZE)]
        curr.append(melSpec_db_norm.numpy())
        curr_x += SHIFT
      if (len(curr) == nt): #5):
        test_data.append(np.asarray(curr))
print("num frame sequences:", len(test_data))
test_dataset = MyDataset(test_data)
test_loader_args = dict(shuffle = False, batch_size = 1, num_workers = 4, pin_memory = True)
test_loader = DataLoader(test_dataset, **test_loader_args)

model = PredNet(R_channels, A_channels, output_mode='prediction')
model.load_state_dict(torch.load('training_upperlayerwts.pt'))

loss = torch.nn.MSELoss()
running_loss = 0.0

if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

frame_errors = [0] * nt
for i, inputs in enumerate(test_loader):
    inputs = Variable(inputs.cuda())
    origin = inputs.data.cpu().byte()[:, nt-1]
    save_image(origin, "origin.jpg") 

    preds,R_0 = model(inputs)
    
    for j in range(nt):
      pred = preds[j]
      pred = pred.data.cpu()#.byte()
      pred = torch.squeeze(pred, 1)
      origin = inputs.data.cpu().byte()[:,j]
      frame_errors[j] += loss(origin.float(),pred.float()).item()
    
for i in range(nt):
    print(i, frame_errors[i]/100)

