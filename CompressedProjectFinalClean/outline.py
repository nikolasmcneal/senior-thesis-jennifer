import torch
import torch.utils.data
import numpy as np
import csv 
import time

# Data preprocessing 
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.X_inner_indicies = np.asarray([j for x in X for j in range(len(x))]).flatten()
    self.X_inner_lens = np.asarray([len(x) for x in X for x_i in x])
    self.X_outer_indicies = np.asarray([i for i in range(len(X)) for j in range(len(X[i]))]).flatten()

    self.Y = Y

  def __len__(self):
    return len(self.Y)

  def __getitem__(self, index):
    CONTEXT = 17
    left_padding = (-1*(self.X_inner_indicies[index] - CONTEXT)) if ((self.X_inner_indicies[index] - CONTEXT) < 0) else 0
    curr_array_end_ind = self.X_inner_lens[index] - 1
    right_padding = (CONTEXT-(curr_array_end_ind-self.X_inner_indicies[index])) if ((self.X_inner_indicies[index] + CONTEXT) > (curr_array_end_ind)) else 0
    left_beg = self.X_inner_indicies[index] - CONTEXT if (left_padding == 0) else 0
    right_end = self.X_inner_indicies[index] + CONTEXT + 1 if (right_padding == 0) else self.X_inner_lens[index]
    X = np.pad(self.X[self.X_outer_indicies[index]][left_beg:right_end],((left_padding,right_padding),(0,0))).astype(np.float32).reshape(-1)

    Y = self.Y[index].astype(np.int_)
    return X, Y

# Model architecture
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = torch.nn.Linear(455,2750)
    self.layer2 = torch.nn.ReLU()
    self.layer3 = torch.nn.BatchNorm1d(2750)
    
    self.layer4 = torch.nn.Linear(2750,2400)
    self.layer5 = torch.nn.ReLU()
    self.layer6 = torch.nn.BatchNorm1d(2400)
    
    self.layer7 = torch.nn.Linear(2400,2000)
    self.layer8 = torch.nn.ReLU()
    self.layer9 = torch.nn.BatchNorm1d(2000)

    self.layer10 = torch.nn.Linear(2000,1750)
    self.layer11 = torch.nn.ReLU()
    self.layer12 = torch.nn.BatchNorm1d(1750)
    
    self.layer13 = torch.nn.Linear(1750,1200)
    self.layer14 = torch.nn.ReLU()
    self.layer15 = torch.nn.BatchNorm1d(1200)

    self.layer16 = torch.nn.Linear(1200,750)
    self.layer17 = torch.nn.ReLU()
    self.layer18 = torch.nn.BatchNorm1d(750)
   
    self.layer19 = torch.nn.Linear(750,500)
    self.layer20 = torch.nn.ReLU()
    self.layer21 = torch.nn.BatchNorm1d(500)
    
    self.layer22 = torch.nn.Linear(500,346)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.layer8(out)
    out = self.layer9(out)
    out = self.layer10(out)
    out = self.layer11(out)
    out = self.layer12(out)
    out = self.layer13(out)
    out = self.layer14(out)
    out = self.layer15(out)
    out = self.layer16(out)
    out = self.layer17(out)
    out = self.layer18(out)
    out = self.layer19(out)
    out = self.layer20(out)
    out = self.layer21(out)
    out = self.layer22(out)
    return out


# Load the training data and perform preprocessing
train_data = np.load("train.npy", allow_pickle=True)
train_labels = np.hstack(np.load("train_labels.npy", allow_pickle=True))

train_dataset = MyDataset(train_data, train_labels)
train_loader_args = dict(shuffle = True, batch_size = 1000, num_workers = 4, pin_memory = True)
train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_args)

# Initialize the model, criterion, and optimizer
model = Model()
device = torch.device("cuda")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Load the dev validation data and perform preprocessing
dev_data = np.load("dev.npy", allow_pickle=True)
dev_labels = np.hstack(np.load("dev_labels.npy", allow_pickle=True))
dev_dataset = MyDataset(dev_data, dev_labels)
dev_loader_args = dict(shuffle = False, batch_size = 1000, num_workers = 4, pin_memory = True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, **dev_loader_args)

# Model training
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for (x,y) in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()

    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss

# Model testing
def test_model(model, test_loader,criterion):
  with torch.no_grad():
    model.eval()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0

    for (x,y) in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += y.size(0)
        correct_predictions += (predicted == y).sum().item()

        loss = criterion(outputs, y).detach()
        running_loss += loss.item()

    running_loss /= len(test_loader)
    acc = (correct_predictions/total_predictions)*100.0
    print('Testing Loss: ', running_loss)
    print('Testing Accuracy: ', acc, '%')
    return running_loss, acc

# Train the data and perform dev validation 

lambda1 = lambda epoch: 0.90 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

NUM_EPOCHS = 4
for i in range(NUM_EPOCHS):
  train_epoch(model, train_loader, criterion, optimizer)
  test_model(model, dev_loader, criterion)
  scheduler.step()

# Perform test validation. Write the results to 'test_results.csv'.

def conversion(text):
	text = text.replace("tensor(","")
	stop_ind = text.index(",")
	return text[:stop_ind]

fields = ['id', 'label']
filename = "test_results.csv"
with open(filename, 'w') as csvfile:
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(fields)

  test_data = np.load("test.npy", allow_pickle=True)
  test_labels = np.hstack([np.zeros(len(d_i)) for d_i in test_data])
  test_dataset = MyDataset(test_data, test_labels)
  test_loader_args = dict(shuffle = False, batch_size = 1000, num_workers = 4, pin_memory = True)
  test_loader = torch.utils.data.DataLoader(test_dataset, **test_loader_args)
  curr = 0
  for (x,y) in test_loader:
      outputs = model(x)
      _, predicted = torch.max(outputs.data, 1)
      for i in range(len(predicted)):
        csvwriter.writerow([str(curr), conversion(str(predicted[i]))])
        curr += 1