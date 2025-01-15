import os
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class MainDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath) #Upload ·· x - Number, y - Labels
        self.x = self.x / 255. #Normalizamos
        self.y = F.one_hot(self.y, num_classes=10).to(float) # Enconder para clasificación por vectores
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix] # i-ésimo numero, label iésima

class MyNeuralNet(nn.Module):
	def __init__(self):    
		super().__init__()
		self.Matrix1 = nn.Linear(28**2,100)
		self.Matrix2 = nn.Linear(100,100)
		self.Matrix3 = nn.Linear(100,10)
		self.R = nn.ReLU()
	def forward(self,x):
		x = x.view(-1,28**2) #Se expande la imagen de 28x28 a un vector de 784
		x = self.R(self.Matrix1(x))
		x = self.R(self.Matrix2(x))
		x = self.Matrix3(x)
		return x.squeeze()

def CheckAccuracy(loader, model, name):
    num_corrects = 0
    num_samples = 0
    model.eval()
    x_fail = []
    y_fail = []
    yhat_fail = []
    with torch.no_grad():
        for x, y in loader:
            # send the data to the device
            x = x.to(device)
            y = y.to(device)
    
            # prepare the data for the model
            x = x.reshape(-1, 784)
    
            # forward
            y_hat = model(x)
    
            # calculations for accuracy
            _, predictions = torch.max(y_hat, dim=1)
            labels = torch.argmax(y, dim=1)

            num_corrects += (predictions == labels).sum().item()
            
            incorrect_indices = (predictions != labels).nonzero(as_tuple=True)[0]
            
            if incorrect_indices.numel() > 0: 
                x_fail.extend(x[incorrect_indices]) 
                y_fail.extend(y[incorrect_indices]) 
                yhat_fail.extend(y_hat[incorrect_indices]) 
            
            num_samples += y.size(0)

    Accuracy = num_corrects/num_samples*100
    print(f"Accuracy of {name} = {Accuracy:.2f}; Received {num_corrects}/{num_samples}")
    model.train()
    return Accuracy, x_fail, y_fail, yhat_fail

class Report:
    def __init__(self, epoch, loss, Accuracy, type = "Training", n_epochs=20):
        self.epoch = epoch
        self.loss = loss
        self.type = type
        self.n_epochs = n_epochs
        self.Accuracy = Accuracy
        self.epoch_data_avgd = self.epoch.reshape(self.n_epochs,-1).mean(axis=1)
        self.loss_data_avgd = self.loss.reshape(self.n_epochs,-1).mean(axis=1)
    
    def plot_training(self):
        plt.figure(figsize=(5,4))
        plt.plot(self.epoch_data_avgd, self.loss_data_avgd, 'o--', label='Loss', color="Cyan")
        plt.xlabel('Epoch Number')
        plt.ylabel('Cross Entropy')
        plt.title(f'Cross Entropy (avgd per epoch) - {self.type}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(5, 4))
        plt.plot([i for i in range(1, self.n_epochs + 1)], self.Accuracy, 'd--', label='Accuracy', color='orange')
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy (%)')
        plt.ylim(min(self.Accuracy)*0.9, 100)
        plt.xlim(0, self.n_epochs)
        plt.title(f'Accuracy Over Epochs - {self.type}')
        plt.legend()

        plt.tight_layout()
        plt.show()

def train_model(dl, f, n_epochs=20, device = "cpu"):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()
	
    # Train model
    losses = []
    epochs = []
    accuracy = []
    for epoch in range(n_epochs):
        N = len(dl)
            
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
            
        print(f"Epoch {epoch + 1} is done; loss = { sum (losses[epoch*i: (epoch + 1)*i]) / 12000 }")
        accuracy.append(CheckAccuracy(train_dl, f, "Train Data")[0])
        
    return np.array(epochs), np.array(losses), accuracy


device = torch.device("cpu" if torch.cuda.is_available() else "cpu" )
print(torch.cuda.get_device_name())

path_abs = os.path.dirname(os.path.abspath(__file__))

train_ds = MainDataset(os.path.join(path_abs, '..\\Database\\MNIST\\processed\\training.pt'))
test_ds = MainDataset(os.path.join(path_abs, '..\\Database\\MNIST\\processed\\test.pt'))

train_dl = DataLoader(train_ds, batch_size=5) # se divide el dataset en batch de a 5
test_dl = DataLoader(test_ds, batch_size=5) # se divide el dataset en batch de a 5

f = MyNeuralNet().to(device)

n_epochs = 20

epoch_data, loss_data, Accurasy_data = train_model(train_dl, f, n_epochs)

Analisis = Report(epoch_data, loss_data,Accurasy_data,  n_epochs)
Analisis.plot_accuracy(), Analisis.plot_training()


acurracy, x_fail, y_fail, yhat_fail = CheckAccuracy(test_dl, f, "Test Data")

fig, ax = plt.subplots(10,10,figsize=(20,20))
for i in range(100, 200):
    plt.subplot(10,10,i-99)
    plt.imshow(x_fail[i].reshape(28,28))
    plt.title(f'WRONG: {np.argmax(yhat_fail[i])} - Label: {np.argmax(y_fail[i])}')
fig.tight_layout()
plt.show()