#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/code/voltvipin/birdclassification-using-pytorch/notebook
# 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import sys
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import models


# In[2]:


# Get the available device

if torch.cuda.is_available():
    dev = "cuda:0"  # Gpu
else:
    dev = "cpu"
device = torch.device(dev)


# # dataset & dataloader
# train / test : caries/crown/filling

# In[3]:


IMG_SIZE = 224
transform = transforms.Compose([
    # transforms.RandomCrop(500),
    # transforms.Resize((IMG_SIZE, IMG_SIZE )),    
    # transforms.RandomRotation(180),
    # transforms.ToTensor(), # it does transforms.Normalize(0., 255.) also
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


trainset = torchvision.datasets.ImageFolder(root="/home/yaxin/caries/all/train/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=0, shuffle=True)

testset = torchvision.datasets.ImageFolder(root="/home/yaxin/caries/all/test/", transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0, shuffle=False)

dataloaders = {
    "train": trainloader,
    "test": testloader
}
datasizes = {
    "train": len(trainset),
    "test": len(testset)
}
# CLASSES = list(trainset.class_to_idx.keys())


# In[5]:


CLASSES


# In[4]:


def imshow(img, size=(10, 10)):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    if size:
        plt.figure(figsize=size)
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("One mini batch")
    plt.axis("off")
    plt.pause(0.001)


# In[6]:


dataiter = iter(trainloader)
images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))


# # model

# # model

# In[9]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# In[33]:


train_losses = []
test_losses =[]
train_accs = []
test_accs = []

def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs))
        print("-"*10)
        
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0 
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parametsrs
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == "train":
                scheduler.step()
            
            epoch_loss = running_loss / datasizes[phase]
            if phase == "train":
                train_losses.append(epoch_loss)
            epoch_acc = running_corrects.double()/datasizes[phase]
            if phase == "test":
                test_accs.append(epoch_acc)

            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if(phase == "test" and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    
    time_elapsed = time.time() - since
    print("Training complete in {:0f}m {:0f}s".format(time_elapsed//60, time_elapsed%60))
    print("Best val Acc: {}:4f".format(best_acc))
    
    # load best model parameters
    model.load_state_dict(best_model_wts)
    return model


# In[16]:


pip install --upgrade ipywidgets


# In[32]:


test_accs


# In[11]:


model_ft = models.resnet18(pretrained=True)

# turn training false for all layers, other than fc layer
for param in model_ft.parameters():
    param.requires_grad = False
    
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(CLASSES))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.003, momentum=0.9)
exp_lr_sc = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[19]:


pip install tqdm==4.40.0


# In[35]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_sc, epochs=200)


# In[36]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].plot(train_losses)
axes[0].set_title("train_losses")
axes[1].plot(test_accs)
axes[1].set_title("test_accs")


# In[37]:


FILE = '/home/yaxin/caries/class/model/resnet18_200.pt'
torch.save({
                'epoch': 200,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
                }, FILE)


#  # Predict

# In[46]:


def imshowaxis(ax, img, orig, pred):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    if orig != pred:
        ax.set_title(orig + "\n" + pred, color="red")
    else:
        ax.set_title(orig + "\n" + pred)
    ax.axis("off")


def vis_model(model, num_images=15):
    was_training = model.training
    model.eval()
    images_so_far = 0
    figure, ax = plt.subplots(3, 5, figsize=(20, 20))
    
    
    with torch.no_grad():
        for i , (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            # print(inputs)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(5):
                for j in range(5):
                    if images_so_far < num_images:
                        imshowaxis(ax[i][j], inputs.cpu().data[images_so_far], CLASSES[labels[images_so_far]], CLASSES[preds[images_so_far]])
                    else:
                        model.train(mode=was_training)
                        return
                    images_so_far += 1
        model.train(mode=was_training)


# In[47]:


# Title: Original vs Predicted 
vis_model(model_ft)

