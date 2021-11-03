import os
import sys
import torch
import json
import data
from models.BasicModel import BasicModel

def train(data,optimizer,model,n_epochs,loss_function,name_model):
    logs =[]
    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    for epoch in range(1, n_epochs + 1):
        for i,batch in enumerate(train_loader):
            model.train()

            inputs, labels = batch
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs,labels)

            loss.backward()
            optimizer.step() #update parameters
        # val loop
        model.eval()
        true_preds = 0
        for batch in val_loader: 
            inputs, labels = batch
            with torch.no_grad():
                preds = model(inputs).argmax(dim=-1)
                true_preds += (preds == labels).sum().item()
        val_acc = true_preds/len(data.val_set)
        print('Epoch {}, Training loss {}, Val accuracy {}'.format(epoch, loss, val_acc))
        logs.append('Epoch {}, Training loss: {}, Val accuracy: {}'.format(epoch, loss, val_acc))
    # write logs
    with open('TrainingLogs/logs_'+name_model+'.txt', 'w') as f:
    for line in logs:
        f.write(line)
        f.write('\n')

# prepare data

cifar100 = data.CIFAR100Dataset() 
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 50
model = BasicModel()

#train model 
train(cifar100,optimizer,model,n_epochs,loss_function,name_model ="BasicModel")