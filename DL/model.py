#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Subset,random_split
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size=9, hidden_size1=64, hidden_size2=32):
        super(BinaryClassificationModel, self).__init__()
        self.layer1 = nn.Linear(9, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x
    
class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.001, num_epochs=20, batch_size=32, input_size=10, hidden_size1=64, hidden_size2=32):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_size = 9
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.model = BinaryClassificationModel(input_size, hidden_size1, hidden_size2)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float().unsqueeze(1))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        outputs = self.model(torch.tensor(X).float())
        return (torch.sigmoid(outputs) > 0.5).numpy().astype(int)
    
    def predict_proba(self, X):
        self.model.eval()
        outputs = self.model(torch.tensor(X).float()).squeeze()  # .squeeze() 추가
        proba = torch.sigmoid(outputs).detach().numpy()
        return np.vstack([1-proba, proba]).T  # 확률 반환

