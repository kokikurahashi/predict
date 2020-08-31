import numpy as np
import math
import time
import torchvision
from torch import nn
import torch
import math
import time
import random
import torch.nn.functional as F
from Utilities.CSV_Reader import CSV_Reader
from models.curve_model import curve_model
import csv

filename = "./toyota.csv"
enc = "ms932"
latest_price_list = CSV_Reader.getPrices(filename,enc)

def train(date,latest_price,learning_rate,kernel_size,theta,model):
  y = 0
  theta = theta*date
  
  y = model(torch.sin(theta).cuda(0),torch.cos(theta).cuda(0))
  
  MSELoss = nn.L1Loss()
  
  p = torch.zeros(1).cuda(0)
  p[0] = latest_price
  MSE = MSELoss(y,p)

  loss = MSE
  
  return model,loss

#パラメータの初期化（最初は全部0）
kernel_size = 10000

model = curve_model(kernel_size).cuda(0)
theta = 2*math.pi*torch.ones(kernel_size)/torch.arange(3,kernel_size+3,1)
#100回パラメータを学習
beta1 = 0.5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
for loop in range(1000):
  sum = 0
  beta1 = 0.5
  for i in range(0,latest_price_list.shape[0]):
    theta = 2*math.pi*torch.ones(kernel_size)/torch.arange(3,kernel_size+3,1)
    model,loss = train(i+1,latest_price_list[i],learning_rate,kernel_size,theta,model)
    sum = loss
    sum.backward() # 誤差逆伝播
    optimizer.step()  # Generatorのパラメータ更新
    model.zero_grad()
  print(loop,sum)

for i in range(0,latest_price_list.shape[0]):
  theta = 2*math.pi*torch.ones(kernel_size)/torch.arange(3,kernel_size+3,1)
  theta = theta*i
  y = model(torch.sin(theta).cuda(0),torch.cos(theta).cuda(0))
  print(y,latest_price_list[i])