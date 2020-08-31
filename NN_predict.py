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
from models.NNmodel import NNmodel
import csv

filename = "./BTC_JPY.csv"
enc = 'utf-8'
latest_price_list = CSV_Reader.getPrices(filename,enc)

#パラメータの初期化（最初は全部0）
probably = 0
MSELoss = nn.L1Loss()
train_length = 100
model = NNmodel(train_length).cuda(0)
model = model.double() 
latest_price_list = torch.from_numpy(latest_price_list)
latest_price_list = latest_price_list.double()
beta1 = 0.5
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
for loop in range(100000):
  start = random.randint(0,latest_price_list.shape[0]-train_length-1)
  price_list = latest_price_list[start:start+train_length]
  price_list = price_list.double().cuda(0)
  mean = price_list.mean()
  var = torch.var(price_list)
  params_L1 = 0
  lamda = 0.1
  l2_reg = 0
  for param in model.parameters():
      l2_reg += torch.norm(param)
  price_list = (price_list - mean)/math.sqrt(var)
  y = model(price_list)
  # if latest_price_list[start+latest_price_list.shape[0]-140] > latest_price_list[start+latest_price_list.shape[0]-139]:
  #   y_collect = torch.ones(1).double().cuda()
  # else:
  #   y_collect = torch.zeros(1).double().cuda()
  y_collect = (latest_price_list[start+train_length].cuda(0)-mean)/math.sqrt(var)
  MSE = MSELoss(y,y_collect)
  loss = MSE + l2_reg*lamda
  loss.backward() # 誤差逆伝播
  optimizer.step()  # Generatorのパラメータ更新
  model.zero_grad()
  # print(y,y_collect,latest_price_list[start+latest_price_list.shape[0]-139])
  if (y*math.sqrt(var)+mean > latest_price_list[start+train_length-1] and  latest_price_list[start+train_length] >  latest_price_list[start+train_length-1]) or(y*math.sqrt(var)+mean < latest_price_list[start+train_length-1] and  latest_price_list[start+train_length] <  latest_price_list[start+train_length-1]):
    probably += 1
  if loop %1000 == 0:
    print("ある日の価格",latest_price_list[start+train_length].cpu().detach().numpy(),"その次の日の価格",latest_price_list[start+train_length-1].cpu().detach().numpy(),"予測価格",(y*math.sqrt(var)+mean).cpu().detach().numpy() )
    print("epoch",loop,"下がるor上がる的中確率",probably/(loop+1))