import csv
import numpy as np
import torch
import math
import time
# import getenc

filename = "./BTC_JPY.csv"
enc = 'utf-8'
# enc = "ms932"
csv_file = open(filename, "r", encoding=enc, errors="", newline="" )
#リスト形式
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
loop = 2
kernel_size = 10000
learning_rate = 0.001
print(f)
for itr in range(loop):
  next(f)
latest_price_list = []
for row in f:
    latest_price_list.append(float(row[4].replace(',', '')))
latest_price_list = np.array(latest_price_list)
latest_price_list = latest_price_list.astype(np.double)

import torch
import torchvision
from torch import nn
import csv
import numpy as np
import torch
import math
import time
import random
import torch.nn.functional as F
class Line(nn.Module):
    def __init__(self,input):
        super(Line, self).__init__()
        self.layer = nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Linear(input,int(input/2)),
                     nn.ReLU()
                    ),  
               
                'layer1': nn.Sequential(
                    nn.Linear(int(input/2),int(input/4)),
                    nn.ReLU()
                    ),  
  
                  'layer2': nn.Sequential(
                    nn.Linear(int(input/4),int(input/8)),
                    nn.ReLU()
                    ),  
                    'layer3': nn.Sequential(
                    nn.Linear(int(input/8),1),


                    ),  
  
                  

                  # 'layer3': nn.Sequential(
                  #   nn.Linear(int(kernel_size/4),int(kernel_size/8)),
                  #   nn.ReLU()
                  #   ),  
                  
                  #  'layer4': nn.Sequential(
                  #   nn.Linear(int(kernel_size/8),int(kernel_size/16)),
                  #   nn.ReLU()
                  #   ),  
                  #   'layer5': nn.Sequential(
                  #   nn.Linear(int(kernel_size/16),int(kernel_size/32)),
                  #   nn.ReLU()
                  #   ),  
                  #   'layer6': nn.Sequential(
                  #   nn.Linear(int(kernel_size/32),1),

                  #   ),  
                    })

    def forward(self, price):
        z = price
 
        for _layer in self.layer.values(): 
              z = _layer(z)

        return z
def train(date,latest_price,learning_rate,kernel_size,theta,model):
  y = 0
  params_L1 = 0
  theta = theta*date
  lamda = 0.1
  l2_reg = 0
  for param in model.parameters():
      l2_reg += torch.norm(param)
  
  # params_L1 += sum(abs(params_cos)) + sum(abs(params_sin))
  # t1 = time.time()
  # for i in range(kernel_size):
  #   theta [i] = 2*date*math.pi/(i+3)
  y = model(torch.sin(theta).cuda(0),torch.cos(theta).cuda(0))
    # y += math.sin(2*date*math.pi/(i+3))*params_sin[i]+math.cos(2*date*math.pi/(i+3))*params_cos[i]
  # t2 = time.time()
  # print(0,t2-t1)
  MSELoss = nn.L1Loss()
  
  p = torch.zeros(1).cuda(0)
  p[0] = latest_price
  MSE = MSELoss(y,p)

  loss = MSE
  loss += lamda * l2_reg
  
  return model,loss
  # t2 = time.time()
  # print(1,t2-t1)

#パラメータの初期化（最初は全部0）
probably = 0
before = 0
loss = 1000000
flag_1 = True
MSELoss = nn.L1Loss()
flag_2 = True
train_length = 100
model = Line(train_length).cuda(0)
model = model.double() 
latest_price_list = torch.from_numpy(latest_price_list)
latest_price_list = latest_price_list.double()
probably = 0
beta1 = 0.5
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


# text_file = open("./drive/My Drive/sample_writ_er_row.csv", "wt")
# for y in y_list:
#       text_file.write(str(y)+"\n")
#       print(str(y))
# text_file.close()