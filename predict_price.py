import csv
import numpy as np
import torch
import math
import time
filename = "./4755_2018.csv"
csv_file = open(filename, "r", encoding="ms932", errors="", newline="" )
#リスト形式
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
loop = 2
kernel_size = 10000
learning_rate = 0.0001

for itr in range(loop):
  next(f)
latest_price_list = []
for row in f:
    latest_price_list.append(float(row[4]))
latest_price_list = np.array(latest_price_list)

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
    def __init__(self,kernel_size):
        super(Line, self).__init__()
        self.layer = nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Linear(kernel_size*2,1),
                    ),  
                })
    def forward(self, sin,cos):
        z = torch.cat((sin,cos))
        for _layer in self.layer.values(): 
              z = _layer(z)
        return z
def train(date,latest_price,learning_rate,kernel_size,theta,model):
  y = 0
  params_L1 = 0
  theta = theta*date
  # params_L1 += sum(abs(params_cos)) + sum(abs(params_sin))
  # t1 = time.time()
  # for i in range(kernel_size):
  #   theta [i] = 2*date*math.pi/(i+3)
  y = model(torch.sin(theta).cuda(0),torch.cos(theta).cuda(0))
    # y += math.sin(2*date*math.pi/(i+3))*params_sin[i]+math.cos(2*date*math.pi/(i+3))*params_cos[i]
  # t2 = time.time()
  # print(0,t2-t1)
  MSELoss = nn.L1Loss()
  beta1 = 0.5
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
  p = torch.zeros(1).cuda(0)
  p[0] = latest_price
  MSE = MSELoss(y,p)
  loss = MSE
  if date == 200:
    print(MSE)
  loss.backward() # 誤差逆伝播
  optimizer.step()  # Generatorのパラメータ更新
  model.zero_grad() 
  return model
  # t2 = time.time()
  # print(1,t2-t1)
model = Line(kernel_size).cuda(0)
theta = 2*math.pi*torch.ones(kernel_size)/torch.arange(3,kernel_size+3,1)
print(theta.shape)
#パラメータの初期化（最初は全部0）

  
#100回パラメータを学習
for loop in range(1000):
  for i in range(latest_price_list.shape[0]-140):
    theta = 2*math.pi*torch.ones(kernel_size)/torch.arange(3,kernel_size+3,1)
    model = train(i+1,latest_price_list[i],learning_rate,kernel_size,theta,model)

y_list = []
for itr in range(latest_price_list.shape[0]):
  y = 0
  theta = 2*math.pi*torch.ones(kernel_size)/torch.arange(3,kernel_size+3,1)
  theta = theta*itr
  y = model(torch.sin(theta).cuda(0),torch.cos(theta).cuda(0))
  y = y.cpu().detach().numpy()
  y_list.append(y)

text_file = open("./drive/My Drive/sample_writ_er_row.csv", "wt")
for y in y_list:
      text_file.write(str(y)+"\n")
      print(str(y))
text_file.close()