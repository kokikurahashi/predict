import csv
import numpy as np
import torch
import math
filename = "./7203_2019.csv"
csv_file = open(filename, "r", encoding="ms932", errors="", newline="" )
#リスト形式
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
loop = 2
kernel_size = 10
learning_rate = 0.001

for itr in range(loop):
  next(f)
latest_price_list = []
for row in f:
    latest_price_list.append(float(row[4]))
latest_price_list = np.array(latest_price_list)


def train(params_sin,params_cos,bias,date,latest_price,learning_rate,kernel_size):
  y = 0
  bias[0] = torch.tensor(bias[0],requires_grad=True)
  for i in range(kernel_size):
    params_cos[i] = torch.tensor(params_sin[i],requires_grad=True)
    params_sin[i] = torch.tensor(params_cos[i],requires_grad=True)
    y += math.sin(2*date*math.pi/(i+3))*params_sin[i]+math.cos(2*date*math.pi/(i+3))*params_cos[i]
  y+=bias[0]
  MSE = ((y - latest_price)**2).mean()
  with torch.no_grad():
    MSE.backward()
  for i in range(kernel_size):
    params_sin[i] = params_sin[i] - params_sin[i].grad*learning_rate
    params_cos[i] = params_cos[i] - params_cos[i].grad*learning_rate
  bias[0] = bias[0] - bias[0].grad*learning_rate
  return params_sin,params_cos

params_sin = []
params_cos = []
bias = []
#パラメータの初期化（最初は全部0）
for i in range(kernel_size):
  params_sin.append(torch.zeros(1, requires_grad=True))
  params_cos.append(torch.zeros(1, requires_grad=True))
bias.append(torch.zeros(1, requires_grad=True))
  
#100回パラメータを学習
for loop in range(1000):
  for i in range(latest_price_list.shape[0]):
    params_sin,params_cos = train(params_sin,params_cos,bias,i+1,latest_price_list[i],learning_rate,kernel_size)
for itr in range(latest_price_list.shape[0]):
  y = 0
  for i in range(kernel_size):
    params_cos[i] = torch.tensor(params_sin[i],requires_grad=True)
    params_sin[i] = torch.tensor(params_cos[i],requires_grad=True)
    y += math.sin(2*itr*math.pi/(i+3))*params_sin[i]+math.cos(2*itr*math.pi/(i+3))*params_cos[i]
  y+=bias[0]
  print(itr,y,latest_price_list[itr])