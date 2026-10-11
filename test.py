import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import csv
import copy

import datetime
import hashlib


import os
import sys

sys.path.append('/home/zhenyu/SMT/nn/tianqin')
from tq_env import TQ_env_sim
from tq_dataloader import load_data_by_date

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import data_dict_to_data_np


from TestNet import MyNet
from trade_env import window_size
from trade_env import trade_env_sim

import os.path
import sys
# setting path
sys.path.append('..')

from sklearn.preprocessing import StandardScaler

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def getData(test_data_np, i, window_size):
    input_np = np.copy(test_data_np[i-window_size:i,:])
    return input_np

def hash_nparray(arr: np.ndarray) -> str:
    """
    生成跨平台一致的 SHA-256 哈希字符串。
    """
    # 组合 dtype, shape 和数据字节
    header = f"{arr.dtype}:{arr.shape}".encode('utf-8')
    data = arr.tobytes()
    return hashlib.sha256(header + data).hexdigest()

#input_files = [
#"sine_wave",
#]


#model_file_name = "best_model.pth"
#model_file_name = "best_model_short_pred.pth"
#model_file_name = "best_model_pred_14.pth"
#model_file_name = "best_model_pred_14_kill_low_no_pos.pth"
#model_file_name = "best_model_pred_14_kill_low.pth"
model_file_name = "best_model_pred_14_kill_low_tanh.pth"
#model_file_name = "best_model_pred_14_kill_low_6ff24ab.pth"


net_input_dim = window_size*8+1
net = MyNet(net_input_dim)
param_cnt = count_parameters(net)
print("param_cnt: %d" %(param_cnt))

## load model
if os.path.isfile(model_file_name):
  print("Found weight file: [%s], loading ..." %(model_file_name))
  checkpoint = torch.load(model_file_name)
  net.load_state_dict(checkpoint['model_state_dict'])
else:
  print("Cannot found weight file: [%s], will create a new one!" %(model_file_name))


total_balance = 0
total_return_rate = 1.0
min_return_rate = 1.0


win_cnt = 0
survive_cnt = 0

symbos = [
#"DCE.a",
#"DCE.b",
#"DCE.cs",
#"DCE.m",
#"DCE.y",

"DCE.c", # 玉米
#"CZCE.TA", # PTA
#"SHFE.rb", # 螺纹钢
#"CZCE.MA", # 甲醇
#"SHFE.cu",
#"SHFE.sn",
#"INE.sc",
#"SHFE.ag",
#"SHFE.ni",

]

input_files = []

for symbo in symbos:

    #start_date = datetime.date(2025, 1, 1)
    #end_date = datetime.date(2026, 5, 30)

    #start_date = datetime.date(2026, 6, 1)
    #end_date = datetime.date(2026, 9, 30)

    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2025, 1, 1)

    # 生成日期范围
    date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    for date in date_range:
        input_files.append((symbo, date.year, date.month, date.day))


valid_input_cnt = 0
trade_days = 0
for input_file in input_files:

    #period = 300 # 5 min
    period = 60 # 1 min

    data_dict = load_data_by_date(period, input_file[0], input_file[1], input_file[2], input_file[3])
    if len(data_dict) == 0:
        print("skipping input")
        print(input_file)
        continue

    #print("data_dict")
    #print(data_dict)
    data_np = data_dict_to_data_np(data_dict)

    if(len(data_np) <= window_size):
      #print("data_np too less data [%d], skipping"%(len(data_np)))
      continue

    #data_np_hash = hash_nparray(data_np[0,:])
    #print("data_np_hash")
    #print(data_np_hash)

    #print("len(data_np)")
    #print(len(data_np))
    
    balance, num_trades, status = trade_env_sim(net, data_np, log_info=True)

    prices = status['prices']
    buys = status['buys']
    sells = status['sells']
    balances = status['balances']
    
    
    print("Case [%s] has balance [%f] with [%d] buys and [%d] sells"%(input_file, balance, len(buys), len(sells)))

    #input_file_name = input_file.replace(".csv", "")
    input_file_name = input_file[0]

    if len(buys) > 0 and len(sells) > 0:
      trade_days += 1

    if len(prices)>0 and len(buys) > 0 and len(sells) > 0:

        file_path = "/tmp/"+input_file_name+"_prices.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入列表，每个元素一行
            for idx in range(0, len(prices)):
                item = prices[idx]
                f.write(str(idx) +","+str(item)+"\n")
        f.close()

        file_path = "/tmp/"+input_file_name+"_buys.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入列表，每个元素一行
            for idx in range(0, len(buys)):
                item = buys[idx]
                f.write(str(item) +","+str(prices[item])+"\n")
        f.close()


        file_path = "/tmp/"+input_file_name+"_sells.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入列表，每个元素一行
            for idx in range(0, len(sells)):
                item = sells[idx]
                f.write(str(item) +","+str(prices[item])+"\n")
        f.close()


        file_path = "/tmp/"+input_file_name+"_balances.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入列表，每个元素一行
            for idx in range(0, len(balances)):
                f.write(str(idx) +","+str(balances[idx])+"\n")
        f.close()

    initial_balance = 100000.0 

    total_balance += balance
    temp_return_rate = balance / initial_balance
    total_return_rate  = total_return_rate * temp_return_rate
    min_return_rate = min(min_return_rate, temp_return_rate)
    #print("temp_return_rate = %f, total_return_rate = %f"%(temp_return_rate, total_return_rate))
    valid_input_cnt += 1

    if(balance > initial_balance):
      win_cnt += 1
    if(balance >= initial_balance):
      survive_cnt += 1

total_balance = 1.0*total_balance/valid_input_cnt
print("[%d/%d] days are trade-days."%(trade_days, valid_input_cnt))
print("total_balance = [%f]"%(total_balance))
avg_return_rate = total_return_rate**(1/valid_input_cnt)
print("total_return_rate = [%f] over [%d] days, avg rate = [%f]"%(total_return_rate, valid_input_cnt, avg_return_rate))
print("min_return_rate = [%f]"%(min_return_rate))
print("win_cnt")
print(win_cnt)
print("win rate= [%f]"%(1.0*win_cnt/valid_input_cnt))
print("survive rate= [%f]"%(1.0*survive_cnt/valid_input_cnt))
