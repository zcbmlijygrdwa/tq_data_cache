


# for open too many files error:
# ulimit -n 4096  # 将限制设置为4096


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import csv
import copy
import hashlib
import io
import pandas as pd

import datetime

import multiprocessing
from tqdm import tqdm

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import data_dict_to_data_np
from trade_env import window_size

from trade_env import trade_env_sim

sys.path.append('/home/zhenyu/SMT/nn/tianqin')
from tq_env import TQ_env_sim
from tq_dataloader import load_data_by_date

from TestNet import MyNet as MyModel
#from TestNet import MyNet2 as MyModel

import os.path
import sys
# setting path
sys.path.append('..')

from sklearn.preprocessing import StandardScaler

use_parallel = True
#use_parallel = False

#num_processes = 2 
#num_processes = 4
#num_processes = 8  
num_processes = 10
#num_processes = 16
#num_processes = 32

symbos = [
#"DCE.a",
#"DCE.b",
#"DCE.c",
#"DCE.cs",
#"DCE.m",
#"DCE.y",

"DCE.c", # 玉米
#"CZCE.TA", # PTA
#"SHFE.rb", # 螺纹钢
#"CZCE.MA", # 甲醇
#"SHFE.cu",
#"SHFE.sn",
#"SHFE.sn",
#"INE.sc",
#"SHFE.ag",
#"SHFE.ni",

]

input_files = []

for symbo in symbos:

    #start_date = datetime.date(2026, 1, 1)
    #end_date = datetime.date(2026, 3, 30)

    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2026, 5, 30)

    # 生成日期范围
    date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    for date in date_range:
        input_files.append((symbo, date.year, date.month, date.day))

#population_size = 1
#population_size = 10
#population_size = 20
population_size = 110

#extra_new_model_cnt = 5
extra_new_model_cnt = 50

#topk_k = 10
topk_k = 20

#mutation_rate = 0.01
#mutation_rate = 0.1
#mutation_rate = 0.5
mutation_rate = 4.0

#model_file_name = "best_model.pth"
#model_file_name = "best_model_short_pred.pth"
#model_file_name = "best_model_pred_14.pth"
#model_file_name = "best_model_pred_14_kill_low_no_pos.pth"
#model_file_name = "best_model_pred_14_kill_low.pth"
model_file_name = "best_model_pred_14_kill_low_tanh.pth"




def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Genetic Algorithm Helpers
# -------------------------
def mutate_model(model, mutation_rate=0.01):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in new_model.parameters():
            noise = torch.randn_like(param) * mutation_rate
            param += noise
    return new_model

def diff_to_target_old(diff):
    delta_ratio_diff = 0.1
    if(diff > delta_ratio_diff):
        target_np = 1
    elif(diff < -1*delta_ratio_diff):
        target_np = 0
    else:
        target_np = 2
    return target_np

def diff_to_target(diff, std):
    if(std < 0.01):
      target_np = 2
    else:
      delta_ratio_diff = 2
      if(diff > delta_ratio_diff*std):
          target_np = 1
      elif(diff < -1*delta_ratio_diff*std):
          target_np = 0
      else:
          target_np = 2
    return target_np


# =====================

net_input_dim = window_size*8+1
net = MyModel(net_input_dim)
param_cnt = count_parameters(net)
print("param_cnt: %d" %(param_cnt))

## load model
if os.path.isfile(model_file_name):
  print("Found weight file: [%s], loading ..." %(model_file_name))
  checkpoint = torch.load(model_file_name)
  net.load_state_dict(checkpoint['model_state_dict'])
else:
  print("Cannot found weight file: [%s], will create a new one!" %(model_file_name))

#criterion = nn.CrossEntropyLoss(ignore_index=2)
criterion = nn.CrossEntropyLoss()

# create your optimizer
##lr = 0.01
##lr = 0.001
##lr = 0.0001
##lr = 0.00001
##lr = 0.0000001
#lr = 0.0000000001
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

individual = {
  "net": net,
  "has_cached_fitness": False,
  "cached_fitness": 0,
}

population = [individual]
#population = [copy.deepcopy(net)]
#for _ in range(int(topk_k) - 1):
#  population.append(mutate_model(copy.deepcopy(net)))



def hash_model_weights(model):
    """
    计算模型权重的 SHA256 哈希值
    """
    # 1. 获取模型的状态字典
    state_dict = model.state_dict()

    # 2. 将状态字典保存到内存缓冲区 (BytesIO)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)

    # 3. 获取字节数据并计算哈希
    data = buffer.getvalue()
    sha256_hash = hashlib.sha256(data).hexdigest()

    return sha256_hash

def min_trade_days_panelty_kill(fitness, min_trade_days, trade_days):
    #print("min_trade_days")
    #print(min_trade_days)
    #print("trade_days")
    #print(trade_days)
    
    if trade_days < min_trade_days:
        fitness = -1e6

    return fitness

def min_trade_days_panelty_discount(fitness, min_trade_days, trade_days):
    #print("min_trade_days")
    #print(min_trade_days)
    #print("trade_days")
    #print(trade_days)
    
    if trade_days < min_trade_days:
        fitness -= 0.1 * (
            (min_trade_days - trade_days) / min_trade_days
        ) ** 2

    return fitness

# 定义要并行处理的函数
def compute_something(temp_individual):

    if temp_individual['has_cached_fitness']:
      #print("skipped cached individual, fitness[%f]"%(temp_individual['cached_fitness']))
      return temp_individual['cached_fitness'], temp_individual

    #period = 1200 # 20 min
    #period = 300 # 5 min
    period = 60 # 1 min
    
    # using files
    balances = []
    dds = []
    num_trades = 0
    trade_days = 0
    valide_days = 0
    actions = []

    temp_model = temp_individual['net']

    #model_hash = hash_model_weights(temp_model)
    #print("model_hash")
    #print(model_hash)

    overall_return_rate = 1.0

    for input_file in input_files:

      data_dict = load_data_by_date(period, input_file[0], input_file[1], input_file[2], input_file[3])
      #print("input_file")
      #print(input_file)
      #print("data_dict")
      #print(data_dict)
      ##exit()

      # data sanity checks
      if len(data_dict) == 0:
        #print("Error: empty data_dict")
        #print(input_file)
        continue

      data_list_timestamp = data_dict['data_list_timestamp']
      if( len(data_list_timestamp) != len(set(data_list_timestamp))):
        print("Error: duplicated timestamp found!")
        print(input_file)
        exit()

      found_empty_key = False
      for key in data_dict:
        if(len(data_dict[key]) == 0):
          found_empty_key = True
          break
      if found_empty_key:
        #print("Skipping empty [%s] data date"%(key))
        #print(input_file)
        continue

      keys_to_check = [
          "data_list_open",
          "data_list_high",
          "data_list_low",
          "data_list_close",
          "data_list_volume",
          "data_list_open_interest",
      ]

      found_invalid_range = False
      for key in keys_to_check:
          min_v = min(data_dict['data_list_open'])
          max_v = max(data_dict['data_list_open'])
          if(min_v == max_v):
            found_invalid_range = True
            break
      if found_invalid_range:
          print("Warning invalid range for [%s], skipping date ..."%(key))
          print(input_file)
          print(data_dict)
          exit()
          continue


      data_np = data_dict_to_data_np(data_dict)
      if(len(data_np) == 0):
        #print("skip empty data_np")
        continue

      #print("data_dict")
      #print(data_dict)
      #print(input_file)
      #exit()
      temp_balance, temp_num_trades, status = trade_env_sim(temp_model, data_np, log_info=False)

      #temp_balance, temp_num_trades, status = trade_env_sim(temp_model, data_np, log_info=True)
      #print(input_file)
      #print("balance [%f] buys [%d] sells[%d]"%(temp_balance, len(status['buys']), len(status['sells'])))
      if(temp_num_trades>0):
        trade_days += 1
      num_trades += temp_num_trades
      valide_days += 1

      actions.extend(status['actions'])

      balances.append(temp_balance)
      dds.append(status['max_draw_down'])


    #action_hash = hash(tuple(actions))
    #print("action_hash")
    #print(action_hash)
    #print("actions")
    #print(actions)
    #print("balances")
    #print(balances)
    
    #print("num_trades = %d"%(num_trades))
    balances_np = np.array(balances)
    dds_np = np.array(dds)

    assert len(balances_np) == len(dds_np)

    initial_balance = 100000.0

    return_rate_np = balances_np/initial_balance -1.0
    #median_period_return = np.median(return_rate_np) # but can easily ignore the small improvement in the beginning
    #c = 0.5
    #c = 0.2
    c = 0.05
    #print("return_rate_np")
    #print(return_rate_np)
    #print("np.clip(return_rate_np, None, c)")
    #print(np.clip(return_rate_np, None, c))

    clipped_return_rate_np = np.clip(return_rate_np, None, c)
    #print("clipped_return_rate_np")
    #print(clipped_return_rate_np)
    robust_return_rate_np = np.mean(clipped_return_rate_np) # no lower limit
    #print("robust_return_rate_np")
    #print(robust_return_rate_np)

    for i in range(0, clipped_return_rate_np.shape[0]):
      overall_return_rate = overall_return_rate * (1.0+clipped_return_rate_np[i])
    #print("overall_return_rate")
    #print(overall_return_rate)
    #exit()

    ## midean maxDD 最大回撤中位数
    #maxDD_Median = np.median(dds_np)

    ## 90% quantile maxDD 最大回撤90%分位
    ## NumPy 方法（默认线性插值）
    #maxDD_Median_90_quantile = np.percentile(dds_np, 90)

    ## Herfindahl
    #Herfindahl = 0
    #profits_np = balances_np-initial_balance
    #profits_np = profits_np[profits_np>0] # only keep positive profits
    #profits_non_zero = profits_np.sum()
    #for i in range(0, len(profits_np)):
    #  w = profits_np[i]/profits_non_zero
    #  Herfindahl += (w*w)


    #total_balance = balances_np.sum()
    #total_balance_avg = balances_np.mean()
    #total_balance_min = balances_np.min()
    #total_balance_max = balances_np.max()
    


    ## using TQ
    #total_balance, status = TQ_env_sim(temp_model, log_info=False)

    #print("total_balance = %f"%(total_balance))

    #fitness = total_balance_avg
    #fitness = total_balance_min

    #l1 = 0.5
    #l2 = 1.0
    ##l3 = 0.5
    #l3 = 0.01
    #fitness = median_period_return - l1*maxDD_Median - l2*maxDD_Median_90_quantile - l3*Herfindahl
    #fitness = robust_return_rate_np - l1*maxDD_Median - l2*maxDD_Median_90_quantile - l3*Herfindahl
    #fitness = robust_return_rate_np - l3*Herfindahl
    #fitness = robust_return_rate_np 
    fitness = overall_return_rate
    #print("balances_np")
    #print(balances_np)
    #print("robust_return_rate_np")
    #print(robust_return_rate_np)
    #print("maxDD_Median")
    #print(maxDD_Median)
    #print("maxDD_Median_90_quantile")
    #print(maxDD_Median_90_quantile)
    #print("Herfindahl")
    #print(Herfindahl)
    #print("fitness")
    #print(fitness)
    #exit()

    ## min trading day panelty
    #min_trade_days = int(0.3 * valide_days)  # 例如至少5%的天有交易
    #fitness = min_trade_days_panelty_kill(fitness, min_trade_days, trade_days)

    return fitness, temp_individual




with torch.no_grad(): # no need to use grad, as we use GA

    for epoch in range(0,10000000000):
        overall_balance = 0
    
    
        # Generate next generation by filling up mutated new individual to population size

        if(len(population) == 0):
          print("Creating whole new generation...")
          for i in range(0, population_size):
            temp_individual = {
              "net": MyModel(net_input_dim),
              "has_cached_fitness": False,
               "cached_fitness": 0,
            }
            population.append(temp_individual)
        else:
          mutated_population = []
          while  len(population) + len(mutated_population) < population_size:
            for individual in population:
              temp_individual = {
                "net": mutate_model(individual['net'], mutation_rate),
                "has_cached_fitness": False,
                "cached_fitness": 0,
              }
              mutated_population.append(temp_individual)
              if len(population) + len(mutated_population) == population_size:
                population.extend(mutated_population)
                break


        # some extra new models, completely random
        for idx in range(0, extra_new_model_cnt):
            temp_individual = {
              "net": MyModel(net_input_dim),
              "has_cached_fitness": False,
              "cached_fitness": 0,
            }
            population.append(temp_individual)
    
        if use_parallel:
            # 使用 multiprocessing.Pool 来并行处理
            with multiprocessing.Pool(processes=num_processes) as pool:
                # 使用 tqdm 来监控进度
                results = []
                for result in tqdm(pool.imap_unordered(compute_something, population), total=len(population)):
                    temp_fitness = result[0]
                    if(temp_fitness > -100):
                        results.append(result)
                    else:
                        print("Skipping low fitness %f)"%(temp_fitness))
            
        else:
          results = []
          for temp_individual in tqdm(population):
              temp_fitness, temp_individual = compute_something(temp_individual)

              if(temp_fitness > -100):
                  results.append((temp_fitness, temp_individual))
              #else:
              #    print("Skipping low fitness %f)"%(temp_fitness))

        # collect individual_healths
        individual_healths = []
        individual_instances = []
        for result in results:
          #print("balance")
          #print(result[0])
          #individual_healths.append(result[0])
          individual_healths.append(result[0])
          individual_instances.append(result[1])

    
        population = []
        if(len(individual_healths) >= topk_k):
            fitness_top_values, fitness_top_ids = torch.topk(torch.tensor(individual_healths), topk_k)
            best_fitnesss = fitness_top_values[0]
            best_net = individual_instances[fitness_top_ids[0]]['net']
            print("fitness_top_values")
            print(fitness_top_values)
    
            for idx in fitness_top_ids:
              temp_individual = individual_instances[idx]
              temp_individual['has_cached_fitness'] = True
              temp_individual['cached_fitness'] = individual_healths[idx]
              #population.append(copy.deepcopy(individual_instances[idx]))
              population.append(temp_individual)
    
    
            # 打开文件用于写入，如果文件不存在则创建
            print("open for writing")
            with open('balances.txt', 'a') as file:
                file.write(str(best_fitnesss.item())+'\n')
    
    
            #if epoch > 0 and epoch % 100 == 0:
            if epoch >= 0 :
                # save model
                torch.save({
                            'model_state_dict': best_net.state_dict(),
                    }, model_file_name)
                print("Model saved to [%s]"%(model_file_name))
        else:
          print("Warning: No individual survived in this round.")


