


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

import multiprocessing
from tqdm import tqdm

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import data_dict_to_data_np
from trade_env import window_size

from trade_env import trade_env_sim

#sys.path.append('/home/zhenyu/SMT/nn/tianqin')
#from tq_env import TQ_env_sim
from tq_dataloader import load_data_by_date

from TestNet import MyNet

import os.path
import sys
# setting path
sys.path.append('..')

from sklearn.preprocessing import StandardScaler

use_parallel = True
#use_parallel = False

#num_processes = 2 
#num_processes = 4
num_processes = 8  # 27:24
#num_processes = 10
#num_processes = 32

symbos = [
"DCE.a1809",
"DCE.b1809",
"DCE.c1809",
"DCE.cs1809",
"DCE.m1809",
#"DCE.y1809",
]

input_files = []

for symbo in symbos:
  for m in range(3,7):
    for i in range(1,30):
      year = 2018
      month = m
      day = i
      input_files.append((symbo, year, month, day))

#entries = os.listdir(data_folder)
#input_files = []
#for temp_file in entries:
#    if("csv" in temp_file):
#      input_files.append(temp_file)
#
#input_files = sorted(input_files)
##input_files = [
##"sine_wave",
##]
#
#input_files = input_files[0:1]
##input_files = input_files[0:5]
##input_files = input_files[0:30]
##input_files = ["sn_60min_factor_table_2020-01-01_2026-04-30.csv"]
#
#
##input_files = [
##"a_60min_factor_table_2020-01-01_2026-04-30.csv",
##"ag_60min_factor_table_2020-01-01_2026-04-30.csv",
##"al_60min_factor_table_2020-01-01_2026-04-30.csv",
##"ao_60min_factor_table_2020-01-01_2026-04-30.csv",
##"AP_60min_factor_table_2020-01-01_2026-04-30.csv",
##"au_60min_factor_table_2020-01-01_2026-04-30.csv",
##]


#population_size = 1
#population_size = 10
#population_size = 20
population_size = 110
topk_k = 5

#lr = 0.01
#lr = 0.001
#lr = 0.0001
#lr = 0.00001
#lr = 0.0000001
lr = 0.0000000001

#mutation_rate = 0.01
mutation_rate = 0.1
#mutation_rate = 0.5
#mutation_rate = 4.0

#model_file_name = "best_model.pth"
model_file_name = "best_model_short_pred.pth"
#model_file_name = "best_model_pred_14.pth"
#model_file_name = "best_model_pred_14_no_commision.pth"
#model_file_name = "best_model_pred_14_local.pth"
#model_file_name = "best_model_pred_14_unittest.pth"

#extra_new_model_cnt = 5
extra_new_model_cnt = 50



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

#net_input_dim = window_size*7
net_input_dim = window_size*6
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

#criterion = nn.CrossEntropyLoss(ignore_index=2)
criterion = nn.CrossEntropyLoss()

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

population = [copy.deepcopy(net)]
for _ in range(int(topk_k) - 1):
  population.append(mutate_model(copy.deepcopy(net)))

cached_scores = []
cached_scores_set = False




# 定义要并行处理的函数
def compute_something(temp_model):

    #period = 1200 # 20 min
    period = 300 # 5 min
    
    # using files
    total_balance = 0
    data_cnt = 0
    for input_file in input_files:
      data_dict = load_data_by_date(period, input_file[0], input_file[1], input_file[2], input_file[3])
      if len(data_dict) == 0:
        print("skipping input")
        print(input_file)
        continue
      data_np = data_dict_to_data_np(data_dict)
      temp_balance, status = trade_env_sim(temp_model, data_np, log_info=False)
      total_balance += temp_balance
      data_cnt += 1
    total_balance = total_balance/data_cnt

    ## using TQ
    #total_balance, status = TQ_env_sim(temp_model, log_info=False)

    #print("total_balance = %f"%(total_balance))


    return total_balance, temp_model




with torch.no_grad(): # no need to use grad, as we use GA

    for epoch in range(0,1000000):
        overall_balance = 0
    
    
        # Generate next generation
        mutated_population = []

        if not cached_scores_set:
            for model in population:
              mutated_population.append(copy.deepcopy(model))

        while len(mutated_population) < population_size:
            for model in population:
                mutated_population.append(mutate_model(model, mutation_rate))
                if(len(mutated_population) >= population_size):
                  break

        # some extra new models, complete random
        for idx in range(0, extra_new_model_cnt):
            mutated_population.append(MyNet(net_input_dim))
    
        if use_parallel:
            # 使用 multiprocessing.Pool 来并行处理
            with multiprocessing.Pool(processes=num_processes) as pool:
                # 使用 tqdm 来监控进度
                results = []
                for result in tqdm(pool.imap_unordered(compute_something, mutated_population), total=len(mutated_population)):
                    results.append(result)
            
        else:
          results = []
          for temp_model in tqdm(mutated_population):
              total_balance, temp_model = compute_something(temp_model)
              results.append((total_balance, temp_model))

        # added parents from last generation
        if cached_scores_set:
            assert len(population) == len(cached_scores)
            for idx in range(0, len(cached_scores)):
                results.append((cached_scores[idx], population[idx]))


        # collect individual_balances
        individual_balances = []
        individual_models = []
        for result in results:
          #print("balance")
          #print(result[0])
          individual_balances.append(result[0])
          individual_models.append(result[1])

    
        balance_top_values, balance_top_ids = torch.topk(torch.tensor(individual_balances), topk_k)
        best_balances = balance_top_values[0]
        best_net = individual_models[balance_top_ids[0]]
        print("balance_top_values")
        print(balance_top_values)
    
        population = []
        cached_scores = []
        for idx in balance_top_ids:
            population.append(copy.deepcopy(individual_models[idx]))
            cached_scores.append(individual_balances[idx])
    
    
        
        cached_scores_set = True
    
        # 打开文件用于写入，如果文件不存在则创建
        print("open for writing")
        with open('balances.txt', 'a') as file:
            file.write(str(best_balances.item())+'\n')
    
    
        #if epoch > 0 and epoch % 100 == 0:
        if epoch >= 0 :
            # save model
            torch.save({
                        'model_state_dict': best_net.state_dict(),
                }, model_file_name)
            print("Model saved to [%s]"%(model_file_name))

