


# for open too many files error:
# ulimit -n 4096  # 将限制设置为4096


import torch
import numpy as np
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#import random
#import math
#import csv
#import copy
#
#import multiprocessing
#from tqdm import tqdm
#
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import getData
from utils import NormalizeData

#commision_rate = 0.000
#commision_rate = 0.001
commision_rate = 0.005

#window_size = 1
#window_size = 2
#window_size = 7
#window_size = 15
window_size = 60
#window_size = 120
#window_size = 180
#window_size = 500
    
def get_asset_value(avg_price, holdings):
  current_asset_value = 0
  for holding in holdings:
    unit = holding[0]
    price = holding[1]
    current_asset_value += avg_price*unit

  return current_asset_value

def trade_env_sim(temp_model, data_np, log_info=False):


    window_step = 1
    #window_step = 5
    #window_step = 50
    #window_step = 150
    #window_step = 200
    #window_step = 300
    #window_step = 500


    #leverage = 1
    leverage = 10

    maintainance_margin_ratio = 0.5
    
    status = {}
    status['prices'] = []
    status['buys'] = []
    status['sells'] = []
    status['balances'] = []


    total_balance = 0
    
    if True:
        #holding = torch.tensor([0], dtype=float, requires_grad=True)
        #balance = torch.tensor([100000], dtype=float, requires_grad=True)
        balance = 100000
        balance_leveraged = leverage*balance
        balance_borrowed = (leverage-1)*balance
        holdings = []
        hold_time = 0
        for idx in range(window_size+1, data_np.shape[0], window_step):

            margin_force_sell = False

            i = idx
            #i = random.randint(window_size, data_np.shape[0]-predict_future_size-1)
            #print(idx)

            input_np = getData(data_np, i, window_size)
            avg_price = np.mean(input_np[-1,0:4])
            if log_info:
                status['prices'].append(avg_price)


            if(len(holdings) > 0):
              hold_time += 1

            need_inference = True
            if(len(holdings) > 0):

                # margin check
                current_asset_value = get_asset_value(avg_price, holdings)
                current_equity = current_asset_value - balance_borrowed
                maintainance_margin = current_asset_value/leverage
                if(current_equity < maintainance_margin):
                #if(current_equity < maintainance_margin*maintainance_margin_ratio):
                  margin_force_sell = True
                  #print("===============")
                  #print("Margin call, force sell!")
                  #print("holdings")
                  #print(holdings)
                  #print("avg_price")
                  #print(avg_price)
                  #print("current_asset_value")
                  #print(current_asset_value)
                  #print("balance_borrowed")
                  #print(balance_borrowed)
                  #print("current_equity")
                  #print(current_equity)
                  #print("maintainance_margin")
                  #print(maintainance_margin)
                  ##exit()
                  action = 0
                  need_inference = False


            # perform inferance
            if need_inference:
              # normalize data
              input_np = NormalizeData(input_np)

              #input_ts = torch.from_numpy(data_normalized).to(torch.float32)
              input_ts = torch.from_numpy(input_np).to(torch.float32)

              rows = input_ts.shape[0]
              cols = input_ts.shape[1]
              #for i in range(0, cols):
              #    input_ts[:,i]  = (input_ts[:,i] - input_ts[:,i].mean(0)) / input_ts[:,i].std(0)
              input_ts = input_ts.reshape(rows*cols)

              #print(input_ts)
              #exit()

              out = temp_model(input_ts)
              action = torch.argmax(out, dim=0)

            if(action == 0): # sell
              if len(holdings)>0:
                if log_info:
                    status['sells'].append(len(status['prices'])-1)
                balance = 0
                for holding in holdings:
                  unit = holding[0]
                  price = holding[1]
                  balance += avg_price*unit
                balance -= balance_borrowed

                # commision
                balance = balance * (1-commision_rate)

                balance_leveraged = balance*leverage
                balance_borrowed = (leverage-1)*balance
                holdings = []
                hold_time = 0
                #print("Sell at [%f], balance = [%f]"%(avg_price, balance))
              #else:
              #  do_noting
            elif(action == 1): # wait
                #do nothing
                a = 1
            elif(action == 2): # buy
                if(len(holdings) == 0):
                    #print("Buy!")
                    if log_info:
                        status['buys'].append(len(status['prices'])-1)
                    unit = balance_leveraged / avg_price
                    holdings.append((unit, avg_price))
                    balance_leveraged = 0
                    balance = 0
                    #print("Buy at [%f]"%(avg_price))
            else:
              print("Error: Unknown actioan: [%d]"%(action))
              exit()

            #if margin_force_sell:
            #  print("holdings")
            #  print(holdings)
            #  print("balance")
            #  print(balance)
            #  current_asset_value = get_asset_value(avg_price, holdings)
            #  print("current_asset_value")
            #  print(current_asset_value)
            #  print("balance_leveraged")
            #  print(balance_leveraged)
            #  print("balance_borrowed")
            #  print(balance_borrowed)
            #  exit()

            if log_info:
                if len(holdings) > 0:
                    current_asset_value = get_asset_value(avg_price, holdings)
                    status['balances'].append(current_asset_value - balance_borrowed)
                    if(current_asset_value < 1 ):
                      print("holdings")
                      print(holdings)
                      print("current_asset_value")
                      print(current_asset_value)
                      exit()
                else:
                    #if(balance < 1):
                    #  print("holdings")
                    #  print(holdings)
                    #  print("balance")
                    #  print(balance)
                    #  exit()
                    status['balances'].append(balance)


        
        # force sell at end
        if len(holdings) > 0:
            balance = 0
            if len(holdings)>0:
              for holding in holdings:
                unit = holding[0]
                price = holding[1]
                balance += avg_price*unit

              balance -= balance_borrowed

              # commision
              balance = balance * (1-commision_rate)

              balance_leveraged = balance * leverage
              balance_borrowed = (leverage-1)*balance
              holdings = []
            #print("Final sell at [%f], balance = [%f]"%(avg_price, balance))
        total_balance += balance

    #return total_balance, temp_model
    return total_balance, status

