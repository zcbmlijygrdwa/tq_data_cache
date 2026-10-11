
import pickle
#from datetime import datetime, date, timedelta
import datetime
from datetime import date
import tqsdk
from tqsdk import TqApi, TqAuth, TqBacktest, BacktestFinished
import numpy as np
import pandas as pd

import datetime

import sys, os
sys.path.append('/home/zhenyu/SMT/nn')
sys.path.append('/home/zhenyu/SMT/nn/short_term_ga')
import torch
from TestNet import MyNet
from utils import NormalizeData
import logging

from tq_utils import klines_to_input_np

def timestamp_to_time_of_day(timestamps):

    timestamps = np.asarray(timestamps, dtype=np.int64)
    seconds = timestamps // 1_000_000_000
    time_of_day = ((seconds + 8 * 3600) % 86400) / 86400.0

    return time_of_day.tolist()

def gen_pkl_name(period, symbol, year, month, day):
    if(period == 1200):
      period_str = "20min"
    elif(period == 300):
      period_str = "5min"
    elif(period == 60):
      period_str = "1min"
    else:
      print("[Error]: Unknown period: [%d] sec"%(period))
      exit()
    pickle_file_name = symbol+"_"+str(year)+"_"+str(month)+"_"+str(day)+"_"+period_str+".pkl"

    return pickle_file_name

def build_contract_symbol(symbol, year, month):
    if symbol.startswith("CZCE."):
        # 郑商所：年份1位 + 月份2位
        contract = f"{year % 10}{month:02d}"
    else:
        # DCE / SHFE 等：年份2位 + 月份2位
        contract = f"{year % 100:02d}{month:02d}"

    return symbol + contract


def empty_data_dict():
    data_dict = {
        "data_list_timestamp": [],
        "data_list_time_of_day": [],
        "data_list_open": [],
        "data_list_high": [],
        "data_list_low": [],
        "data_list_close": [],
        "data_list_volume": [],
        "data_list_open_interest": [],
    }

    return data_dict


def load_data_by_date(period, symbol, year, month, day):
    
    pickle_file_dir = "/home/zhenyu/SMT/nn/tianqin/data/cache/"

    #klines_sec = 60
    #klines_sec = 60*20 # 20 min
    klines_sec = period

    api = None
    bars = {}
        
  

    # estimate active contract based on date
    if(symbol=="DCE.c"):
        if(month>=11):
          next_mon = 1
        else:
          C_MONTHS = [1, 3, 5, 7, 9, 11]
          next_mon = -1
          for m in C_MONTHS:
            if(m >= month+1):
              next_mon = m
              break
          assert next_mon > 0
    else:
        next_mon = (month + 1)%12

    #if(next_mon <10):
    #  next_mon_str = "0"+str(next_mon)
    #else:
    #  next_mon_str = str(next_mon)

    #symbol_with_mon = symbol + str(year-2000) + next_mon_str
    symbol_with_mon = build_contract_symbol(symbol, year, next_mon)

    pickle_file_name = gen_pkl_name(period, symbol_with_mon, year, month, day)
    pickle_file_path = pickle_file_dir+pickle_file_name

    data_dict = {}
    is_trading_day = False

    if os.path.exists(pickle_file_path):
        #print("Loading data from [%s] ... "%(pickle_file_path))
        # 打开文件以二进制读取模式
        with open(pickle_file_path, 'rb') as file:
            # 加载pickle数据
            data_dict = pickle.load(file)
    else:
        print("Load data from TQ server for [%s]..."%(pickle_file_name))
        auth=TqAuth("zcbmlijygrdwa", "zcbmlijygrdwa")


        target_date = datetime.date(year, month, day)

        backtest=TqBacktest(
            start_dt=target_date,
            end_dt=target_date
        )
        start_ns = backtest._start_dt
        end_ns = backtest._end_dt

        try:
            api = TqApi(
                backtest=backtest,
                auth=auth,
                disable_print=True
            )

            calendar = api.get_trading_calendar(
                start_dt=target_date,
                end_dt=target_date
            )
            
            is_trading_day = bool(calendar.iloc[0]["trading"])

            
            klines = api.get_kline_serial(
                symbol_with_mon,
                klines_sec,
                data_length=2
            )
            
            while True:
                api.wait_update()
        
                bar = klines.iloc[-1]
                ts = int(bar["datetime"])
        
                # 关键：只接受本次交易日范围内的数据
                if start_ns <= ts <= end_ns:
                    bars[ts] = {
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(bar["volume"]),
                        "open_interest": float(bar["open_oi"]),
                    }
        
        except BacktestFinished:
            if api is not None:
                api.close()

        except Exception as e:
            print("[Error]: error when loading data for:")
            print(target_date)
            print(e)

        if not is_trading_day:
            print("Skipping Non-trading day")
            print(target_date)
            data_dict = empty_data_dict()
        elif len(bars) == 0:
            data_dict = empty_data_dict()
        else:
            timestamps = sorted(bars)
            times_of_day = timestamp_to_time_of_day(timestamps)
            
            data_dict = {
                "data_list_timestamp": timestamps,
                "data_list_time_of_day": times_of_day,
                "data_list_open": [bars[t]["open"] for t in timestamps],
                "data_list_high": [bars[t]["high"] for t in timestamps],
                "data_list_low": [bars[t]["low"] for t in timestamps],
                "data_list_close": [bars[t]["close"] for t in timestamps],
                "data_list_volume": [bars[t]["volume"] for t in timestamps],
                "data_list_open_interest": [
                    bars[t]["open_interest"] for t in timestamps
                ],
            }

        with open(pickle_file_path, 'wb') as file:
            # 使用pickle的dump方法保存变量
            pickle.dump(data_dict, file)

            print("Data cached to [%s]"%(pickle_file_path))

    return data_dict


def load_data_by_date_with_cache(cache_dict, period, symbol, year, month, day):
    
    pickle_file_name = gen_pkl_name(period, symbol, year, month, day)

    if(pickle_file_name not in cache_dict):
      data_dict = load_data_by_date(period, symbol, year, month, day)
      cache_dict[pickle_file_name] = data_dict

    return cache_dict[pickle_file_name], cache_dict

    #klines_sec = 60
    
#symbol = "DCE.m1809"
#year = 2018
#month = 5
#day = 1
#data_dict = load_data_by_date(symbol, year, month, day)
#print(len(data_dict["data_list_timestamp"]))
#print(len(data_dict["data_list_open"]))
##data_list_high = []
##data_list_low = []
##data_list_close = []
##data_list_volume = []
##data_list_open_interest = []
