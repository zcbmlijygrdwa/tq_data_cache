
import pickle
#from datetime import datetime, date, timedelta
import datetime
from tqsdk import TqApi, TqAuth, TqBacktest, BacktestFinished
import numpy as np

import datetime
import pytz

import sys, os
sys.path.append('/home/zhenyu/SMT/nn')
sys.path.append('/home/zhenyu/SMT/nn/short_term_ga')
import torch
from TestNet import MyNet
from utils import NormalizeData
import logging

from tq_utils import klines_to_input_np


def load_data_by_date(period, symbo, year, month, day):
    
    pickle_file_dir = "/home/zhenyu/SMT/nn/tianqin/data/cache/"

    # load model
    model_file_name = "/home/zhenyu/SMT/nn/short_term_ga/best_model_pred_14.pth"
    #model_file_name = "/home/zhenyu/SMT/nn/short_term_ga/best_model_pred_14_local.pth"
    
    #klines_sec = 60
    #klines_sec = 60*20 # 20 min
    klines_sec = period


    if(klines_sec == 1200):
      period_str = "20min"
    elif(klines_sec == 300):
      period_str = "5min"
    else:
      print("[Error]: Unknown period: [%d] sec"%(klines_sec))
      exit()
    pickle_file_name = symbo+"_"+str(year)+"_"+str(month)+"_"+str(day)+"_"+period_str+".pkl"
    pickle_file_path = pickle_file_dir+pickle_file_name

    if os.path.exists(pickle_file_path):
        #print("Path exists")
        # 打开文件以二进制读取模式
        with open(pickle_file_path, 'rb') as file:
            # 加载pickle数据
            data_dict = pickle.load(file)
    else:
        print("Load data from TQ server for [%s]..."%(pickle_file_name))
        auth=TqAuth("zcbmlijygrdwa", "zcbmlijygrdwa")


        target_date = datetime.datetime(year, month, day)
        one_day_after = target_date + datetime.timedelta(days=1)
        #print("target_date")
        #print(target_date)
        #print("one_day_after")
        #print(one_day_after)

        data_list_timestamp = []
        data_list_open = []
        data_list_high = []
        data_list_low = []
        data_list_close = []
        data_list_volume = []
        data_list_open_interest = []
        try:
            api = TqApi(backtest=TqBacktest(start_dt=target_date, end_dt=one_day_after), auth=auth, disable_print=True)
            klines = api.get_kline_serial(symbo, klines_sec, data_length=1)
            while True:
                api.wait_update()
            
                if api.is_changing(klines):
                    account = api.get_account()
                    position = api.get_position(symbo)
                    #print("account balance = %f"%(account.balance))
                    #print("klines keys")
                    #print(klines.keys)
                    #print("klines")
                    #print(klines)

                    data_list_timestamp.append(int(klines.datetime[0]))

                    temp_data_list_open = klines.open.to_list()
                    temp_data_list_high = klines.high.to_list()
                    temp_data_list_low = klines.low.to_list()
                    temp_data_list_close = klines.close.to_list()
                    temp_data_list_volume = klines.volume.to_list()
                    temp_data_list_open_interest = klines.open_oi.to_list()

                    data_list_open.append(temp_data_list_open[0])
                    data_list_high.append(temp_data_list_high[0])
                    data_list_low.append(temp_data_list_low[0])
                    data_list_close.append(temp_data_list_close[0])
                    data_list_volume.append(temp_data_list_volume[0])
                    data_list_open_interest.append(temp_data_list_open_interest[0])
            
        
        except BacktestFinished as e:
            # 回测结束时会执行这里的代码
            api.close()
            #total_balance += acc.balance

        except Exception as e:
            print("[Error]: error when loading data for:")
            print(target_date)
            print(e)

        data_dict = {
            "data_list_timestamp": data_list_timestamp,
            "data_list_open": data_list_open,
            "data_list_high": data_list_high,
            "data_list_low": data_list_low,
            "data_list_close": data_list_close,
            "data_list_volume": data_list_volume,
            "data_list_open_interest": data_list_open_interest,
          }

        with open(pickle_file_path, 'wb') as file:
            # 使用pickle的dump方法保存变量
            pickle.dump(data_dict, file)

            print("Data cached to [%s]"%(pickle_file_path))

    #for nano_seconds in data_dict["data_list_timestamp"]:
    #    # 假设你有一个纳秒级别的Unix时间戳，例如：1633081605000000000（这是1599888000秒加上一些纳秒）
    #    #nano_seconds = 1633081605000000000
    #    
    #    # 将纳秒转换为秒（去掉最后9个数字）
    #    seconds = nano_seconds / 1e9
    #    
    #    # 将秒级别的Unix时间戳转换为datetime对象
    #    dt = datetime.datetime.utcfromtimestamp(seconds)
    #    
    #    # 设置北京时区（CST为中国标准时间，UTC+8）
    #    beijing_tz = pytz.timezone('Asia/Shanghai')
    #    
    #    # 本地化时间到北京时区
    #    beijing_dt = dt.replace(tzinfo=pytz.utc).astimezone(beijing_tz)
    #    
    #    print("北京时间:", beijing_dt.strftime('%Y-%m-%d %H:%M:%S'))

    #print(len(data_dict["data_list_timestamp"]))
    #print(len(data_dict["data_list_open"]))

    #assert len(data_dict["data_list_timestamp"]) == 42 or len(data_dict["data_list_timestamp"]) == 0

    return data_dict


    
#symbo = "DCE.m1809"
#year = 2018
#month = 5
#day = 1
#data_dict = load_data_by_date(symbo, year, month, day)
#print(len(data_dict["data_list_timestamp"]))
#print(len(data_dict["data_list_open"]))
##data_list_high = []
##data_list_low = []
##data_list_close = []
##data_list_volume = []
##data_list_open_interest = []
