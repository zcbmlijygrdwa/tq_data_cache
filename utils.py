





import csv
import math
from tqdm import tqdm
import numpy as np


def getData(test_data_np, i, window_size):
    #input_np = np.copy(test_data_np[i-window_size:i,:])
    input_np = np.copy(test_data_np[i-window_size-1:i-1,:]) # skip current data
    return input_np

def NormalizeData(input_np):

    # control data range
    price_max = np.max(input_np[:,0:4])
    price_min = np.min(input_np[:,0:4])
    price_range = price_max-price_min
    input_np[:,0:4] = input_np[:,0:4] - (price_min + price_range/2.0)
    input_np[:,0:4] = input_np[:,0:4] / (price_range/2.0)
    #print("max")
    #print(np.max(input_np[:,0:4]))
    #print("min")
    #print(np.min(input_np[:,0:4]))

    volume_max = np.max(input_np[:,4:5])
    volume_min = np.min(input_np[:,4:5])
    volume_range = volume_max-volume_min
    input_np[:,4:5] = input_np[:,4:5] - (volume_min + volume_range/2.0)
    input_np[:,4:5] = input_np[:,4:5] / (volume_range/2.0)
    #print("max")
    #print(np.max(input_np[:,4:5]))
    #print("min")
    #print(np.min(input_np[:,4:5]))

    open_interest_max = np.max(input_np[:,5:6])
    open_interest_min = np.min(input_np[:,5:6])
    open_interest_range = open_interest_max-open_interest_min
    input_np[:,5:6] = input_np[:,5:6] - (open_interest_min + open_interest_range/2.0)
    input_np[:,5:6] = input_np[:,5:6] / (open_interest_range/2.0)
    #print("max")
    #print(np.max(input_np[:,5:6]))
    #print("min")
    #print(np.min(input_np[:,5:6]))
    #exit()

    return input_np


def read_data(input_files, data_folder):
    data_map = {}
    
    for input_file in tqdm(input_files):
   
        data_list_open=[]
        data_list_high=[]
        data_list_low=[]
        data_list_close=[]
        data_list_volume=[]
        data_list_open_interest=[]
    
        if("sine_wave" in input_file):
          print("Found sine_wave!")
          for i in range(0,10000):
            price = 10 + 5*math.sin(i/100.0)
            data_list_open.append(price)
            data_list_high.append(price)
            data_list_low.append(price)
            data_list_close.append(price)
            data_list_volume.append(i%100)
            data_list_open_interest.append(i%1000)
        else:
            with open(data_folder+"/"+input_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
    
                # Skip the first line
                next(reader)
    
                for row in reader:
                  data_list_open.append(float(row[5]))
                  data_list_high.append(float(row[6]))
                  data_list_low.append(float(row[7]))
                  data_list_close.append(float(row[8]))
                  data_list_volume.append(float(row[10]))
                  data_list_open_interest.append(float(row[11]))
    
                  #if len(data_list_open) > 200:
                  #  break #debug only
    
    
        # Combine all features
        data_np = np.array([
            data_list_open,
            data_list_high,
            data_list_low,
            data_list_close,
            data_list_volume,
            data_list_open_interest
        ]).T  # shape (N, 6)
    
        print("Loadded [%d] data_np"%(len(data_list_open)))
        data_map[input_file] = data_np
    
    return data_map

def data_dict_to_data_np(data_dict):
    data_list_open=           data_dict["data_list_open"]
    data_list_high=           data_dict["data_list_high"]
    data_list_low=            data_dict["data_list_low"]
    data_list_close=          data_dict["data_list_close"]
    data_list_volume=         data_dict["data_list_volume"]
    data_list_open_interest=  data_dict["data_list_open_interest"]
    
    # Combine all features
    data_np = np.array([
        data_list_open,
        data_list_high,
        data_list_low,
        data_list_close,
        data_list_volume,
        data_list_open_interest
    ]).T  # shape (N, 6)
    
    return data_np
