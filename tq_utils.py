

import numpy as np

def klines_to_input_np(klines):
    #print("klines keys")
    #print(klines.keys())
    #print("klines.datetime")
    #print(klines.datetime)
    #datetime_list = klines.datetime.to_list()
    #for time in datetime_list:
    #  print(int(time))
    #exit()
    data_list_open = klines.open.to_list()
    data_list_high = klines.high.to_list()
    data_list_low = klines.low.to_list()
    data_list_close = klines.close.to_list()
    data_list_volume = klines.volume.to_list()
    data_list_open_interest = klines.open_oi.to_list()


    # Combine all features
    input_np = np.array([
        data_list_open,
        data_list_high,
        data_list_low,
        data_list_close,
        data_list_volume,
        data_list_open_interest
    ]).T  # shape (N, 7)

    return input_np
