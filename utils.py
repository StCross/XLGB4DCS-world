import pandas as pd
import numpy as np

def series_to_supervised(_data, n_step):
    res = list()
    data = _data.copy()
    length = len(data)
    for i in range(length - n_step + 1):
        fragment = data.iloc[i: i + n_step].values
        res.append(fragment)
    res = np.array(res)
    return res
    
