import pandas as pd
import Levenshtein as lv
import matplotlib.pyplot as plt
import numpy as np
def process_songs():
    pth_csv = "/home/xiaoke/private/dataset/dedoop/EC/songs/msd.csv"
    pth_map = "/home/xiaoke/private/dataset/dedoop/EC/songs/matches_msd_msd.csv"

    csv_ = pd.read_csv(pth_csv, index_col=False)
    map_ = pd.read_csv(pth_map, index_col=False)

    pd.set_option('display.max_columns', None)

    count = 0
    map__ = pd.DataFrame(columns=['id1', 'id2'])

    for index, row in map_.iterrows():
        count+=1
        if(row['id1'] != row['id2']):
            l_slice = csv_.loc[csv_['id'] == (row['id1'])]
            r_slice = csv_.loc[csv_['id'] == (row['id2'])]
            if l_slice.empty:
                print(row)
                print(l_slice)
                print(r_slice)
                map__ = map__.append(row)
            else:
                print("aaa")
        else:
            continue
    map__.to_csv("/home/xiaoke/private/dataset/dedoop/EC/songs/map_filtered.csv",index=False)
    pass
def clean_():
    pth_map = "/home/xiaoke/private/dataset/dedoop/EC/songs/map_filtered.csv"
    map_ = pd.read_csv(pth_map, index_col=False)
    map__ = pd.read_csv(pth_map, index_col=False)
    print(map_)
    count= 0
    for index, row in map_.iterrows():
        l_slice = map_.loc[map_['id2'] == (row['id1'])]
        if(l_slice.empty):
            continue
        a = l_slice['id1'].to_numpy()[0]
        b = row['id2']
        if(a == b):
            map_ = map_.drop(index=index)
    print(map_)
    map__.to_csv("/home/xiaoke/private/dataset/dedoop/EC/songs/map_filtered2.csv",index=False)

    pass
if __name__ == '__main__':
    clean_()
