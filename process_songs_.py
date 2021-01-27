import pandas as pd
import Levenshtein as lv
import matplotlib.pyplot as plt
import numpy as np
def process_songs():
    pth_csv = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/msd.csv"
    pth_map = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/matches_msd_msd.csv"

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
            if(not r_slice.empty and  not l_slice.empty):
                print("l:", l_slice)
                print("r:", r_slice)
                map__ = map__.append(row)

        else:
            continue



    map__.to_csv("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/map_filtered1.csv",index=False)
    pass
def clean_():
    pth_map = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/map_filtered.csv"
    map_ = pd.read_csv(pth_map, index_col=False)
    map__ = pd.read_csv(pth_map, index_col=False)
    count= 0
    for index, row in map_.iterrows():
        l_slice = map_.loc[map_['id2'] == (row['id1'])]
        if(l_slice.empty):
            continue
        a = map_[(map_.id1==row['id2'])&(map_.id2==row['id1'])].index.tolist()
        if (len(a) == 0):
            continue
        map_ = map_.drop(index=a)
       # a = l_slice['id1'].to_numpy()[0]
       # b = row['id2']
       # if(a == b):
       #     map_ = map_.drop(index=index)
    map_ = map_.drop_duplicates()
    #map__.to_csv("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/map_filtered3.csv",index=False)
    map_.to_csv("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/map_filtered2.csv",index=False)

    pass
def deepmatcher_generate():
    pth_map = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/map_filtered.csv"
    pth_source = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/msd.csv"
    train_pth = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/train.csv"
    test_pth = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/valid.csv"
    valid_pth ="/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/test.csv"
    order = ['id', 'label','id_x', 'title_x', 'release_x', 'artist_name_x', 'duration_x', 'artist_familiarity_x', 'artist_hotttnesss_x', 'year', 'id_y', 'title_y', 'release_y', 'artist_name_y', 'duration_y', 'artist_familiarity_y', 'artist_hotttnesss_y', 'year_y']

    map_ = pd.read_csv(pth_map, index_col=False)
    source_ = pd.read_csv(pth_source, index_col=False)

    count = 0
    l_ = pd.DataFrame()
    r_ = pd.DataFrame()

    for index, row in map_.iterrows():
       # print(row['id1'])
       # print(row['id2'])
        l_slice = source_.loc[source_['id'] == int(row['id1'])]
        r_slice = source_.loc[source_['id'] == int(row['id2'])]
   #     print(l_slice)
   #     print(r_slice)
        count += 1
        l_ = l_.append(l_slice)
        r_ = r_.append(r_slice)
        if(count >1000):
            break
        

    cartesian_ = pd.merge(l_,r_, how='outer',on='year')
    cartesian_['year_y'] = cartesian_['year']
    cartesian_['label'] = 0
    cartesian__ = pd.DataFrame()
    #cartesian_.set_index(['id_x'])
    for index, row in cartesian_.iterrows():
        r_slice = map_.loc[map_['id1'] == int(row['id_x'])]
        l_slice = r_slice.loc[r_slice['id2'] == int(row['id_y'])]
        if( l_slice.empty or r_slice.empty):
            cartesian__ = cartesian__.append(row)
            print(row)
            continue
        else:
            row['label'] = 1
            cartesian__ = cartesian__.append(row)
    pd.set_option('display.max_columns', None)
    cartesian__['id'] = cartesian__.index
    cartesian__ = cartesian__[order]
    del cartesian__['id_x']
    del cartesian__['id_y']
    cartesian__.to_csv("/home/LAB/zhuxk/project/data/ER-dataset-benchmark/EC/songs/msd_attr.csv",index=False)
    pass

if __name__ == '__main__':
    #process_songs()
    #deepmatcher_generate()

    clean_()
