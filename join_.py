import pandas as pd
import Levenshtein as lv
import matplotlib.pyplot as plt
import numpy as np

pth_l = "/home/xiaoke/private/dataset/dedoop/EC/5Party-ocp20/cleared/ncvr_numrec_1000000_modrec_2_ocp_20_myp_1_nump_5_cleared.csv"
pth_r = "/home/xiaoke/private/dataset/dedoop/EC/5Party-ocp20/cleared/ncvr_numrec_1000000_modrec_2_ocp_20_myp_0_nump_5_cleared.csv"
csv_l = pd.read_csv(pth_l, index_col=False)
csv_r = pd.read_csv(pth_r, index_col=False)
pd.set_option('display.max_columns', None)

cartisian = pd.merge(csv_l, csv_r, how='left', on='recid')
print(cartisian)

