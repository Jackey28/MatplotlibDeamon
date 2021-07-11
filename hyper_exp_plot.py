import numpy as np
# from basic_units import cm, inch
import matplotlib.pyplot as plt

import numpy as np
from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter
import math
import matplotlib.cbook as cbook





font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,

         }
patterns = ('/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*')
# patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')


# plt.style.use("seaborn-deep")
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-muted')
# plt.style.use('classic')
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-pastel')
# plt.style.use('bmh')
# plt.style.use('ggplot')
# plt.style.use('seaborn-paper')


fig, axs = plt.subplots(3, 4)

colors = ['navy', 'cornflowerblue', 'deepskyblue']

plt.legend(loc='lower center')  # 标签位置

# hyper = [(26.326664 + 24.401422)/2, (48.479328 + 48.601736)/2, (131.609672 + 139.679083)/2,  (138.982280 + 148.364814+147.848467 +133 + 134.616868)/5, 176.615500]
hyper = [1.923683, 1.922421, 1.848152, 1.846108, 1.851214, 1.960189]
# hyper_inc = [16.332565, 23.053892, 41.854730, 60.919483, 73.585759]
hyper_inc = [17.2677, 26.4467, 44.5676, 63.3853, 68.725]
# hyper_nml = [ 23.95,42.094,89.075, 143.322,164.301]
hyper_ext = [1.939147, 1.607237, 1.666666, 1.471315, 1.447754, 1.519771]
hyper_tqm = [5, 5, 5, 5, 5]
f1 = [0.969533, 0.969505, 0.969505, 0.950541, 0.973622, 0.949332]
rec = [0.965393, 0.964494, 0.964494, 0.928539, 0.928989, 0.926292]
baseline_ = [0, 0, 0, 0, 0, 0]

x = [8, 16, 32, 64, 96, 128]
ax2 = axs[0, 0].twinx()
ax2.set_ylabel('recall', fontdict=font1)
# ax2.plot(x, cmp,  color='blue',    markersize=5, alpha=0.1)
ax2.fill_between(x, rec, baseline_, facecolor='black', label="recall", alpha=0.1)
ax2.set_ylim([0.8, 1.0])
# ax2.set_xlim([0, 127])
axs[0, 0].plot(x, hyper, marker='x', color='orange', linestyle="--", label='$Hyper_1$', markersize=9)
# axs[0, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":",label='$Hyper_{inc}$', markersize=7)
# axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[0, 0].plot(x, hyper_ext, marker='<', color='lightseagreen',  label='$Hyper$', markersize=9)
# axs[0, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":",label='$Hyper_{tqm}$', markersize=8)
axs[0, 0].set_ylabel('time (sec.)', fontdict=font1)
# axs[0, 0].set_xlabel('$|B|$', fontdict=font1)
axs[0, 0].set_title('(a) DBLP-ACM: Varying $|\mathcal{B}|$', fontdict=font1)
axs[0, 0].set_xlim([8, 128])
# axs[0, 0].set_ylim([0, 1150])
axs[0, 0].tick_params(labelsize=10)
for ytick in axs[0, 0].get_yticklabels():
    ytick.set_rotation(30)
for xtick in axs[0, 0].get_xticklabels():
    xtick.set_rotation(30)
for ytick in ax2.get_yticklabels():
    ytick.set_rotation(30)
for ytick in ax2.get_xticklabels():
    ytick.set_rotation(30)

# hyper = [(26.326664 + 24.401422)/2, (48.479328 + 48.601736)/2, (131.609672 + 139.679083)/2,  (138.982280 + 148.364814+147.848467 +133 + 134.616868)/5, 176.615500]
hyper = [1.739, 1.818, 1.916, 2.079, 2.37, 2.37]
# hyper_inc = [16.332565, 23.053892, 41.854730, 60.919483, 73.585759]
hyper_ext = [1.66987, 2.225, 1.816, 2.7418, 2.652, 2.64744]
# hyper_nml = [ 23.95,42.094,89.075, 143.322,164.301]
hyper_tqm = [5, 5, 5, 5, 5]
f1 = [0.969533, 0.969505, 0.969505, 0.950541, 0.973622, 0.949332]
rec = [0.411642, 0.410603, 0.409563, 0.407484, 0.392931, 0.072765]
baseline_ = [0, 0, 0, 0, 0, 0]

x = [8, 16, 32, 64, 96, 128]
ax2 = axs[0, 1].twinx()
ax2.set_ylabel('recall', fontdict=font1)
# ax2.plot(x, cmp,  color='blue',    markersize=5, alpha=0.1)
ax2.fill_between(x, rec, baseline_, facecolor='black', alpha=0.1)
ax2.set_ylim([0.0, 0.5])
# ax2.set_xlim([0, 127])
axs[0, 1].plot(x, hyper, marker='x', color='orange', linestyle="--", markersize=7)
# axs[0, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":",label='$Hyper_{inc}$', markersize=7)
# axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[0, 1].plot(x, hyper_ext, marker='<', color='#42b395', linestyle=":", markersize=7)
# axs[0, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":",label='$Hyper_{tqm}$', markersize=8)
axs[0, 1].set_ylabel('time (sec.)', fontdict=font1)
# axs[0, 1].set_xlabel('$|B|$', fontdict=font1)
axs[0, 1].set_title('(b) Walmart: Varying $|\mathcal{B}|}$', fontdict=font1)
axs[0, 1].set_xlim([8, 128])
# axs[0, 0].set_ylim([0, 1150])
axs[0, 1].tick_params(labelsize=10)
for ytick in axs[0, 1].get_yticklabels():
    ytick.set_rotation(30)
for xtick in axs[0, 1].get_xticklabels():
    xtick.set_rotation(30)
for ytick in ax2.get_yticklabels():
    ytick.set_rotation(30)
for ytick in ax2.get_xticklabels():
    ytick.set_rotation(30)

# hyper = [(26.326664 + 24.401422)/2, (48.479328 + 48.601736)/2, (131.609672 + 139.679083)/2,  (138.982280 + 148.364814+147.848467 +133 + 134.616868)/5, 176.615500]
hyper = [12.201353, 8.383923, 6.790110, 6.010503, 6.079603, 5.740590, 5.895164, 6.008211, 6.2759, 6.3959]
# hyper_inc = [16.332565, 23.053892, 41.854730, 60.919483, 73.585759]
hyper_ext = [11.159868, 10.108526, 8.160054, 8.920646, 8.251598, 8.593966, 8.329888, 7.853542, 9.1136, 8.6821]
# hyper_nml = [ 23.95,42.094,89.075, 143.322,164.301]
f1 = [0.969533, 0.969505, 0.969505, 0.950541, 0.973622, 0.949332, 0, 0]
rec = [0.865, 0.865, 0.865, 0.865, 0.865, 0.865, 0.865, 0.865, 0.846, 0.846]
baseline_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

x = [8, 16, 32, 64, 128, 256, 384, 512, 768, 1024]
ax2 = axs[0, 2].twinx()
ax2.set_ylabel('recall', fontdict=font1)
# ax2.plot(x, cmp,  color='blue',    markersize=5, alpha=0.1)
ax2.fill_between(x, rec, baseline_, facecolor='black', alpha=0.1)
ax2.set_ylim([0.8, 0.9])
# ax2.set_xlim([0, 127])
axs[0, 2].plot(x, hyper, marker='x', color='orange', linestyle="--", markersize=7)
# axs[0, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":",label='$Hyper_{inc}$', markersize=7)
# axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[0, 2].plot(x, hyper_ext, marker='<', color='#42b395', linestyle=":", markersize=7)
# axs[0, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":",label='$Hyper_{tqm}$', markersize=8)
axs[0, 2].set_ylabel('time (sec.)', fontdict=font1)
# axs[0, 1].set_xlabel('$|B|$', fontdict=font1)
axs[0, 2].set_title('(c) Songs: Varying $|\mathcal{B}|$', fontdict=font1)
axs[0, 2].set_xlim([8, 1024])
# axs[0, 0].set_ylim([0, 1150])
axs[0, 2].tick_params(labelsize=10)
for ytick in axs[0, 2].get_yticklabels():
    ytick.set_rotation(30)
for xtick in axs[0, 2].get_xticklabels():
    xtick.set_rotation(30)
for ytick in ax2.get_yticklabels():
    ytick.set_rotation(30)
for ytick in ax2.get_xticklabels():
    ytick.set_rotation(30)

# hyper = [(26.326664 + 24.401422)/2, (48.479328 + 48.601736)/2, (131.609672 + 139.679083)/2,  (138.982280 + 148.364814+147.848467 +133 + 134.616868)/5, 176.615500]
hyper = [31.692157,  19.508912, 19.468878,  15.128439, 12.441956, 12.017819, 12.032889, 11.478547,  11.914635]
# hyper_inc = [16.332565, 23.053892, 41.854730, 60.919483, 73.585759]
hyper_ext = [31.709884, 19.891377, 20.487001, 14.213062,13.838866,  14.347395, 14.183675,  13.913377, 13.934814]
# hyper_nml = [ 23.95,42.094,89.075, 143.322,164.301]
f1 = [0.969533, 0.969505, 0.969505, 0.950541, 0.973622, 0.949332, 0, 0]
rec = [ 0.902494, 0.811600, 0.810092,  0.791061, 0.791002, 0.790963, 0.790969, 0.790959,0.790956]
baseline_ = [0, 0, 0, 0, 0, 0, 0, 0, 0 ]

x = [ 16, 32, 64, 128, 256, 384, 512, 768, 1024]
ax2 = axs[0, 3].twinx()
ax2.set_ylabel('recall', fontdict=font1)
# ax2.plot(x, cmp,  color='blue',    markersize=5, alpha=0.1)
ax2.fill_between(x, rec, baseline_, facecolor='black', alpha=0.1)
ax2.set_ylim([0.75, 0.95])
# ax2.set_xlim([0, 127])
axs[0, 3].plot(x, hyper, marker='x', color='orange', linestyle="--", markersize=7)
# axs[0, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":",label='$Hyper_{inc}$', markersize=7)
# axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[0, 3].plot(x, hyper_ext, marker='<', color='#42b395', linestyle=":", markersize=7)
# axs[0, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":",label='$Hyper_{tqm}$', markersize=8)
axs[0, 3].set_ylabel('time (sec.)', fontdict=font1)
# axs[0, 1].set_xlabel('$|B|$', fontdict=font1)
axs[0, 3].set_title('(d) NCV: Varying $|\mathcal{B}|$', fontdict=font1)
axs[0, 3].set_xlim([16, 1024])
# axs[0, 0].set_ylim([0, 1150])
axs[0, 3].tick_params(labelsize=10)
for ytick in axs[0, 3].get_yticklabels():
    ytick.set_rotation(30)
for xtick in axs[0, 3].get_xticklabels():
    xtick.set_rotation(30)
for ytick in ax2.get_yticklabels():
    ytick.set_rotation(30)
for ytick in ax2.get_xticklabels():
    ytick.set_rotation(30)

spark_er = [169.2491631507873, 334.80596590042114, 653.613538026809, 986.5015697479248, 1113.9779381752014]
dedoop = [0, 0, 0, 0, 0]
disdedup = [1392, 2784, ]
x_ = [0.6, 1.2]
# hyper = [(26.326664 + 24.401422)/2, (48.479328 + 48.601736)/2, (131.609672 + 139.679083)/2,  (138.982280 + 148.364814+147.848467 +133 + 134.616868)/5, 176.615500]
hyper = [24.68, 42.2843, 91.5795, 144.55, 164.08]
# hyper_inc = [16.332565, 23.053892, 41.854730, 60.919483, 73.585759]
hyper_inc = [17.2677, 26.4467, 44.5676, 63.3853, 68.725]
# hyper_nml = [ 23.95,42.094,89.075, 143.322,164.301]
hyper_ext = [24.379791, 45.910275, 85.566736, 123.502402, 136.503737]
hyper_ext = np.log2(hyper_ext)
hyper = np.log2(hyper)
hyper_inc = np.log2(hyper_inc)
spark_er = np.log2(spark_er)
disdedup = np.log2(disdedup)
hyper_tqm = [5, 5, 5, 5, 5]
x = [0.6, 1.2, 2.4, 3.6, 4]

axs[1, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle=':', markersize=5, label='Sparker')
# axs[0, 0].plot(x, dedoop, marker='*', color='darkslateblue', linestyle='--', label='Dedoop', markersize=4)
axs[1, 0].plot(x_, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), label='Disdedup')
axs[1, 0].plot(x, hyper, marker='x', color='orange', linestyle="--", label='$Hyper$', markersize=7)
axs[1, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":", markersize=7)
# axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[1, 0].plot(x, hyper_ext, marker='<', color='#42b395', linestyle=":", markersize=7)
# axs[0, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":",label='$Hyper_{tqm}$', markersize=8)

axs[1, 0].set_ylabel('time ($log$ sec.)', fontdict=font1)
axs[1, 0].set_xlabel('$mio$', fontdict=font1)
axs[1, 0].set_title('(a) TPCH: Varying $|D|$', fontdict=font1)
axs[1, 0].set_xlim([0.6, 4])
# axs[0, 0].set_ylim([0, 1150])
axs[1, 0].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(30)

# hyper = [ 212.545285, 137.476279, 117.308165, 104.527871, 97.416963, 89.340965, 88.380350]
hyper = [271.084665, 187.453172, 149.914119, 151.032481, 134.543713, 133.198432, 133.041249]
hyper_ext = [217.713251, 160.747026, 137.658129, 131.760152, 120.550031, 121.949107, 122.881350]
hyper_inc = [63.600067, 65.604519, 68.250209, 70.410183, 70.957951, 71.437485, 74.126610]
# hyper_nml = [ 192.156395, 128.776147, 109.266810, 99.405487, 92.458567, 90.039718, 94.679042 ]
# hyper_tqm = [5,5,5,5,5]
x = [1, 2, 3, 4, 5, 6, 7]
axs[2, 0].plot(x, hyper, marker='x', color='orange', linestyle="--", markersize=7)
axs[2, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":", markersize=7)
axs[2, 0].plot(x, hyper_ext, marker='<', color='#42b395', linestyle=":", markersize=7)
axs[2, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[2, 0].set_xlabel('$n$', fontdict=font1)
axs[2, 0].set_title('(a) TPCH: Varying $n$', fontdict=font1)
axs[2, 0].set_xlim([1, 7])
axs[2, 0].set_ylim([60, 80])
axs[2, 0].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(30)


y1 = [2300+436+139.654+1.761+1+185.92, 3900+1200, 5100+742.682+214.14, 6700+986.747+243.296, 8300+1200+292.0457]
y2 = [2300, 3900, 5100, 6700, 8300]
y3 = [436.761, 1200, 742.682, 986.747, 1200]
y4 = [139.654, 185.92, 214.14,  243.296227, 292.04257]
y5 = [647.56, 1468.53, 1070.146582, 1380.035864,  1604.833644]


x_ = np.array([0.55, 0.66, 0.77, 0.88, 1])
width = 0.02
axs[1, 2].bar(x_ + width, y1, width=width,
       color='skyblue',
    hatch="xxx",
              label='sum time'
              )
axs[1, 2].bar(x_ + 2*width, y2, width=width,
       color='royalblue',
    hatch="...",
              label='GPU activities time'
              )
axs[1, 2].bar(x_ + 3*width, y3, width=width,
       color='violet',
    hatch="///",
              label='scheduling time'
              )
axs[1, 2].bar(x_ + 4*width, y4, width=width,
              color='red',
              hatch="+++",
              label='time to prepare data'
              )
axs[1, 2].bar(x_ + 5*width, y5, width=width,
              color='#ffb07c',
              hatch='***',
                label='actual time'
              )

axs[1, 2].set_ylabel('time (sec.)', fontdict=font1)
ticks = axs[1, 2].set_xticks([0.55+3*width, 0.66+3*width,0.77+3*width, 0.88+3*width, 1+3*width])
axs[1, 2].set_xticklabels(['0.55','0.66','0.77','0.88','1.0'],rotation = 30)
for ytick in axs[1, 2].get_yticklabels():
    ytick.set_rotation(30)


y1 = [2300+436+139.654+1.761+1+185.92, 3900+1200, 5100+742.682+214.14, 6700+986.747+243.296, 8300+1200+292.0457]
y2 = [2300, 3900, 5100, 6700, 8300]
y3 = [436.761, 1200, 742.682, 986.747, 1200]
y4 = [139.654, 185.92, 214.14,  243.296227, 292.04257]
y5 = [647.56, 1468.53, 1070.146582, 1380.035864,  1604.833644]


x_ = np.array([0.55, 0.66, 0.77, 0.88, 1])
width = 0.02
axs[1, 3].bar(x_ + width, y1, width=width,
       color='skyblue',
    hatch="xxx",
              )
axs[1, 3].bar(x_ + 2*width, y2, width=width,
       color='royalblue',
    hatch="...",
              )
axs[1, 3].bar(x_ + 3*width, y3, width=width,
       color='violet',
    hatch="///",
              )
axs[1, 3].bar(x_ + 4*width, y4, width=width,
              color='red',
              hatch="+++",
              )
axs[1, 3].bar(x_ + 5*width, y5, width=width,
              color='#ffb07c',
              hatch='***',
              )

axs[1, 3].set_ylabel('time (sec.)', fontdict=font1)
ticks = axs[1, 3].set_xticks([0.55+3*width, 0.66+3*width,0.77+3*width, 0.88+3*width, 1+3*width])
axs[1, 3].set_xticklabels(['0.55','0.66','0.77','0.88','1.0'],rotation = 30)
for ytick in axs[1, 3].get_yticklabels():
    ytick.set_rotation(30)




fig.legend(loc='upper center', ncol=8, fontsize=13)

plt.show()
