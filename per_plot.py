import numpy as np
# from basic_units import cm, inch
import matplotlib.pyplot as plt

import numpy as np
from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter
import math


def custom_log(a):
    return math.log(a, 2)


def normalnize(a):
    for i in range(len(a)):
        if a[i] > 4000:
            a[i] /= 50
        elif (a[i] < 4000 and a[i] > 3000):
            a[i] /= 40
        elif (a[i] < 3000 and a[i] > 2000):
            a[i] /= 30
        elif (a[i] < 2000 and a[i] > 1000):
            a[i] /= 20
        elif (a[i] < 1000 and a[i] > 500):
            a[i] /= 10
        elif (a[i] < 500 and a[i] > 1):
            a[i] /= 3
    return a


def alige(a):
    if a < 5:
        return 5
    elif a < 50:
        return 50
    else:
        return a


class MercatorLatitudeScale(mscale.ScaleBase):
    """
    Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
    the system used to scale latitudes in a Mercator__ projection.
    The scale function:
      ln(tan(y) + sec(y))
    The inverse scale function:
      atan(sinh(y))
    Since the Mercator scale tends to infinity at +/- 90 degrees,
    there is user-defined threshold, above and below which nothing
    will be plotted.  This defaults to +/- 85 degrees.
    __ http://en.wikipedia.org/wiki/Mercator_projection
    """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``gca().set_yscale("mercator")`` would
    # be used to select this scale.
    name = 'mercator'

    def __init__(self, axis, *, thresh=np.deg2rad(100), **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.
        thresh: The degree above which to crop the data.
        """
        thresh = 4000
        super().__init__(axis)
        self.thresh = thresh

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.
        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.MercatorLatitudeTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.
        In our case, the Mercator example uses a fixed locator from -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        fmt = FuncFormatter(
            lambda x, pos=None: f"{alige(int(math.pow(2, x))):.0f}")

        a = [5, 50, 100, 500, 2000, 4000]
        a = list(map(custom_log, a))
        axis.set(major_locator=FixedLocator(a),
                 major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, -self.thresh), min(vmax, self.thresh)

    class MercatorLatitudeTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = output_dims = 1

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)

            self.thresh = thresh

        def transform_non_affine(self, a):
            """
            This transform takes a numpy array and returns a transformed copy.
            Since the range of the Mercator scale is limited by the
            user-specified threshold, the input array must be masked to
            contain only valid values.  Matplotlib will handle masked arrays
            and remove the out-of-range data from the plot.  However, the
            returned array *must* have the same shape as the input array, since
            these values need to remain synchronized with values in the other
            dimension.
            """
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            return masked

        def inverted(self):
            """
            Override this method so Matplotlib knows how to get the
            inverse transform for this transform.
            """
            # return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(
            #     self.thresh)
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(
                self.thresh)

    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return np.arctan(np.sinh(a))

        def inverted(self):
            #           return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)
            return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)
def getf1(tp, match, all):
    prec = tp / match
    rec = tp/all
    return 2 * (prec * rec) / (prec + rec)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
patterns = ('/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*')
# patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')


# plt.style.use("seaborn-deep")
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn-muted')
# plt.style.use('classic')
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-pastel')
# plt.style.use('bmh')
# plt.style.use('ggplot')
# plt.style.use('seaborn-paper')


fig, axs = plt.subplots(3, 4)


#spark_er = [1362.117,1389.114, 1352.117*33991239/7633745,1362.117*34131239/7633745,1362.117*36096944/7633745]
spark_er = [1362.117,1389.114, 1352.117,1362.117,1362.117]
x_spark_er = [0.2, 0.4]
#dedoop = [30*30000000/183745,66*30100000/270001,33991239/400001*300,99999,99999]
dedoop = [30,66,90,120,150]
disdedup = [338+460, 449+508+330+310,1992,1295+1235,1916+1602]
#per_mqo = [49.101077*56,47.249413*57,203.481106*24,202.890326*24,211.290956*24]
per_mqo = [49.101077,47.249413,203.481106,202.890326,211.290956]

tim = 0
for i in range(5):
    tim += disdedup[i]/per_mqo[i]
#    tim += spark_er[i]/per_mqo[i]

print("tim:", tim, tim /5)


#per = [5,5,5,5,5]
x = [0.1, 0.2, 0.3, 0.4, 0.5]

# axs[2, 3].plot(x, with_mqo,   marker='o', color='mediumslateblue')
# axs[2, 3].plot(x, dedoop,   marker='+', color='khaki')

"""
gt_only:
    TFACC:
        sparkliclearn: 10w, data gen 460sec. exe: 338 sec.
        sparkliclearn: 20w, data gen (449 + 508) sec. exe: (330 + 310)sec.
        sparkliclearn: 30w, data gen (271) x 4 sec. exe: (227) x 4sec.
        sparkliclearn: 40w, data gen 1235 sec. exe: 1295 sec.
        sparkliclearn: 50w, data gen 1916 sec. exe: 1602 sec.
"""

axs[0, 3].plot(x, per_mqo, marker='*', color='crimson',  label='$DMatch$', markersize=9,  linewidth=2)
axs[0, 3].plot(x,[-100,-100,-100,-100,-100], marker='<', color='lightseagreen', label='$DMatch_{noMQO}$',linestyle=':', markersize=9,  linewidth=2)
axs[0, 3].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=9, label ='Sparker',  linewidth=2)
axs[0, 3].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', label='Dedoop', markersize=9,  linewidth=2)
axs[0, 3].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), label='Disdedup', markersize=9,  linewidth=2)
axs[0, 3].set_ylabel('time (sec.)', fontdict=font1)
#axs[0, 2].set_xlabel('$mio.$', fontdict=font1)
axs[0, 3].set_title('(d) TFACC: Varying $dup(mio)$', fontdict=font1, y=-0.35)
axs[0, 3].set_xlim([0.1, 0.5])
axs[0, 3].set_ylim([30, 2500])
axs[0, 3].tick_params(labelsize=12)
#for ytick in axs[0, 3].get_yticklabels():
#    ytick.set_rotation(30)





#similate: suppose that the whole data set is blocked, we just process one block. the final result are block num times one block time.
spark_er = [45.67359495162964, 76.46437311172485,105.16027498245239,126.60856580734253,150.44120454788208]
#dedoop = [(24096605/99809)*68,  (24096605/199809 )*68, (24136605/279595)*66, ()*67,2]
dedoop = [68 * 40.616*0.9897, 68*40.616*0.9917, 68*40.616*0.9934,68*40.616*0.9958,69*40.616]
dedoop = [30,29,31,37,30]
disdedup = [57,70,76,95,101]
per_mqo = [16.573876,22.39253,38.79271,57.52258, 66.15057]
per = [0,0,0,0,0]
x = [0.1, 0.2, 0.3, 0.4, 0.5]
axs[0, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=9,  linewidth=2 )
axs[0, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=9,  linewidth=2)
axs[0, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), markersize=9,  linewidth=2)
axs[0, 2].plot(x, per_mqo, marker='*', color='crimson',   markersize=9,  linewidth=2)
axs[0, 2].set_ylabel('time (sec.)', fontdict=font1)
#axs[0, 3].set_xlabel('$mio.$', fontdict=font1)
axs[0, 2].set_title('(c) TPCH: Varying $dup(mio)$', fontdict=font1, y=-0.35)
axs[0, 2].set_xlim([0.1, 0.5])
axs[0, 2].set_ylim([16, 160])
axs[0, 2].tick_params(labelsize=12)
#for ytick in axs[0, 2].get_yticklabels():
#    ytick.set_rotation(30)

#spark_er = [0,0.4504,0.4504,0.4504,0.4504,0.4504]
#spark_er = [0,0.37,0.37,0.37,0.37,0.37]
#dedoop = [0,0.49,0.49,0.49,0.49,0.49]
disdedup = [ math.log(485*(750595/200000),2), math.log(485*(1499580/200000),2), math.log(485*(2999672/200000),2), math.log(485*(6001216/200000),2),math.log(485*(11997997/200000),2)]
dedoop = [ math.log(1125,2), math.log(1125*2,2), math.log(3375,2), math.log(4500,2),math.log(5725,2)]
spark_er = [ math.log(3640,2), math.log(6181.8,2), math.log(9272,2), math.log(14823.3,2),math.log(18529,2)]
print(disdedup)
per_nomqo = [math.log(102),math.log(227.394), math.log(361.8362),math.log(499.939),math.log(639.472)]
per_mqo = [math.log(45.8263), math.log(107.95), math.log(171.314), math.log(233.52), math.log(299.958)]
x = [ 0.2, 0.4, 0.6, 0.8, 1]
axs[2, 3].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=9,  linewidth=2)
axs[2, 3].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=9,  linewidth=2)
axs[2, 3].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), markersize=9,  linewidth=2)
#axs[2, 2].plot(x, per_mqo, marker='*', color='crimson',  linestyle="dashdot",  markersize=7)
axs[2, 3].plot(x, per_nomqo, marker='<', color='lightseagreen',linestyle=':', markersize=9,  linewidth=2)
axs[2, 3].set_ylabel('time ($log$ sec.)', fontdict=font1)
#axs[1, 2].set_xlabel('$|D|$', fontdict=font1)
axs[2, 3].set_title('(l) TFACC: Varying $|D|$', fontdict=font1, y=-0.35)
axs[2, 3].set_ylim([3, 14])
axs[2, 3].set_xlim([0.2, 1.0])
axs[2, 3].tick_params(labelsize=12)
#for ytick in axs[2, 3].get_yticklabels():
#    ytick.set_rotation(30)



per_nomqo = [24.711, 55.303,114.863, 259.447, 607.32]
#spark_er = [188.766, 188.766*2, 188.766*4, 188.766*8, 188.766*16]
#spark_er = [188.766, 188.766*2, 188.766*4, 188.766*8, 188.766*16]
spark_er = [141.125, 588.2, 1170.87, 2337, 2337]
disdedup = [126.73, 126.73*2, 126.73*4, 126.73*8, 126.73*16]
per_mqo = [18.5693, 44.11, 99.88, 221.033, 504.977]
dedoop = [30, 30*2, 30*4, 30*8, 30*16]

dedoop = np.array(dedoop)
per_nomqo = np.array(per_nomqo)
per_mqo = np.array(per_mqo)
disdedup = np.array(disdedup)
spark_er = np.array(spark_er)

dedoop = np.log2(dedoop)
per_nomqo = np.log2(per_nomqo)
per_mqo = np.log2(per_mqo)
spark_er = np.log2(spark_er)
disdedup=np.log2(disdedup)

x = [ 0.05,0.1,0.25,0.5,1.0]
axs[2, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=9, linewidth=2)
axs[2, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=9, linewidth=2)
axs[2, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), markersize=9, linewidth=2)
axs[2, 2].plot(x, per_nomqo, marker='<', color='lightseagreen',linestyle=':', markersize=9, linewidth=2)
axs[2, 2].plot(x, per_mqo, marker='*', color='crimson',   markersize=9, linewidth=2)
axs[2, 2].set_ylabel('time ($log$ sec.)', fontdict=font1)
#axs[1, 3].set_xlabel('$|D|$', fontdict=font1)
axs[2, 2].set_title('(k) TPCH: Varying $|D|$', fontdict=font1, y=-0.35)
axs[2, 2].set_ylim([4, 11])
axs[2, 2].set_xlim([0.05, 1.0])
axs[2, 2].tick_params(labelsize=12)
#for ytick in axs[2, 2].get_yticklabels():
#    ytick.set_rotation(30)









#spark_er = [0,45.67359495162964,45.67359495162964,45.67359495162964,45.67359495162964,45.67359495162964]
#dedoop = [0,104,104,104,104,104]
#disdedup = [0,57,57,57,57,57]
per_mqo = [31.856, 59.6171, 91.315,500.567,504.977]
per_nomqo = [42.227,79.554,121.399,566.824,607.322]
x = [ 15, 30, 45, 60, 75]
#axs[0, 3].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
#axs[0, 3].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
#axs[0, 3].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )

axs[1, 2].plot(x, per_mqo, marker='*', color='crimson',   markersize=9,  linewidth=2)
axs[1, 2].plot(x, per_nomqo, marker='<', color='lightseagreen', linestyle=':', markersize=9,  linewidth=2 )

#axs[0,13].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[1, 2].set_ylabel('time (sec.)', fontdict=font1)
#axs[1, 1].set_xlabel('$|\Sigma|$', fontdict=font1)
axs[1, 2].set_title('(g) TPCH: Varying $||\Sigma||$', fontdict=font1, y=-0.35)
axs[1, 2].set_xlim([15, 75])
#axs[0, 3].set_ylim([0, 1.0])
axs[1, 2].tick_params(labelsize=12)
#for ytick in axs[1, 2].get_yticklabels():
#    ytick.set_rotation(30)
#for xtick in axs[1, 1].get_xticklabels():
#    xtick.set_rotation(30)


"""
tfacc:
    #dedoop tfacc_result_10w tp: 44596 fp: 3744  total: 134000 prec:0.922 recall:0.3328 f1: 0.489
    #dedoop tfacc_result_20w tp:  89174 fp: 9942 total: 270000 prec: 0.8996932886718593 recall: 0.33027407407407405 f1: 0.4831760205463865
    #dedoop tfacc_result_30w tp:  120416 fp:  43952 total: 400000  prec: 0.7326000194685097 recall: 0.30104 f1: 0.4267286593144899
    #dedoop tfacc_result_40w  tp:  160620 fp:  26494 total: 540000 prec: 0.8584071742360272 recall: 0.29744444444444446 f1: 0.4418014231606048
    #dedoop tfacc_result_50w tp:  200844 fp:  43740 total: 660000  prec: 0.8211657344715926 recall: 0.3043090909090909 f1: 0.4440582632458678

    #spakrlyclearn tfacc_result_10w: tp：16923 whole: 19081 prec:  0.8869032021382527 rec:  0.16932990464373981 f1:  0.28436759590663907
    #spakrlyclearn tfacc_result_20w: tp：35839 dec:40464 total:199808 prec:  0.8857 rec:  0.1793 f1:  0.298227
    #spakrlyclearn tfacc_result_30w: tp：65906 detect:71617 total:400000  prec:  0.9202563637125264 rec:  0.164765 f1:  0.27948950101459447
    #spakrlyclearn tfacc_result_40w: tp：34894+32492+16402 detect:36773+34012+18154 total:540000  prec:  0.9420839002012615 rec:  0.15516296296296297 f1:  0.26644237358471967
    #spakrlyclearn tfacc_result_50w: tp：109657 detect:120165 total:660000  prec:  0.9125535721715974 rec:  0.1661469696969697 f1:  0.2811123288022405
    
"""

#spark_er = [0.45, 1, 1, 1, 1]
#spark_er = [0.37,0.2229,0.2051,0.1312,0.1108]
#dedoop = [0.49, 0.48, 0.42, 0.4418014231606048, 0.44405826324586]
#disdedup = [0.28436759590663907, 0.298227, 0.27948950101459447, 0.26644, 0.281112]
#per_mqo = [0.854160773,0.850199439,0.856484633,0.850074491,0.860576862]
#per_deep = [0.68,0.68,0.68,0.68,0.68]
#per_collective = [0.58,0.58,0.58,0.58,0.58]
##per = [5, 5, 5, 5, 5]
#x = [0.1, 0.2,0.3,0.4,0.5]
## axs[2, 3].plot(x, with_mqo,   marker='o', color='mediumslateblue')
## axs[2, 3].plot(x, dedoop,   marker='+', color='khaki')
#axs[0, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5, )
#axs[0, 0].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=7)
#axs[0, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
##axs[0, 0].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
##axs[0, 0].plot(x, per_mqo, marker='2', color='darkred', markersize=8)
#axs[0, 0].plot(x, per_mqo, marker= "*", color='crimson',  linestyle="dashdot", label="$DMatch$", markersize=7)
#axs[0, 0].plot(x, per_deep, marker= "*", color='#bdb40c',  linestyle=":", label="$DMatch^{D}$", markersize=7)
#axs[0, 0].plot(x, per_collective, marker= "*", color='#ff5b00',  linestyle="--", label="$DMatch^{C}$", markersize=7)
#
#axs[0, 0].set_ylabel('F1', fontdict=font1)
##axs[0, 0].set_xlabel('$mio.$', fontdict=font1)
#axs[0, 0].set_title('(a) TFACC: Varying $dup(mio)$', fontdict=font1, y=-0.2)
#axs[0, 0].set_xlim([0.1, 0.5])
#axs[0, 0].set_ylim([0.1, 1])
#axs[0, 0].tick_params(labelsize=10)
#for ytick in axs[0, 0].get_yticklabels():
#    ytick.set_rotation(30)
width = 0.4
x_ = np.arange(6)
def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [0.44, 0.28, 0.11, 0.58, 0.68, 0.86]
axs[0, 1].bar(x_ + width, y1, width,
       color='skyblue',
    hatch="xxx"
             )
axs[0, 1].set_xticks(x_ + width)
axs[0, 1].plot(x__, y1, linestyle=":", color='grey')
axs[0, 1].set_xticklabels(['Dedoop', '$Disdedup$', '$Sparker$', '$DMatch_C$', '$DMatch_D$', '$DMatch$'])
axs[0, 1].set_ylabel('F1', fontdict=font1)
axs[0, 1].tick_params(labelsize=8)
#axs[0, 0].set_ylim([0.1, 0.9])
#axs[0, 0].set_xlim([0.1, 0.9])
axs[0, 1].set_title('(b) TFACC: Accuracy', fontdict=font1, y=-0.35)
for xtick in axs[0, 1].get_xticklabels():
    xtick.set_rotation(20)
#for ytick in axs[0, 1].get_yticklabels():
#    ytick.set_rotation(30)

"""
gt_only:
    TPCH:
        dedoop: 10w, single_match: 99942/2 all_math 557605 tp:  49288 fp:  56 prec: 0.9988651102464332 recall: 0.4933511503485829 f1: 0.6604823499063642
        dedoop: 20w, single_match: 99904 all_match 1088303 tp:  98547 fp:  189 prec: 0.9980858045697618 recall: 0.9864120234824257 f1: 0.9922145785980201
        dedoop: 30w, single_match: 139798 all_match 1263659  tp:  138052 fp:  393 prec: 0.9971613276030192 recall: 0.9875140828698653 f1: 0.9923142582459545
        dedoop: 40w, single_match: 199600 all_match 1595559 tp:  197047 fp:  786 prec: 0.996026952025193 recall: 0.9872069458743841 f1: 0.9915973364097389
        dedoop: 50w, single_match: 299098.0 all_match 2085851 tp:  295582 fp:  1766 prec: 0.9940608310800813 recall: 0.9882446555978308 f1: 0.9911442108757541
        
        customer:
            dedoop: 50w, single_match: 299098.0 all_match 2085851 tp:  295582 fp:  1766 prec: 0.9940608310800813 recall: 0.9882446555978308 f1: 0.9911442108757541
"""

#spark_er = [0.612577, 0.4918439591198297, 0.4030027209800559, 0.27916341475565, 0.27565971334990363]
##dedoop = [getf1(49228,99942/2,557605), getf1(98547, 99904,1088303), getf1(138052,139798,1263659), getf1(197047,199600,1595559), getf1(295582,299098,2085851)]
#dedoop = [0.65345138, 0.619535499,0.637043981,0.642176755,0.68459379]
#disdedup = [0.777174306,0.734008045,0.750800222,0.755016998,0.738586876]
#per_mqo = [0.916625743, 0.91646583, 0.916630957, 0.916526389, 0.915264417]
#per_deep = [0.68,0.68,0.68,0.68,0.68]
#per_collective = [0.58,0.58,0.58,0.58,0.58]
##per = [5, 5, 5, 5, 5]
#x = [0.1,0.2,0.3,0.4,0.5]
#axs[0, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5)
#axs[0, 1].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=7)
#axs[0, 1].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
##axs[0, 1].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
##axs[0, 1].plot(x, per_mqo, marker='2', color='darkred', markersize=8)
#axs[0, 1].plot(x, per_mqo, marker='*', color='crimson',  linestyle="dashdot",  markersize=7)
#axs[0, 1].plot(x, per_deep, marker= "*", color='#bdb40c',  linestyle=":", markersize=7)
#axs[0, 1].plot(x, per_collective, marker= "*", color='#ff5b00',  linestyle="--", markersize=7)
#
#axs[0, 1].set_ylabel('F1', fontdict=font1)
##axs[0, 1].set_xlabel('$mio.$', fontdict=font1)
#axs[0, 1].set_title('(b) TPCH: Varying $dup(mio)$', fontdict=font1, y=-0.2)
#axs[0, 1].set_xlim([0.1, 0.5])
#axs[0, 1].set_ylim([0.2, 1])
#axs[0, 1].tick_params(labelsize=10)
#for ytick in axs[0, 1].get_yticklabels():
#    ytick.set_rotation(30)

width = 0.4
x_ = np.arange(6)
def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [0.68459379, 0.738586876, 0.2756, 0.58, 0.68, 0.915]
axs[0, 0].bar(x_ + width, y1, width,
       color='skyblue',
    hatch="xxx"
             )
#axs[0, 1].set_yscale('log')
axs[0, 0].set_xticks(x_ + width)
axs[0, 0].plot(x__, y1, linestyle=":", color='grey')
axs[0, 0].set_xticklabels(['Dedoop', '$Disdedup$', '$Sparker$', '$DMatch_C$', '$DMatch_D$', '$DMatch$'])
axs[0, 0].set_ylabel('F1', fontdict=font1)
axs[0, 0].set_ylim([0.25, 0.95])
axs[0, 0].tick_params(labelsize=8)
axs[0, 0].set_title('(a) TPCH: Accuracy', fontdict=font1, y=-0.35)
#axs[0, 0].tick_params(labelsize=8.3)
for xtick in axs[0, 0].get_xticklabels():
    xtick.set_rotation(20)
#for ytick in axs[0, 0].get_yticklabels():
#    ytick.set_rotation(30)

"""
    tpacc:
        10w: 
            32 reducer - data gen: 460, exe model: 338
            24 reducer - data gen: 469, exe model: 376
            16 reducer - data gen: 510, exe model: 377
            8 reducer - data gen: 429, exe model: 366
            4 reducer - data gen: 441, exe model: 309
"""
spark_er = [1661.34, 1755.53, 1746.1, 1743.27, 1759.28]
#dedoop = [41*60, 41*60, 40*59, 41*43, 41*22]
#dedoop = [41, 41, 40, 41, 41]
#disdedup = [338+460, 469+376, 510+377, 429+366, 441+309]
#disdedup = []
per_mqo = [322, 249.261, 248.426, 251.643, 187.465]
#per = [209.251, 208.327,253.664, 303.949, 397.949]
per_nomqo =  [397.949, 303.949,253.664,208.327,209.251 ]
per_mqo = np.log2(per_mqo)
per_nomqo = np.log2(per_nomqo)
spark_er = np.log2(spark_er)
print("sdasd:", spark_er)
x = [4,8,16,24,32]
axs[2, 1].plot(x, per_mqo, marker='*', color='crimson',  markersize=9,  linewidth=2)
axs[2, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=9,  linewidth=2)
axs[2, 1].plot(x, per_nomqo, marker='<', color='lightseagreen',linestyle=':', markersize=9,  linewidth=2)

axs[2, 1].set_ylabel('time ($log$ sec.)', fontdict=font1)
#axs[2, 0].set_xlabel('$n$', fontdict=font1)
#axs[2, 0].set_xlabel('scale factor', fontdict=font1)
axs[2, 1].set_title('(j) TFACC: Varying $n$', fontdict=font1, y=-0.35)
axs[2, 1].set_xlim([4, 32])
#axs[2, 0].set_ylim([0, 1500])
axs[2, 1].tick_params(labelsize=12)
#for ytick in axs[2, 1].get_yticklabels():
#    ytick.set_rotation(30)

spark_er = [1220.788, 1224.29, 1207.74,1196.188,1178.11]
#spark_er = list(reversed(spark_er))
disdedup = [263, 142, 139, 150]
disdedup = np.array(disdedup)
disdedup = disdedup *8
x_ = [8,16,24,32]
#dedoop = [26, 25, 27, 30, 72]
#dedoop = list(reversed(dedoop))
per_nomqo = [49.79, 60.695, 65.795, 68.47, 71.77]
#per_nomqo = [71.77, 68.47,65.795,60.695,49.79]
per_nomqo = [245.6,217.7,126.5,70.2,57.3]
per_mqo = [36.8781, 39.23, 41.5, 46.4565, 51.45]
#per_mqo = [51.45, 46.4565, 41.5,39.23,36.8]
disdedup = np.log2(disdedup)
per_mqo = np.log2(per_mqo)
per_nomqo = np.log2(per_nomqo)
spark_er = np.log2(spark_er)
x = [4,8,16,24,32]
#axs[2, 1].plot(x, per_mqo, marker='*', color='crimson',  linestyle="dashdot",  markersize=7)
axs[2, 0].plot(x, per_nomqo, marker='<', color='lightseagreen', linestyle=':', markersize=9,  linewidth=2)
axs[2, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=9,  linewidth=2)
axs[2, 0].plot(x_, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), markersize=9,  linewidth=2)

axs[2, 0].set_ylabel('time ($log$ sec.)', fontdict=font1)
#axs[2, 1].set_xlabel('$n$', fontdict=font1)
axs[2, 0].set_title('(i) TPCH: Varying $n$', fontdict=font1, y=-0.35)
axs[2, 0].set_xlim([4, 32])
#axs[2, 1].set_ylim([0, 75])
axs[2, 0].tick_params(labelsize=12)
#for ytick in axs[2, 0].get_yticklabels():
#    ytick.set_rotation(30)

"""
colors = ['navy', 'cornflowerblue', 'deepskyblue']

x_ = [ 0.2, 0.4, 0.6, 0.8, 1]
population_by_continent = {
    '$local match$': list([15.7217,20.6634,36.1438,53.9623,61.0086]),
    '$closure$': list([0.852176,1.72913,2.64891,3.56028,5.14197]),
}

axs[2, 3].stackplot(x_, population_by_continent.values(), labels=population_by_continent.keys(), colors=colors)
axs[2, 3].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 3].set_xlabel('scale factor', fontdict=font1)
axs[2, 3].set_title('(l) TPCH: Cost Breakdown', fontdict=font1)
axs[2, 3].set_xlim([0.2, 1])
axs[2, 3].set_ylim([12, 68])
axs[2, 3].tick_params(labelsize=10)
for ytick in axs[2, 3].get_yticklabels():
    ytick.set_rotation(30)

x_ = [ 4, 8, 16, 24, 32]
population_by_continent = {
    '$local match$': list([22.533,19.5301,17.3519,16.7003,15.7581]),
    '$closure$': list([0.793491,0.84791,0.813199,0.810379,0.852537])
}
axs[2, 2].stackplot(x_, population_by_continent.values(), colors=colors)
axs[2, 2].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 2].set_xlabel('$n$', fontdict=font1)
axs[2, 2].set_title('(k) TPCH: Cost Breakdown', fontdict=font1)
axs[2, 2].set_xlim([4, 32])
axs[2, 2].set_ylim([15, 24])
axs[2, 2].tick_params(labelsize=10)
for ytick in axs[2, 2].get_yticklabels():
    ytick.set_rotation(30)
"""

spark_er = [ 1389.114,1389.114 ,1389.114,1389.114,1389.114]
dedoop = [30*30000000/183745,30*30000000/183745,30*30000000/183745,30*30000000/183745,30*30000000/183745]
disdedup = [338+460,338+460,338+460,338+460,338+460]
per_nomqo = [633.303,902.319,1182.66,1492.63,1678.07]
per = [501.685, 497.66,590.76,618.144, 639.203]
x = [ 10, 15, 20, 25, 30]
#axs[0, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
#axs[0, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
#axs[0, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
axs[1, 3].plot(x, per, marker='*', color='crimson',   markersize=9,  linewidth=2)
axs[1, 3].plot(x, per_nomqo, marker='<', color='lightseagreen', linestyle=':', markersize=9,  linewidth=2)
axs[1, 3].set_ylabel('time (sec.)', fontdict=font1)
#axs[1, 0].set_xlabel('$|\Sigma|$', fontdict=font1)
axs[1, 3].set_title('(h) TFACC: Varying $||\Sigma||$', fontdict=font1, y=-0.35)
axs[1, 3].set_xlim([10, 30])
#axs10,02].set_ylim([0, 1.0])
axs[1, 3].tick_params(labelsize=12)
#for ytick in axs[1, 3].get_yticklabels():
#    ytick.set_rotation(30)
#for xtick in axs[1, 0].get_xticklabels():
#    xtick.set_rotation(30)


per_nomqo = [59, 61.299, 63.9705, 63.427, 66.427]
per_mqo = [58.09, 60.28, 63.1646, 65.935, 67.89]
x = [4,5,6,7,8]
#axs[1, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
#axs[1, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
#axs[1, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
#axs[1, 2].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
#axs[1, 2].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
axs[1, 1].plot(x, per_mqo, marker='*', color='crimson',   markersize=9,  linewidth=2)
axs[1, 1].plot(x, per_nomqo, marker='<', color='lightseagreen',linestyle=':', markersize=9,  linewidth=2)
axs[1, 1].set_ylabel('time (sec.)', fontdict=font1)
#axs[2, 2].set_xlabel('$|X|$', fontdict=font1)
axs[1, 1].set_title(r'(f) TFACC: Varying $|\varphi|$', fontdict=font1, y=-0.35)
#axs[1, 2].set_xlim([0, 20])
axs[1, 1].set_xlim([4, 8])
axs[1, 1].tick_params(labelsize=14)
#for ytick in axs[2, 2].get_yticklabels():
#    ytick.set_rotation(30)

#per_nomqo = [3.4, 3.0, 3.1, 2.8, 2.3]
per_nomqo = [ 0.012, 0.885, 1.14,3.39,4.77]
#per_mqo = [58.09, 60.28, 63.1646, 65.935, 67.89]
x = [ 2,3,4,5,6]
#axs[1, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
#axs[1, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
#axs[1, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
#axs[1, 2].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
#axs[2, 3].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)
#axs[1,32].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
axs[1, 0].plot(x, per_nomqo, marker='<', color='lightseagreen',linestyle=':', markersize=9,  linewidth=2)

axs[1, 0].set_ylabel('time (sec.)', fontdict=font1)
#axs[2, 3].set_xlabel('$|X|$', fontdict=font1)
axs[1, 0].set_title(r'(e) TPCH: Varying $|\varphi|$', fontdict=font1, y=-0.35)
#axs[1, 2].set_xlim([0, 20])
axs[1, 0].set_xlim([2, 6])
axs[1, 0].tick_params(labelsize=12)
#for ytick in axs[1, 0].get_yticklabels():
#    ytick.set_rotation(30)



fig.legend(loc='upper center', ncol=14, fontsize=15)


t1 = [0.97, 0.96, 0.99, 0.98]
t2 = [0.71, 0.88, 0.44, 0.38]
#t2 = [0.71, 0.92, 0.99, 0.82]
#t2 = [0.91, 0.66, 0.37, 0.25]
#t2 = [0.71, 0.64, 0.47, 0.66]
#t2 = [0.67, 0.82, 0.9, 0.16]
#t2 = [0.53, 0.19, 0.65, 0.65]
#t2 = [0.66, 0.77, 0.03, 0.09]



#t2 = [0.28436759590663907, 0.298227, 0.27948950101459447, 0.26644, 0.281112]
#t2 = [0.49, 0.48, 0.42, 0.4418014231606048, 0.44405826324586]
#t2 = [0.37,0.2229,0.2051,0.1312,0.1108]
#t1 = [0.854160773,0.850199439,0.856484633,0.850074491,0.860576862]

t1 = np.array(t1)
t2 = np.array(t2)
print(t1)
print(t2)
res = 0
for i in range(4):
    res += t1[i] - t2[i]
    #print(res)
#    res += 1- (t1[i] / t2[i])
#    print(res)
res = res/4

print(res)
print("out:", 1-res)

plt.show()