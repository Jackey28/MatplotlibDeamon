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
         'size': 10,
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


spark_er = [1362.117,1389.114, 1352.117*33991239/7633745,1362.117*34131239/7633745,1362.117*36096944/7633745]
x_spark_er = [0.2, 0.4]
dedoop = [30*30000000/183745,66*30100000/270001,33991239/400001*300,99999,99999]
disdedup = [338+460, 449+508+330+310,1992,1295+1235,1916+1602]
per_mqo = [4,4,4,4,4]
per = [5,5,5,5,5]
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

axs[0, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5, label ='Sparker')
axs[0, 0].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', label='Dedoop', markersize=4)
axs[0, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), label='Disdedup')
axs[0, 0].plot(x, per, marker='2', color='crimson',  linestyle="dashdot", label='$MRLsMatch_{IH}$', markersize=8)
axs[0, 0].plot(x, per_mqo, marker='2', color='darkred', label='$MRLsMatch$', markersize=8)

axs[0, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 0].set_xlabel('scale factor', fontdict=font1)
axs[0, 0].set_title('(a) TPACC: Varying $dup(mio)$', fontdict=font1)
axs[0, 0].set_xlim([0.1, 0.5])
axs[0, 0].set_ylim([0, 10000])
axs[0, 0].tick_params(labelsize=10)
for ytick in axs[0, 0].get_yticklabels():
    ytick.set_rotation(30)





#similate: suppose that the whole data set is blocked, we just process one block. the final result are block num times one block time.
spark_er = [45.67359495162964, 76.46437311172485,105.16027498245239,126.60856580734253,150.44120454788208]
#dedoop = [(24096605/99809)*68,  (24096605/199809 )*68, (24136605/279595)*66, ()*67,2]
dedoop = [68 * 40.616*0.9897, 68*40.616*0.9917, 68*40.616*0.9934,68*40.616*0.9958,69*40.616]
dedoop = [30,29,31,37,30]
disdedup = [57,70,76,95,101]
per_mqo = [16.573876,22.39253,38.79271,57.52258, 66.15057]
per = [0,0,0,0,0]
x = [0.1, 0.2, 0.3, 0.4, 0.5]
axs[0, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[0, 1].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[0, 1].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
axs[0, 1].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
axs[0, 1].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[0, 1].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 1].set_xlabel('scale factor', fontdict=font1)
axs[0, 1].set_title('(b) TPCH: Varying $dup(mio)$', fontdict=font1)
axs[0, 1].set_xlim([0.1, 0.5])
axs[0, 1].set_ylim([0, 160])
axs[0, 1].tick_params(labelsize=10)
for ytick in axs[0, 1].get_yticklabels():
    ytick.set_rotation(30)


spark_er = [0,0.4504,0.4504,0.4504,0.4504,0.4504]
spark_er = [0,0.37,0.37,0.37,0.37,0.37]
dedoop = [0,0.49,0.49,0.49,0.49,0.49]
disdedup = [0, 0.28436759590663907, 0.28436759590663907, 0.28436759590663907, 0.28436759590663907,0.28436759590663907]
per_mqo = [0,4,4,4,4,4]
per = [0,5,5,5,5,5]
x = [0, 1, 5, 10, 15, 20]
axs[1, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[1, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[1, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
#axs[1, 2].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
#axs[1, 2].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[1, 2].set_ylabel('F1', fontdict=font1)
axs[1, 2].set_xlabel('$|\Sigma|$', fontdict=font1)
axs[1, 2].set_title('(g) TFACC: Varying $|\Sigma|$', fontdict=font1)
axs[1, 2].set_xlim([0, 20])
axs[1, 2].set_ylim([0, 1.0])
axs[1, 2].tick_params(labelsize=10)
for ytick in axs[1, 2].get_yticklabels():
    ytick.set_rotation(30)


spark_er = [0,0.612577,0.612577,0.612577,0.612577,0.612577]
dedoop = [0,0.65345138,0.65345138,0.65345138,0.65345138,0.65345138]
disdedup = [0,0.777174306,0.777174306,0.777174306,0.777174306,0.777174306]
per_mqo = [0,1,1,1,1,1]
per = [0,1,1,1,1,1]
x = [0, 1, 5, 10, 15, 20]
axs[1, 3].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[1, 3].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[1, 3].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
#axs[1, 3].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
#axs[1, 3].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[1, 3].set_ylabel('F1', fontdict=font1)
axs[1, 3].set_xlabel('$|\Sigma|$', fontdict=font1)
axs[1, 3].set_title('(h) TFACC: Varying $|\Sigma|$', fontdict=font1)
axs[1, 3].set_xlim([0, 20])
axs[1, 3].set_ylim([0, 1])
axs[1, 3].tick_params(labelsize=10)
for ytick in axs[1, 3].get_yticklabels():
    ytick.set_rotation(30)



spark_er = [0, 1389.114,1389.114 ,1389.114,1389.114,1389.114]
dedoop = [0,30*30000000/183745,30*30000000/183745,30*30000000/183745,30*30000000/183745,30*30000000/183745]
disdedup = [0,338+460,338+460,338+460,338+460,338+460]
#per_mqo = [0,4,4,4,4,4]
#per = [0,5,5,5,5,5]
x = [0, 1, 5, 10, 15, 20]
axs[0, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[0, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[0, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
#axs[0, 2].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
#axs[0, 2].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[0, 2].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 2].set_xlabel('$|\Sigma|$', fontdict=font1)
axs[0, 2].set_title('(c) TFACC: Varying $|\Sigma|$', fontdict=font1)
axs[0, 2].set_xlim([0, 20])
#axs[0, 2].set_ylim([0, 1.0])
axs[0, 2].tick_params(labelsize=10)
for ytick in axs[0, 2].get_yticklabels():
    ytick.set_rotation(30)



spark_er = [0,45.67359495162964,45.67359495162964,45.67359495162964,45.67359495162964,45.67359495162964]
dedoop = [0,30,30,30,30,30]
disdedup = [0,57,57,57,57,57]
#per_mqo = [0,4,4,4,4,0]
#per = [0,5,5,5,5,5]
x = [0, 1, 5, 10, 15, 20]
axs[0, 3].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[0, 3].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[0, 3].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
#axs[0, 3].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
#axs[0, 3].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[0, 3].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 3].set_xlabel('$|\Sigma|$', fontdict=font1)
axs[0, 3].set_title('(d) TPCH: Varying $|\Sigma|$', fontdict=font1)
axs[0, 3].set_xlim([0, 20])
#axs[0, 3].set_ylim([0, 1.0])
axs[0, 3].tick_params(labelsize=10)
for ytick in axs[0, 3].get_yticklabels():
    ytick.set_rotation(30)


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
spark_er = [0.37,0.2229,0.2051,0.1312,0.1108]
dedoop = [0.49, 0.48, 0.42, 0.4418014231606048, 0.44405826324586]
disdedup = [0.28436759590663907, 0.298227, 0.27948950101459447, 0.26644, 0.281112]
per_mqo = [4, 4, 4, 4, 4]
per = [5, 5, 5, 5, 5]
x = [0.2, 0.4, 0.6, 0.8, 1]
# axs[2, 3].plot(x, with_mqo,   marker='o', color='mediumslateblue')
# axs[2, 3].plot(x, dedoop,   marker='+', color='khaki')
axs[1, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5, )
axs[1, 0].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=4)
axs[1, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
axs[1, 0].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
axs[1, 0].plot(x, per_mqo, marker='2', color='darkred', markersize=8)

axs[1, 0].set_ylabel('F1', fontdict=font1)
axs[1, 0].set_xlabel('scale factor', fontdict=font1)
axs[1, 0].set_title('(e) TFACC: Varying $dup(mio)$', fontdict=font1)
axs[1, 0].set_xlim([0.2, 1])
axs[1, 0].set_ylim([0, 1])
axs[1, 0].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(30)
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

spark_er = [0.612577, 0.4918439591198297, 0.4030027209800559, 0.27916341475565, 0.27565971334990363]
#dedoop = [getf1(49228,99942/2,557605), getf1(98547, 99904,1088303), getf1(138052,139798,1263659), getf1(197047,199600,1595559), getf1(295582,299098,2085851)]
dedoop = [0.65345138, 0.619535499,0.637043981,0.642176755,0.68459379]
disdedup = [0.777174306,0.734008045,0.750800222,0.755016998,0.738586876]
per_mqo = [0.916625743, 0.91646583, 0.916630957, 0.916526389, 0.915264417]
per = [5, 5, 5, 5, 5]
x = [0.2, 0.4, 0.6, 0.8, 1]
axs[1, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5)
axs[1, 1].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=4)
axs[1, 1].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
axs[1, 1].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
axs[1, 1].plot(x, per_mqo, marker='2', color='darkred', markersize=8)

axs[1, 1].set_ylabel('F1', fontdict=font1)
axs[1, 1].set_xlabel('scale factor', fontdict=font1)
axs[1, 1].set_title('(f) TPCH: Varying $dup(mio)$', fontdict=font1)
axs[1, 1].set_xlim([0.2, 1])
axs[1, 1].set_ylim([0.2, 1])
axs[1, 1].tick_params(labelsize=10)
for ytick in axs[1, 1].get_yticklabels():
    ytick.set_rotation(30)


"""
    tpacc:
        10w: 
            32 reducer - data gen: 460, exe model: 338
            24 reducer - data gen: 469, exe model: 376
            16 reducer - data gen: 510, exe model: 377
            8 reducer - data gen: 429, exe model: 366
            4 reducer - data gen: 441, exe model: 309
            
            
"""
spark_er = [1389.138, 1372.726, 1370.04, 1379.25, 1362.117]
dedoop = [41*60, 41*60, 40*59, 41*43, 41*22]
disdedup = [338+460, 469+376, 510+377, 429+366, 441+309]
per_mqo = [4, 4, 4, 4, 4]
per = [5, 5, 5, 5, 5]
x = [4,8,16,24,32]
axs[2, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5)
axs[2, 0].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=4)
axs[2, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
axs[2, 0].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
axs[2, 0].plot(x, per_mqo, marker='2', color='darkred', markersize=8)

axs[2, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[2, 0].set_xlabel('$n$', fontdict=font1)
#axs[2, 0].set_xlabel('scale factor', fontdict=font1)
axs[2, 0].set_title('(i) TFACC: Varying $n$', fontdict=font1)
axs[2, 0].set_xlim([4, 32])
axs[2, 0].set_ylim([0, 3000])
axs[2, 0].tick_params(labelsize=10)
for ytick in axs[2, 0].get_yticklabels():
    ytick.set_rotation(30)

spark_er = [27.14, 29.18, 29.11, 28.03, 46.03]
spark_er = list(reversed(spark_er))
dedoop = [26, 25, 27, 30, 72]
dedoop = list(reversed(dedoop))
disdedup = [56, 55, 55, 54, 54]
per_mqo = [23.326491,20.37801,18.165099,17.510679, 16.610637]
per = [5, 5, 5, 5, 5]
x = [4,8,16,24,32]
axs[2, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5)
axs[2, 1].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=4)
axs[2, 1].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
#axs[2, 1].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
axs[2, 1].plot(x, per_mqo, marker='2', color='darkred', markersize=8)

axs[2, 1].set_ylabel('time (sec.)', fontdict=font1)
axs[2, 1].set_xlabel('$n$', fontdict=font1)
axs[2, 1].set_title('(j) TPCH: Varying $n$', fontdict=font1)
axs[2, 1].set_xlim([4, 32])
axs[2, 1].set_ylim([0, 75])
axs[2, 1].tick_params(labelsize=10)
for ytick in axs[2, 1].get_yticklabels():
    ytick.set_rotation(30)


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



fig.legend(loc='upper center', ncol=4, fontsize=10)

plt.show()