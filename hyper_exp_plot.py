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


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,

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


fig, axs = plt.subplots(4, 4)

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
ax2.set_ylabel('F1', fontdict=font1)
# ax2.plot(x, cmp,  color='blue',    markersize=5, alpha=0.1)
ax2.fill_between(x, rec, baseline_, facecolor='black', label="recall", alpha=0.1)
ax2.set_ylim([0.8, 1.0])
# ax2.set_xlim([0, 127])
axs[0, 0].plot(x, hyper, marker='x', color='orange', linestyle="--", label='$Hyper$', markersize=7)
# axs[0, 0].plot(x, hyper_inc, marker='*', color='tomato', linestyle=":",label='$Hyper_{inc}$', markersize=7)
# axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[0, 0].plot(x, hyper_ext, marker='<', color='#42b395', linestyle=":", label='$Hyper_{ext}$', markersize=7)
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
ax2.set_ylabel('F1', fontdict=font1)
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
axs[0, 1].set_title('(b) Walmart: Varying $|\mathical{B}|}$', fontdict=font1)
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
ax2.set_ylabel('F1', fontdict=font1)
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
axs[0, 2].set_title('(c) Songs: Varying $|\mathical{B}|$', fontdict=font1)
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
ax2.set_ylabel('F1', fontdict=font1)
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
axs[0, 3].set_title('(d) NCV: Varying $|\mathical{B}|$', fontdict=font1)
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

"""
np.random.seed(19680801)

# load up some sample financial data
r = (cbook.get_sample_data('goog.npz', np_load=True)['price_data']
     .view(np.recarray))
pricemin = r.close.min()
axs[3, 0].plot(r.date, r.close, lw=2)

"""

time_hyper = [274.83,
              437.03,
              269.25,
              250.54,
              254.86,
              386.00,
              386.70,
              374.09,
              363.11,
              346.30,
              372.61,
              358.92,
              346.96,
              354.28,
              350.78,
              333.38,
              369.85,
              229.25,
              348.54,
              370.71,
              483.67,
              230.80,
              380.08,
              363.60,
              418.54,
              401.65,
              330.89,
              196.91,
              351.02,
              387.06,
              339.51,
              338.28,
              362.87,
              339.76,
              237.57,
              354.16,
              207.93,
              402.24,
              347.59,
              216.87,
              375.30,
              329.02,
              340.29,
              405.59,
              374.21,
              284.90,
              413.97,
              415.31,
              339.82,
              332.54,
              419.12,
              378.50,
              360.95,
              251.95,
              340.42,
              316.49,
              368.16,
              371.67,
              334.38,
              349.78,
              338.67,
              363.98,
              388.02,
              347.52,
              395.72,
              320.60,
              271.80,
              403.16,
              294.03,
              420.67,
              299.79,
              389.05,
              340.19,
              408.42,
              217.46,
              356.64,
              319.63,
              308.94,
              352.75,
              418.89,
              264.42,
              335.11,
              307.21,
              397.36,
              319.61,
              358.23,
              389.68,
              285.90,
              365.35,
              311.19,
              377.85,
              460.10,
              362.13,
              318.77,
              344.23,
              324.70,
              277.67,
              394.55,
              348.13,
              302.05,
              396.27,
              341.34,
              293.05,
              333.26,
              342.93,
              297.73,
              299.04,
              483.82,
              339.66,
              384.00,
              350.92,
              381.21,
              213.96,
              338.37,
              343.46,
              355.48,
              352.63,
              362.32,
              286.63,
              373.34,
              264.08,
              220.12,
              396.25,
              378.94,
              339.75,
              352.24,
              341.52,
              333.92]

cmp = [24720784,
       29975625,
       24354225,
       24324624,
       25060036,
       25623844,
       26378496,
       23785129,
       19456921,
       24314761,
       22061809,
       21846276,
       22458121,
       21206025,
       21455424,
       25230529,
       18774889,
       18757561,
       21585316,
       23629321,
       26460736,
       18481401,
       26071236,
       25472209,
       30371121,
       27931225,
       17472400,
       14100025,
       23435281,
       23629321,
       26132544,
       19873764,
       19954089,
       23270976,
       22992025,
       18139081,
       15437041,
       15944049,
       28334329,
       24393721,
       22024249,
       24940036,
       21372129,
       28783225,
       25421764,
       18284176,
       21252100,
       27825625,
       28654609,
       29322225,
       23261329,
       24930049,
       21538881,
       23435281,
       18905104,
       18507204,
       24117921,
       25573249,
       20666116,
       21492496,
       19660356,
       24681024,
       26450449,
       22591009,
       21706281,
       20611600,
       27217089,
       29735209,
       28026436,
       18028516,
       20250000,
       26091664,
       28547649,
       22344529,
       16516096,
       17472400,
       23648769,
       22743361,
       19201924,
       17089956,
       19545241,
       30735936,
       26863489,
       19456921,
       20493729,
       21270544,
       26450449,
       25250625,
       19474569,
       25190361,
       24186724,
       32148900,
       25010001,
       19342404,
       19829209,
       21446161,
       17749369,
       24019801,
       18361225,
       27426169,
       26904969,
       22184100,
       20967241,
       18438436,
       22108804,
       22819729,
       17842176,
       23571025,
       24186724,
       26327161,
       37234404,
       23824161,
       15745024,
       19749136,
       22534009,
       24324624,
       23164969,
       18939904,
       23435281,
       25240576,
       25583364,
       16867449,
       16000000,
       22372900,
       27321529,
       22165264,
       20466576,
       23088025]
time_randome_allocation_threads = [394.48,
                                   432.11,
                                   397.59,
                                   370.69,
                                   375.32,
                                   393.94,
                                   388.97,
                                   372.56,
                                   359.14,
                                   353.25,
                                   374.61,
                                   469.77,
                                   364.57,
                                   464.93,
                                   358.30,
                                   333.22,
                                   378.56,
                                   329.11,
                                   356.24,
                                   374.91,
                                   390.06,
                                   392.23,
                                   385.45,
                                   344.15,
                                   503.65,
                                   415.05,
                                   226.20,
                                   232.02,
                                   365.32,
                                   378.08,
                                   387.88,
                                   381.71,
                                   347.73,
                                   374.09,
                                   221.43,
                                   235.23,
                                   376.96,
                                   211.15,
                                   372.11,
                                   380.66,
                                   429.81,
                                   362.64,
                                   439.24,
                                   377.04,
                                   330.97,
                                   374.77,
                                   405.73,
                                   472.19,
                                   476.77,
                                   433.91,
                                   390.33,
                                   387.80,
                                   398.93,
                                   384.01,
                                   465.67,
                                   354.08,
                                   334.26,
                                   398.64,
                                   375.28,
                                   355.19,
                                   368.90,
                                   371.20,
                                   382.32,
                                   367.40,
                                   392.90,
                                   489.50,
                                   365.78,
                                   459.62,
                                   277.39,
                                   441.42,
                                   413.51,
                                   340.82,
                                   242.20,
                                   433.57,
                                   364.56,
                                   368.27,
                                   271.44,
                                   411.87,
                                   401.18,
                                   342.15,
                                   230.66,
                                   407.83,
                                   344.99,
                                   428.94,
                                   423.52,
                                   357.63,
                                   390.79,
                                   356.15,
                                   429.87,
                                   383.76,
                                   401.69,
                                   442.48,
                                   405.04,
                                   407.57,
                                   393.66,
                                   269.77,
                                   355.17,
                                   308.25,
                                   361.47,
                                   390.29,
                                   406.11,
                                   370.93,
                                   430.74,
                                   378.06,
                                   337.26,
                                   386.02,
                                   297.29,
                                   379.34,
                                   402.14,
                                   393.48,
                                   401.66,
                                   496.34,
                                   400.37,
                                   390.77,
                                   360.25,
                                   281.64,
                                   397.23,
                                   344.95,
                                   411.24,
                                   378.96,
                                   226.52,
                                   412.87,
                                   238.93,
                                   398.18,
                                   358.08,
                                   381.06,
                                   415.80,
                                   369.74]

# avg_hyper = [ 415.73, 278.42, 167.75, 137.53]
# min_hyper = [301.53, 232.2, 86.407, 72.88]
# max_hyper = [ 626.83, 391.23, 252, 175.0]
# avg_hyper = np.array(avg_hyper)
# min_hyper = np.array(min_hyper)
# max_hyper = np.array(max_hyper)
# max_cmp = np.array([ 98942809,  ])
# min_cmp = np.array([ 66064384, ])
# avg_cmp = (max_cmp+min_cmp)/2
x = [i for i in range(128)]
baseline_ = [0 for i in range(128)]
x = np.sort(x)
time_hyper = np.sort(time_hyper)
time_randome_allocation_threads = np.sort(time_randome_allocation_threads)
cmp = np.sort(cmp)
print(x)
print(time_hyper)
print(cmp)
# axs[2, 0].plot(x, avg_hyper)
axs[3, 0].plot(x, time_hyper, color='orange', linestyle=":", markersize=5)
axs[3, 0].plot(x, time_randome_allocation_threads, linestyle=":", color='royalblue', markersize=5)
# axs[2, 0].plot(x, cmp, color='black',  linestyle="--",  markersize=5)
# axs[2, 0].fill_between(x, time_hyper, baseline_, facecolor='orange', alpha=0.2)
axs[3, 0].set_ylim([196.91, 600])
# axs[2, 0].set_ylim([0,600])
# axs[2, 0].tick_params(labelsize=10)
ax2 = axs[3, 0].twinx()
# ax2.plot(x, cmp,  color='blue',    markersize=5, alpha=0.1)
ax2.fill_between(x, cmp, baseline_, facecolor='black', label="distribution of cmp", alpha=0.1)
ax2.set_ylim([0, 37234404])
ax2.set_xlim([0, 127])

axs[3, 0].set_xlim([0, 127])
axs[3, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[3, 0].set_xlabel('$block$ $id$', fontdict=font1)
axs[3, 0].set_title('(a) TPCH: handling skewness$', fontdict=font1)
axs[3, 0].tick_params(labelsize=10)
for ytick in axs[3, 0].get_yticklabels():
    ytick.set_rotation(30)

fig.legend(loc='upper center', ncol=8, fontsize=10)

plt.show()
