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


fig, axs = plt.subplots(4, 4)


spark_er = [1362.117,1389.114, 6076,6101.6,6453]
x_spark_er = [0.2, 0.4]
dedoop = [2,2,2,2,2]
disdedup = [3,3,3,3,3]
per_mqo = [4,4,4,4,4]
per = [5,5,5,5,5]
x = [0.2, 0.4, 0.6, 0.8, 1.0]
# axs[2, 3].plot(x, with_mqo,   marker='o', color='mediumslateblue')
# axs[2, 3].plot(x, dedoop,   marker='+', color='khaki')
axs[0, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5, label ='Sparker')
axs[0, 0].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', label='Dedoop', markersize=4)
axs[0, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), label='Disdedup')
axs[0, 0].plot(x, per, marker='2', color='crimson',  linestyle="dashdot", label='$MRLsMatch_{IH}$', markersize=8)
axs[0, 0].plot(x, per_mqo, marker='2', color='darkred', label='$MRLsMatch$', markersize=8)

axs[0, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 0].set_xlabel('scale factor', fontdict=font1)
axs[0, 0].set_title('(a) TFACC: Varying $|D|$', fontdict=font1)
axs[0, 0].set_xlim([0.2, 1.0])
#axs[0, 0].set_ylim([0, 5000])
axs[0, 0].tick_params(labelsize=10)
for ytick in axs[0, 0].get_yticklabels():
    ytick.set_rotation(30)

spark_er = [0.4504,0.2229,0.2051,0.1312,0.1108]
dedoop = [2,2,2,2,2]
disdedup = [3,3,3,3,3]
per_mqo = [4,4,4,4,4]
per = [5,5,5,5,5]
x = [0.2, 0.4, 0.6, 0.8, 1.0]
axs[0, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[0, 1].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[0, 1].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
axs[0, 1].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
axs[0, 1].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[0, 1].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 1].set_xlabel('scale factor', fontdict=font1)
axs[0, 1].set_title('(b) TPCH: Varying $|D|$', fontdict=font1 )
axs[0, 1].set_xlim([0.2, 1.0])
axs[0, 1].set_ylim([0, 1.0])
axs[0, 1].tick_params(labelsize=10)
for ytick in axs[0, 1].get_yticklabels():
    ytick.set_rotation(30)


spark_er = [1,1,1,1,1]
dedoop = [2,2,2,2,2]
disdedup = [3,3,3,3,3]
per_mqo = [4,4,4,4,4]
per = [5,5,5,5,5]
x = [0.2, 0.4, 0.6, 0.8, 1.0]
axs[0, 2].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5 )
axs[0, 2].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':',  markersize=4)
axs[0, 2].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), )
axs[0, 2].plot(x, per, marker='2', color='crimson',  linestyle="dashdot",  markersize=8)
axs[0, 2].plot(x, per_mqo, marker='2', color='darkred',  markersize=8)

axs[0, 2].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 2].set_xlabel('scale factor', fontdict=font1)
axs[0, 2].set_title('(b) TPCH: Varying $|D|$', fontdict=font1)
axs[0, 2].set_xlim([0.2, 1.0])
axs[0, 2].tick_params(labelsize=10)
for ytick in axs[0, 1].get_yticklabels():
    ytick.set_rotation(30)


spark_er = [0.37, 1, 1, 1, 1]
dedoop = [2, 2, 2, 2, 2]
disdedup = [3, 3, 3, 3, 3]
per_mqo = [4, 4, 4, 4, 4]
per = [5, 5, 5, 5, 5]
x = [0.2, 0.4, 0.6, 0.8, 1.0]
# axs[2, 3].plot(x, with_mqo,   marker='o', color='mediumslateblue')
# axs[2, 3].plot(x, dedoop,   marker='+', color='khaki')
axs[1, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5, )
axs[1, 0].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=4)
axs[1, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
axs[1, 0].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
axs[1, 0].plot(x, per_mqo, marker='2', color='darkred', markersize=8)

axs[1, 0].set_ylabel('F1', fontdict=font1)
axs[1, 0].set_xlabel('scale factor', fontdict=font1)
axs[1, 0].set_title('(e) TFACC: Varying $|D|$', fontdict=font1)
axs[1, 0].set_xlim([0.2, 1.0])
# axs[0, 0].set_ylim([0.2, 1])
axs[0, 0].tick_params(labelsize=10)
for ytick in axs[0, 0].get_yticklabels():
    ytick.set_rotation(30)

spark_er = [1, 1, 1, 1, 1]
dedoop = [2, 2, 2, 2, 2]
disdedup = [3, 3, 3, 3, 3]
per_mqo = [4, 4, 4, 4, 4]
per = [5, 5, 5, 5, 5]
x = [0.2, 0.4, 0.6, 0.8, 1.0]
axs[1, 1].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5)
axs[1, 1].plot(x, dedoop, marker='x', color='darkslateblue', linestyle=':', markersize=4)
axs[1, 1].plot(x, disdedup, marker='o', color='olive', linestyle=(0, (5, 1)), )
axs[1, 1].plot(x, per, marker='2', color='crimson', linestyle="dashdot", markersize=8)
axs[1, 1].plot(x, per_mqo, marker='2', color='darkred', markersize=8)

axs[1, 1].set_ylabel('F1', fontdict=font1)
axs[1, 1].set_xlabel('scale factor', fontdict=font1)
axs[1, 1].set_title('(b) TPCH: Varying $|D|$', fontdict=font1)
axs[1, 1].set_xlim([0.2, 1.0])
axs[1, 1].tick_params(labelsize=10)
for ytick in axs[1, 1].get_yticklabels():
    ytick.set_rotation(30)

spark_er = [1389.138, 1372.726, 1370.04, 1379.25, 1362.117]
dedoop = [2, 2, 2, 2, 2]
disdedup = [3, 3, 3, 3, 3]
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
axs[2, 0].tick_params(labelsize=10)
for ytick in axs[2, 0].get_yticklabels():
    ytick.set_rotation(30)


fig.legend(loc='upper center', ncol=6, fontsize=10)

plt.show()