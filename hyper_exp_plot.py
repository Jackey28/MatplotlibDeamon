
import numpy as np
#from basic_units import cm, inch
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
            lambda x, pos=None: f"{alige(int(math.pow(2,x))):.0f}")
        
        a = [5, 50, 100,  500, 2000, 4000]
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




font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
         }
patterns = ('/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*')
#patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')


#plt.style.use("seaborn-deep")
#plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-muted')
#plt.style.use('classic')
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-pastel')
#plt.style.use('bmh')
#plt.style.use('ggplot')
#plt.style.use('seaborn-paper')


fig, axs = plt.subplots(5, 4)


colors = ['navy', 'cornflowerblue', 'deepskyblue']

plt.legend(loc='lower center') # 标签位置

spark_er = [0,0,0,0,0]
dedoop = [0,0,0,0,0]
disdedup = [0,0,0,0,0]
#hyper = [(26.326664 + 24.401422)/2, (48.479328 + 48.601736)/2, (131.609672 + 139.679083)/2,  (138.982280 + 148.364814+147.848467 +133 + 134.616868)/5, 176.615500]
hyper = [24.68  , 42.2843, 91.5795,  144.55, 164.08]
#hyper_inc = [16.332565, 23.053892, 41.854730, 60.919483, 73.585759]
hyper_inc = [17.2677, 26.4467, 44.5676, 63.3853, 68.725]
#hyper_nml = [ 23.95,42.094,89.075, 143.322,164.301]
hyper_ext = [ 24.379791, 45.910275 , 85.566736, 123.502402, 136.503737]
hyper_tqm = [5,5,5,5,5]
x = [1, 2, 4, 6, 8]

axs[0, 0].plot(x, spark_er, marker='d', color='mediumslateblue', linestyle='--', markersize=5, label ='Sparker')
axs[0, 0].plot(x, dedoop, marker='*', color='darkslateblue', linestyle='--', label='Dedoop', markersize=4)
axs[0, 0].plot(x, disdedup, marker='o', color='olive', linestyle=(0,(5,1)), label='Disdedup')
axs[0, 0].plot(x, hyper, marker='2', color='crimson',  linestyle="--", label='$Hyper$', markersize=8)
axs[0, 0].plot(x, hyper_inc, marker='3', color='k', linestyle=":",label='$Hyper_{inc}$', markersize=8)
#axs[0, 0].plot(x, hyper_nml, marker='*', color='lightseagreen', linestyle=":",label='$Hyper_{nml}$', markersize=8)
axs[0, 0].plot(x, hyper_ext, marker='1', color='tomato', linestyle=":",label='$Hyper_{ext}$', markersize=8)
axs[0, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":",label='$Hyper_{tqm}$', markersize=8)

axs[0, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 0].set_xlabel('$mio$', fontdict=font1)
axs[0, 0].set_title('(a) TPCH: Varying $|D|$', fontdict=font1)
axs[0, 0].set_xlim([1, 8    ])
axs[0, 0].set_ylim([0, 200])
axs[0, 0].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(30)


#hyper = [ 212.545285, 137.476279, 117.308165, 104.527871, 97.416963, 89.340965, 88.380350]
hyper = [ 323.666747, 183.821694, 0,158.845615,0,0,0]
hyper = [  271.084665,  187.453172,  149.914119, 151.032481, 134.543713, 133.198432, 133.041249]
hyper_inc = [0,0, 68.250209, 70.410183, 70.957951, 71.437485, 0]
#hyper_nml = [ 192.156395, 128.776147, 109.266810, 99.405487, 92.458567, 90.039718, 94.679042 ]
#hyper_tqm = [5,5,5,5,5]
x = [1, 2, 3, 4, 5,6,7]
axs[1, 0].plot(x, hyper, marker='2', color='crimson',  linestyle="-",  markersize=8)
axs[1, 0].plot(x, hyper_inc, marker='3', color='k', linestyle=":", markersize=8)
#axs[1, 0].plot(x, hyper_nml, marker='x', color='lightseagreen', linestyle=":", markersize=8)
#axs[1, 0].plot(x, hyper_tqm, marker='+', color='orange', linestyle=":", markersize=8)


axs[1, 0].set_ylabel('time (sec.)', fontdict=font1)
axs[1, 0].set_xlabel('$n$', fontdict=font1)
axs[1, 0].set_title('(a) TPCH: Varying $n$', fontdict=font1)
axs[1, 0].set_xlim([1, 7])
axs[1, 0].set_ylim([0, 300])
axs[1, 0].tick_params(labelsize=10)
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
fig.legend(loc='upper center', ncol = 8, fontsize=10)

plt.show()