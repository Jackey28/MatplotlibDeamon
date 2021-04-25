
import numpy as np
#from basic_units import cm, inch
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


fig, axs = plt.subplots(3, 4)


colors = ['navy', 'cornflowerblue', 'deepskyblue']

plt.legend(loc='lower center') # 标签位置





labels = ['DBLP', 'Walmart', 'Songs', 'NCV']
jedai = [0.88, 0.3437, 0.125, 0.04]
dedoop = [0.19, 0.4375, 0.95675,0.84375]
DM = [0.92, 0.28125,0.99, 0.96]
CUER = [0.907, 0.4375,0.98, 0.84375]
#deep_matcher = [0.9423, 0.9168, 0.9927, 0.5811, 0.9989]
x = np.arange(len(labels))  # the label locations
width = 0.65 # the width of the bars

rects1 = axs[0, 0].bar(x - width/2, jedai, width/4, label='ERBlox', color = 'grey', hatch=patterns[4],edgecolor='black', linewidth = 0.1)
rects2 = axs[0, 0].bar(x - width/4, dedoop, width/4, label='DM', color = 'white', hatch=patterns[5], edgecolor='black', linewidth = 0.1)
rects3 = axs[0, 0].bar(x , DM, width/4, label='Dedoop', color = 'whitesmoke', hatch=patterns[7],edgecolor='black', linewidth = 0.1)
rects5 = axs[0, 0].bar(x + width/4, CUER, width/4, label='CUER', color = 'black', hatch=patterns[1], edgecolor='black', linewidth = 0.1)

axs[0, 0].set_ylabel('F1-score', fontdict=font1)
axs[0, 0].set_xticks(x)
axs[0, 0].set_title('(a) Accuracy',  fontdict=font1)
axs[0, 0].set_ylim([0.0, 1])
axs[0, 0].set_xticklabels(labels)
axs[0, 0].tick_params(labelsize=7.5)
#axs[0, 0].tick_params(labelsize=10)

for xtick in axs[0, 0].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 0].get_yticklabels():
    ytick.set_rotation(30)

width = 0.3
x_ = np.arange(5)

def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [6.5, 59.375, 100, 31.25, 3.75]
axs[0, 1].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[0]['color']
             )
#axs[0, 1].set_yscale('log')
axs[0, 1].set_xticks(x_ + width)
axs[0, 1].plot(x__, y1, color='salmon')
axs[0, 1].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[0, 1].set_ylabel('time (sec.)',  fontdict=font1)
axs[0, 1].set_title('(b) DBLP: Efficiency',  fontdict=font1)
#axs[0, 1].tick_params(labelsize=7.5)
for xtick in axs[0, 1].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 1].get_yticklabels():
    ytick.set_rotation(30)


width = 0.3
x_ = np.arange(5)
def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [82.5, 64.5, 105, 31, 8]
axs[0, 2].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[0]['color']
             )
#axs[0, 2].set_yscale('log')
axs[0, 2].set_xticks(x_ + width)
axs[0, 2].plot(x__, y1, color='salmon')
axs[0, 2].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[0, 2].set_ylabel('time (sec.)',  fontdict=font1)
axs[0, 2].set_title('(c) Walmart: Efficiency',  fontdict=font1)
#axs[0, 2].set_ylim([0, math.pow(2,13)])
for xtick in axs[0, 2].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 2].get_yticklabels():
    ytick.set_rotation(30)



width = 0.3
x_ = np.arange(5)

def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [math.pow(2,11), math.pow(2,13), math.pow(2,12), math.pow(2,11.5), math.pow(2,4)]
axs[0, 3].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[0]['color']
             )
axs[0, 3].set_yscale('log')
axs[0, 3].set_xticks(x_ + width)
axs[0, 3].plot(x__, y1, color='salmon')
axs[0, 3].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[0, 3].set_ylabel('time (sec.)',  fontdict=font1)
axs[0, 3].set_title('(d) Songs: Efficiency',  fontdict=font1)
axs[0, 3].set_ylim([0, math.pow(2,13)])
for xtick in axs[0, 3].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 3].get_yticklabels():
    ytick.set_rotation(30)



width = 0.3
x_ = np.arange(5)
def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [math.pow(2,9.5), math.pow(2,9), math.pow(2,13.5), math.pow(2,12.5), math.pow(2,4)]
axs[1, 0].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[0]['color']
             )
axs[1, 0].set_yscale('log')
axs[1, 0].set_xticks(x_ + width)
axs[1, 0].plot(x__, y1, color='salmon')
axs[1, 0].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[1, 0].set_ylabel('time (sec.)',  fontdict=font1)
axs[1, 0].set_title('(e) Songs: Efficiency',  fontdict=font1)
axs[1, 0].set_ylim([0, math.pow(2,13)])
for xtick in axs[1, 0].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(30)


width = 0.3
x_ = np.arange(5)

def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [400, 6400, 1700, 4750, 650]
y2 = [0, 0, 0, 2000, 500]
axs[1, 1].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[5]['color']
             )
axs[1, 1].bar(x_ + width, y2, width,
       color=list(plt.rcParams['axes.prop_cycle'])[2]['color']
             )
#axs[1, 1].set_yscale('log')
axs[1, 1].set_xticks(x_ + width)
axs[1, 1].plot(x__, y1, color='salmon')
axs[1, 1].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[1, 1].set_ylabel('memory (mb.)',  fontdict=font1)
axs[1, 1].set_title('(f) DBLP: Memory Cost',  fontdict=font1)
for xtick in axs[1, 1].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[1, 1].get_yticklabels():
    ytick.set_rotation(30)

width = 0.3
x_ = np.arange(5)

def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [1600, 7500, 1094, 4000+2500, 781+625]
y2 = [0, 0, 0, 2500, 781]
axs[1, 2].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[5]['color']
             )
axs[1, 2].bar(x_ + width, y2, width,
       color=list(plt.rcParams['axes.prop_cycle'])[2]['color']
             )
axs[1, 2].set_xticks(x_ + width)
axs[1, 2].plot(x__, y1, color='salmon')
axs[1, 2].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[1, 2].set_ylabel('memory (mb.)',  fontdict=font1)
axs[1, 2].set_title('(f) Walmart: Memory Cost',  fontdict=font1)
for xtick in axs[1, 2].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[1, 2].get_yticklabels():
    ytick.set_rotation(30)

width = 0.3
x_ = np.arange(5)

def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [2200, 10000, 7000, 10500+2000, 1500+500]
y2 = [0, 0, 0, 10500, 1500]
axs[1, 3].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[5]['color']
             )
axs[1, 3].bar(x_ + width, y2, width,
       color=list(plt.rcParams['axes.prop_cycle'])[2]['color']
             )
axs[1, 3].set_xticks(x_ + width)
axs[1, 3].plot(x__, y1, color='salmon')
axs[1, 3].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[1, 3].set_ylabel('memory (mb.)',  fontdict=font1)
axs[1, 3].set_title('(g) Songs: Memory Cost',  fontdict=font1)
for xtick in axs[1, 3].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[1, 3].get_yticklabels():
    ytick.set_rotation(30)


width = 0.3
x_ = np.arange(5)

def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = [12000, 12800, 8700, 12000, 500+2500]
y2 = [0, 0, 0, 10500, 2500]
axs[2, 0].bar(x_ + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[5]['color']
             )
axs[2, 0].bar(x_ + width, y2, width,
       color=list(plt.rcParams['axes.prop_cycle'])[2]['color']
             )
axs[2, 0].set_xticks(x_ + width)
axs[2, 0].plot(x__, y1, color='salmon')
axs[2, 0].set_xticklabels(['$JedAi$', '$Dedoop$', '$CPU:DM$', '$GPU:DM$', '$CUER$'])
axs[2, 0].set_ylabel('memory (mb.)',  fontdict=font1)
axs[2, 0].set_title('(g) Songs: Memory Cost',  fontdict=font1)
for xtick in axs[2, 0].get_xticklabels():
    xtick.set_rotation(30)

for ytick in axs[2, 0].get_yticklabels():
    ytick.set_rotation(30)

width = 0.3
x_ = np.arange(10)
def func(count):
    return count + 0.3
x__ = list(map(func, x_))
y1 = []
y2 = []
y3 = []
y4 = []
base = 0.1
for i in range(10):
    y1.append(base+ 2.5*i)
    y2.append(base + 2.5 * (i+1))

for i in range(10):
    y3.append(base + 0.1 * i)
    y4.append(18+ 0.1*i)
#y2 = [17.4, 17.61, 17.75, 17.99, 18.4,18.55,19.0,19.25,19.44,19.6]

axs[2, 1].plot(x__, y1, color='black', marker='+', linestyle='--', label = "start points: synchronize")
axs[2, 1].plot(x__, y2, color='black', marker='x', linestyle=':', label = "end points: synchronize")
axs[2, 1].fill_between(x__, y1, y2, color='yellow', alpha=.25, label="elapse time to finish task of synchronize")
axs[2, 1].plot(x__, y3, color='black', marker='.', linestyle='-.', label = "start points: asynchronize ")
axs[2, 1].plot(x__, y4, color='black', marker='x', linestyle=':', label = "end points: asynchronize")
axs[2, 1].fill_between(x__, y3, y4, color='blue', alpha=.25, label="elapse time to finish task of asynchronize")
axs[2, 1].set_ylabel('time (sec.)',  fontdict=font1)
axs[2, 1].set_title('(h)  Songs: synch. vs no asynch.',  fontdict=font1)
axs[2, 1].set_xlim([1,9])
axs[2, 1].set_ylim([0,25])
for xtick in axs[2, 0].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[2, 0].get_yticklabels():
    ytick.set_rotation(30)

fig.legend(loc='upper center', ncol = 4, fontsize=10)

plt.show()