
import numpy as np
from basic_units import cm, inch
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


#plt.style.use("seaborn-deep")
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
#plt.style.use('seaborn-paper')

cms = cm * np.arange(0, 10, 2)
bottom = 0 * cm
width = 0.8 * cm

fig, axs = plt.subplots(3, 4)



x = [10, 20, 30, 40, 50]
without_mqo = [ 147.7863, 191.5051, 285.9294, 290.1875, 436.2881]
with_mqo = [ 154.7838, 179.7277, 250.4791, 260.7774, 357.8257]


axs[0, 2].plot(x, with_mqo,   marker='.', color='mediumslateblue')
axs[0, 2].plot(x, without_mqo, marker=5,color='hotpink')
axs[0, 2].set_ylabel('time (sec.)', fontdict=font1)
axs[0, 2].set_title('(c) IMDB: Varying $||\Sigma||$',  fontdict=font1)
axs[0, 2].set_xlim([10 ,50])
axs[0, 2].tick_params(labelsize=10)
for ytick in axs[0, 2].get_yticklabels():
    ytick.set_rotation(45)

x = [10, 20, 30, 40, 50]
without_mqo = [ 1620458, 2568899, 3449440, 4965037, 6457767]
with_mqo = [471598, 601753, 1538554, 1421046, 1734553]

axs[0, 3].plot(x, with_mqo,   marker='.', label="$MRLsMatch$", color='mediumslateblue')
axs[0, 3].plot(x, without_mqo, marker=5, label="$MRLsMatch_{IH}$", color='hotpink')
axs[0, 3].set_ylabel('maximum workload', fontdict=font1)
axs[0, 3].set_title('(d) Varying $||\Sigma||$',  fontdict=font1)
axs[0, 3].set_xlim([10 ,50])
axs[0, 3].tick_params(labelsize=10)
for ytick in axs[0, 3].get_yticklabels():
    ytick.set_rotation(45)





population_by_continent = {
    '$data partition$': list(([43.401,44.246,45.413,46.342,48.69])),
    '$local match$': list(([313.610,270.703,224.156,199.840,202.533])),
    '$closure$': list(([30.272,30.754,26.275,30.274,25.584])),
}

year = [ 16, 20, 24, 28, 32]
axs[2, 0].stackplot(year, population_by_continent.values())
axs[2, 0].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 0].set_xlabel('$n$', fontdict=font1)
axs[2, 0].set_title('(i) IMDB: Cost Breakdown',  fontdict=font1)
axs[2, 0].set_xlim([16, 32])
axs[2, 0].set_ylim([0, 520])
axs[2, 0].tick_params(labelsize=10)
for ytick in axs[2, 0].get_yticklabels():
    ytick.set_rotation(45)


year = [ 2, 4, 6, 8, 10]
population_by_continent = {
    '$data partition$': list([0.050243,0.076026,0.093415,0.0978,0.095583]),
    '$local match$': list([34.7923,19.1718,18.1284,9.33846,9.19014]),
    '$closure$': list([0.150651,0.262626,0.260611,0.253624,0.255128]),
}
axs[2, 1].stackplot(year, population_by_continent.values())
axs[2, 1].set_xlabel('$n$', fontdict=font1)
axs[2, 1].set_ylabel('Time (sec.)', fontdict=font1)
#axs[2, 1].set_title('(i) Varying $n$',  fontsize=10)
axs[2, 1].set_title('(j) ACM-DBLP: Cost Breakdown', fontdict=font1)
axs[2, 1].set_xlim([2, 10])
axs[2, 1].tick_params(labelsize=10)
for ytick in axs[2, 1].get_yticklabels():
    ytick.set_rotation(45)

plt.legend(loc='lower center') # 标签位置


year = [ 16, 20, 24, 28, 32]
population_by_continent = {
    '$data partition$': list([5.891,10.475,10.685,10.298,10.341]),
    '$local match$': list([5102.919, 2886.537, 2481.410, 3146, 796.029]),
    '$closure$': list([11.868,12.066,11.764,11.923,13.168]),
}
axs[2, 2].stackplot(year, population_by_continent.values(),
                    labels=population_by_continent.keys())
axs[2, 2].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 2].set_xlabel('$n$', fontdict=font1)
axs[2, 2].set_title('(k) Movie: Cost Breakdown',  fontdict=font1)
axs[2, 2].set_xlim([16, 32])
axs[2, 2].tick_params(labelsize=10)
for ytick in axs[2, 2].get_yticklabels():
    ytick.set_rotation(45)

year = [ 16, 20, 24, 28, 32]
population_by_continent = {
    '$data partition$': list([10.059,19.822,40.883,58.335,79.311] ),
    '$local match$': list([79.857,195.872,807.512,1974.192,3086.702]),
    '$closure$': list([22.514,26.265,53.593,112.350,147.661]),
}
axs[2, 3].stackplot(year, population_by_continent.values(),
                   labels=population_by_continent.keys())
axs[2, 3].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 3].set_xlabel('$n$', fontdict=font1)
axs[2, 3].set_title('(k) Movie: Cost Breakdown',  fontdict=font1)
axs[2, 3].set_xlim([16, 32])
axs[2, 3].tick_params(labelsize=10)
for ytick in axs[2, 3].get_yticklabels():
    ytick.set_rotation(45)

x = [16, 20, 24, 28, 32]
without_mqo = [
    247338  ,216860 , 192864, 178389, 176848
]
with_mqo = [
    121925,  110471, 106886, 100463, 109804
]

axs[1, 0].plot(x, with_mqo,   marker='.', color='mediumslateblue')
axs[1, 0].plot(x, without_mqo, marker=5, color='hotpink')
axs[1, 0].set_ylabel('Maximum workload', fontdict=font1)
axs[1, 0].set_xlabel('$n$', fontdict=font1)
axs[1, 0].set_xlim([16, 32])
axs[1, 0].set_title('(e) Varying $n$',  fontdict=font1)
axs[1, 0].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(45)

x = [16, 20, 24, 28, 32]
without_mqo = [
 #   (132260+229873+46914.5)/1000, (105715 + 217281 + 54466.9)/1000, (104208+297850+49574.1)/1000, (83953.2+340841+48299.1)/1000, (64624+250573+48079.7)/1000
(132260+229873+46914.5)/1000, (105715 + 217281 + 54466.9)/1000, (104208+ 229873 +49574.1)/1000, (83953.2+ 229873 +48299.1)/1000, (64624+ 229873 +48079.7)/1000
]

#(132260 + 229873 + 46914.5) / 1000, (142593 + 282142 + 52006.7) / 1000, (104208 + 297850 + 49574.1) / 1000, ( 83953.2 + 340841 + 48299.1) / 1000, (64624 + 250573 + 48079.7) / 1000
with_mqo = [
    (43.401+313.610+30.272), (44.246+270.703+30.754), (45.413+224.156+26.275), (46.342+199.840+30.274), ( 48.69+ 202.533 + 25.584)
]
axs[1, 1].plot(x, with_mqo,   marker='.', color='mediumslateblue')
axs[1, 1].plot(x, without_mqo, marker=5,color='hotpink')
axs[1, 1].set_ylabel('Time (sec.)', fontdict=font1)
axs[1, 1].set_xlabel('$n$', fontdict=font1)
axs[1, 1].set_xlim([16,32])
axs[1, 1].set_title('(f) Varying $n$',  fontdict=font1)
axs[1, 1].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(45)




x = [16, 20, 24, 28, 32]
without_mqo = [
     492.7256, 40746664, 55104556, 76178808, 98401750

]
with_mqo = [
    462.2421, 9548420, 23386408, 24410876, 26519046
]
axs[1, 2].plot(x, with_mqo,  linestyle='--', color='mediumslateblue')
axs[1, 2].plot(x, without_mqo, marker=5, color='pink')
axs[1, 2].set_ylabel('Shipment', font1)
axs[1, 2].set_xlabel('$|\Sigma|$', font1)
axs[1, 2].set_title('(g) IMDB: Varying $|\Sigma|$',  fontdict=font1)
axs[1, 2].tick_params(labelsize=10)
labels = ['IMDB', 'ACM-DBLP', 'Movie' ]
for ytick in axs[1, 2].get_yticklabels():
    ytick.set_rotation(45)



erblox = [0.9134, 0.6561, 0.3732]
deep_matcher = [0.9423, 0.9168, 0.9927]
dedoop = [0.53, 0.18763, 0.64937]
sc = [0.674, 0.8194, 0.9]
MRLsMatch  = [0.9708, 0.95733, 0.98761]
x = np.arange(len(labels))  # the label locations
width = 0.5 # the width of the bars


rects1 = axs[0, 0].bar(x - width/2, erblox, width/4, label='ERBlox', color = 'steelblue')
rects2 = axs[0, 0].bar(x - width/4, deep_matcher, width/4, label='DM', color = 'turquoise')
rects3 = axs[0, 0].bar(x , dedoop, width/4, label='Dedoop', color = 'khaki')
rects5 = axs[0, 0].bar(x + width/4, sc, width/4, label='SparklyClean', color = 'salmon')
rects6 = axs[0, 0].bar(x + width/2, MRLsMatch, width/4, label='$MRLsMatch$', color = 'mediumslateblue')

axs[0, 0].set_ylabel('F1-score', fontdict=font1)
axs[0, 0].set_xticks(x)
axs[0, 0].set_title('(a) Accuracy',  fontdict=font1)
axs[0, 0].set_ylim([0.1, 1])
axs[0, 0].set_xticklabels(labels)
axs[0, 0].tick_params(labelsize=10)

for xtick in axs[0, 0].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 0].get_yticklabels():
    ytick.set_rotation(45)



"""
dedoop = [( 758.506), 17.9, 3719]
sc = [534.301, 12.286, 2485.006667]
MRLsMatch  = [482.8056, 2.762, 0.98761]
MRLsMatch_IH  = [0.9708, 0.95733, 0.98761]
MRLsMatch_WH  = [0.9708, 0.95733, 0.98761]
"""
def auto_text(rects, ax):
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')

dedoop = list (map(custom_log, [ 758.506, 17.9, 3719]))
sc = list(map(custom_log, [534.301, 12.286, 2485.006667]))
MRLsMatch  = list(map(custom_log, [ (43.401+313.610+30.272), (0.07789+3.162148+0.235625), (796.029+13.168+ 10.341)]))
MRLsMatch_IH  = list(map(custom_log, [(132260+229873+46914.5)/1000, (0.07786+3.147147+0.232840), (796.029+13.168+ 10.341)]))
MRLsMatch_WH  = list(map(custom_log, [0.9708, 0.95733, 0.98761]))
x = np.arange(len(labels))  # the label locations
width = 0.5 # the width of the bars

mscale.register_scale(MercatorLatitudeScale)
axs[0, 1].set_yscale('mercator')
rects3 = axs[0, 1].bar(x  - width/2,(dedoop), width/4,  color = 'khaki')
rects5 = axs[0, 1].bar(x - width/4, sc, width/4,  color = 'salmon')
rects6 = axs[0, 1].bar(x, MRLsMatch, width/4,  color = 'mediumslateblue')
rects7 = axs[0, 1].bar(x + width/4, MRLsMatch_IH, width/4, label='$MRLsMatch_{IH}$', color = 'hotpink')
rects8 = axs[0, 1].bar(x + width/2, MRLsMatch, width/4, label='$MRLsMatch_{WH}$', color = 'deepskyblue')
axs[0, 1].set_ylabel('Time (sec.)', fontdict=font1)
axs[0, 1].set_xticks(x)
axs[0, 1].set_title('(b) Efficiency',  fontdict=font1)
axs[0, 1].set_xticklabels(labels)
axs[0, 1].set_xlim([-0.25 - 0.125, 2.25 + 0.125])
axs[0, 1].set_ylim([0, custom_log(5200)])
axs[0, 1].tick_params(labelsize=10)
for xtick in axs[0, 1].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 1].get_yticklabels():
    ytick.set_rotation(45)



fig.legend(loc='upper center', ncol = 6, fontsize=10)

plt.show()
