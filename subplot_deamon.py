
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
plt.style.use('seaborn-muted')
#plt.style.use('classic')
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-pastel')
#plt.style.use('bmh')
#plt.style.use('ggplot')
#plt.style.use('seaborn-paper')


fig, axs = plt.subplots(4, 4)


x = [10, 20, 30, 40, 50]
without_mqo = [ 147.7863, 191.5051, 285.9294, 290.1875, 436.2881]
with_mqo = [ 154.7838, 179.7277, 250.4791, 260.7774, 357.8257]


axs[0, 3].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5, label="$MRLsMatch$")
axs[0, 3].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8, label="$MRLsMatch_{IH}$")
axs[0, 3].set_ylabel('Time (sec.)', fontdict=font1)
axs[0, 3].set_title('(d) IMDB: Varying $||\Sigma||$',  fontdict=font1)
axs[0, 3].set_xlim([10 ,50])
axs[0, 3].tick_params(labelsize=10)
for ytick in axs[0, 3].get_yticklabels():
    ytick.set_rotation(30)

x = [1, 10, 20, 30, 40, 50]
without_mqo = [64988, 1620458, 2568899, 3449440, 4965037, 6457767]
with_mqo = [64988, 471598, 601753, 1538554, 1421046, 1734553]

axs[1, 2].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5)
axs[1, 2].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8)
axs[1, 2].set_ylabel('Data Shipment', fontdict=font1)
axs[1, 2].set_title('(g) IMDB: Varying $||\Sigma||$',  fontdict=font1)
axs[1, 2].set_xlim([1 ,50])
axs[1, 2].tick_params(labelsize=10)
for ytick in axs[1, 2].get_yticklabels():
    ytick.set_rotation(30)



"""
x = [10, 20, 30, 40, 50]
without_mqo = [ 147.7863, 191.5051, 285.9294, 290.1875, 436.2881]
with_mqo = [ 154.7838, 179.7277, 250.4791, 260.7774, 357.8257]


axs[3, 0].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5, label="$MRLsMatch$")
axs[3, 0].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8, label="$MRLsMatch_{IH}$")
axs[3, 0].set_ylabel('Maximum Workload', fontdict=font1)
axs[3, 0].set_title('(c) IMDB: Varying $||\Sigma||$',  fontdict=font1)
axs[3, 0].set_xlim([10 ,50])
axs[3, 0].tick_params(labelsize=10)
for ytick in axs[0, 2].get_yticklabels():
    ytick.set_rotation(30)
"""

population_by_continent = {
    '$data partition$': list(([27.171,28.128,28.803,29.507,30.747]        )),
    '$local match$': list(([205.378,186.258,180.076,164.938,158.376 ])),
    '$closure$': list(([2.611,2.708,2.667,2.689,2.686 ])),
}

colors = ['navy', 'cornflowerblue', 'deepskyblue']
year = [ 16, 20, 24, 28, 32]
axs[2, 3].stackplot(year, population_by_continent.values(),colors=colors)
axs[2, 3].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 3].set_title('(l) IMDB: Cost Breakdown',  fontdict=font1)
axs[2, 3].set_xlim([16, 32])
axs[2, 3].set_ylim([0, 520])
axs[2, 3].tick_params(labelsize=10)
for ytick in axs[2, 3].get_yticklabels():
    ytick.set_rotation(30)


year = [ 2, 4, 6, 8, 10]
population_by_continent = {
    '$data partition$': list([0.050243,0.076026,0.093415,0.0978,0.095583]),
    '$local match$': list([34.7923,19.1718,18.1284,9.33846,9.19014]),
    '$closure$': list([0.150651,0.262626,0.260611,0.253624,0.255128]),
}
axs[3, 0].stackplot(year, population_by_continent.values(), colors=colors)
axs[3, 0].set_ylabel('Time (sec.)', fontdict=font1)
axs[3, 0].set_title('(m) ACM-DBLP: Cost Breakdown', fontdict=font1)
axs[3, 0].set_xlim([2, 10])
axs[3, 0].tick_params(labelsize=10)
for tick in axs[3, 0].get_yticklabels():
    ytick.set_rotation(30)

plt.legend(loc='lower center') # 标签位置


year = [ 16, 20, 24, 28,32]
population_by_continent = {
    '$data partition$': list([5.891,10.475,10.685,10.298,10.341]),
    '$local match$': list([5102.919, 2886.537, 2481.410, 3146, 796.029]),
    '$closure$': list([11.868,12.066,11.764,11.923,13.168]),
}
axs[3, 1].stackplot(year, population_by_continent.values(), colors=colors)
axs[3, 1].set_ylabel('Time (sec.)', fontdict=font1)
#axs32,12].set_xlabel('$n$', fontdict=font1)
axs[3, 1].set_title('(n) Movie: Cost Breakdown',  fontdict=font1)
axs[3, 1].set_xlim([16, 32])
axs[3, 1].tick_params(labelsize=10)
for ytick in axs[2, 3].get_yticklabels():
    ytick.set_rotation(30)

year = [0.125, 0.25, 0.5, 0.75, 1]
population_by_continent = {
    '$data partition$': list([10.059,19.822,40.883,58.335,79.311] ),
    '$local match$': list([79.857,195.872,807.512,1974.192,3086.702]),
    '$closure$': list([22.514,26.265,53.593,112.350,147.661]),
}
axs[2, 2].stackplot(year, population_by_continent.values(),
              labels=population_by_continent.keys(), colors=colors)
axs[2, 2].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 2].set_title('(k) Synthetic: Varying Scale Factor',  fontdict=font1)
axs[2, 2].set_xlim([0.125, 1])
axs[2, 2].tick_params(labelsize=10)
for ytick in axs[3, 0].get_yticklabels():
    ytick.set_rotation(30)

x = [16, 20, 24, 28, 32]
without_mqo = [
    247338  ,216860 , 192864, 178389, 176848
]
with_mqo = [
    121925,  110471, 106886,  109804, 100462
]

axs[1, 3].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5)
axs[1, 3].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8)
axs[1, 3].set_ylabel('Maximum Workload', fontdict=font1)
axs[1, 3].set_xlim([16, 32])
axs[1, 3].set_title('(h) IMDB: Varying $n$',  fontdict=font1)
axs[1, 3].tick_params(labelsize=10)
for ytick in axs[1, 3].get_yticklabels():
    ytick.set_rotation(30)

x = [16, 20, 24, 28, 32]
without_mqo = [
 #   (132260+229873+46914.5)/1000, (105715 + 217281 + 54466.9)/1000, (104208+297850+49574.1)/1000, (83953.2+340841+48299.1)/1000, (64624+250573+48079.7)/1000
(132260+229873+46914.5)/1000, (105715 + 217281 + 54466.9)/1000, (104208+ 229873 +49574.1)/1000, (83953.2+ 229873 +48299.1)/1000, (64624+ 229873 +48079.7)/1000
]
#(132260 + 229873 + 46914.5) / 1000, (142593 + 282142 + 52006.7) / 1000, (104208 + 297850 + 49574.1) / 1000, ( 83953.2 + 340841 + 48299.1) / 1000, (64624 + 250573 + 48079.7) / 1000
with_mqo = [
    (43.401+313.610+30.272), (44.246+270.703+30.754), (45.413+224.156+26.275), (46.342+199.840+30.274), ( 48.69+ 202.533 + 25.584)
]
print(without_mqo)
print(with_mqo)
#axs[1, 1].plot(x, with_mqo,   marker='.', color='mediumslateblue')
#axs[1, 1].plot(x, without_mqo, marker=5,color='hotpink')
axs[2, 1].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5)
axs[2, 1].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8)
axs[2, 1].set_ylabel('Time (sec.)', fontdict=font1)
axs[2, 1].set_xlim([16,32])
axs[2, 1].set_title('(j) IMDB: Varying $n$',  fontdict=font1)
axs[2, 1].tick_params(labelsize=10)
for ytick in axs[2, 1].get_yticklabels():
    ytick.set_rotation(30)





sigma = [ 2, 4, 6, 8, 10]
population_by_continent = {
    '$data partition$': list([9.669,10.952,17.586,31.56,30.778] ),
    '$local match$': list([27.222, 48.980, 64.674, 82.878, 141.146]),
    '$closure$': list([0.810, 0.506, 0.182, 0.409, 0.070]),
}
axs[1, 0].stackplot(sigma, population_by_continent.values(), colors=colors)
axs[1, 0].set_ylabel('Time (sec.)', fontdict=font1)
axs[1, 0].set_title('(e) IMDB: Varying $|\Sigma|$',  fontdict=font1)
axs[1, 0].set_xlim([2, 10])
axs[1, 0].tick_params(labelsize=10)
for ytick in axs[1, 0].get_yticklabels():
    ytick.set_rotation(30)


labels = ['IMDB', 'ACM-DBLP', 'Movie', 'Product', 'Songs']
erblox = [0.9134, 0.6561, 0.3732, 0.0411, 0.2521]
#deep_matcher = [0.9423, 0.9168, 0.9927, 0.5811, 0.9989]
deep_matcher = [0.9423, 0.9168, 0.9927, 0.2, 0.9989]
dedoop = [0.53, 0.18763, 0.64937, 2*(221/1301)*(221/679)/((221/1301)+(221/679)), 2*(28433/31218)*(28433/56499)/((28433/31218)+(28433/56499))]
sc = [0.674, 0.8194, 0.9, 0.3, \
      (2 * ((1717+1937+1778+1833+1784)/(1889+2024+2097+2071+1949)) * ((1717+1937+1778+1833+1784)/106499) ) \
      / ( ((1717+1937+1778+1833+1784)/(1889+2024+2097+2071+1949))+((1717+1937+1778+1833+1784)/106499) ) ]
MRLsMatch  = [0.9708, 0.95733, 0.98761, 0.47, 0.97598]
x = np.arange(len(labels))  # the label locations
width = 0.65 # the width of the bars

"""
rects1 = axs[0, 0].bar(x - width/2, erblox
, width/4, label='ERBlox', color = 'steelblue', hatch=patterns[5], edgecolor='black')
rects2 = axs[0, 0].bar(x - width/4, deep_matcher, width/4, label='DM', color = 'turquoise', hatch=patterns[8])
rects3 = axs[0, 0].bar(x , dedoop, width/4, label='Dedoop', color = 'khaki', hatch=patterns[0])
rects5 = axs[0, 0].bar(x + width/4, sc, width/4, label='SparklyClean', color = 'salmon', hatch=patterns[1] )
rects6 = axs[0, 0].bar(x + width/2, MRLsMatch, width/4, label='$MRLsMatch$', color = 'mediumslateblue')
"""

rects1 = axs[0, 0].bar(x - width/2, erblox, width/4, label='ERBlox', color = 'grey', hatch=patterns[5],edgecolor='black', linewidth = 0.1)
rects2 = axs[0, 0].bar(x - width/4, deep_matcher, width/4, label='DM', color = 'white', hatch=patterns[8], edgecolor='black', linewidth = 0.1)
rects3 = axs[0, 0].bar(x , dedoop, width/4, label='Dedoop', color = 'whitesmoke', hatch=patterns[0],edgecolor='black', linewidth = 0.1)
rects5 = axs[0, 0].bar(x + width/4, sc, width/4, label='Dis-Dup', color = 'black', hatch=patterns[1], edgecolor='black', linewidth = 0.1)
rects6 = axs[0, 0].bar(x + width/2, MRLsMatch, width/4, label='$MRLsMatch$', color = 'darkgrey', edgecolor='black', linewidth = 0.1)

axs[0, 0].set_ylabel('F1-score', fontdict=font1)
axs[0, 0].set_xticks(x)
axs[0, 0].set_title('(a) Accuracy',  fontdict=font1)
axs[0, 0].set_ylim([0.0, 1])
axs[0, 0].set_xticklabels(labels)
axs[0, 0].tick_params(labelsize=7.5)
#axs[0, 0].tick_params(labelsize=10)

for xtick in axs[0, 0].get_xticklabels():
    xtick.set_rotation(30)



"""
dedoop = [( 758.506), 17.9, 3719]
sc = [534.301, 12.286, 2485.006667]
MRLsMatch  = [482.8056, 2.762, 0.98761]
MRLsMatch_IH  = [0.9708, 0.95733, 0.98761]
MRLsMatch_WH  = [0.9708, 0.95733, 0.98761]
"""
#labels = ['IMDB', 'ACM-DBLP', 'Movie' ,'Amazon-Google Products', 'Songs']
def auto_text(rects, ax):
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')

#x = [16, 20, 24, 28, 32, 33, 334]
#dedoop = list (map(custom_log, [ 758.506, 17.9, 1290, 0, 0]))
dedoop = list (map(custom_log, [ (344+25+25+25), 17.9, 844, 26, 45]))
#is going to be filled
#sc = list(map(custom_log, [534.301, 12.286, 2485.006667, 759, 45.672]))
sc = list(map(custom_log, [534.301, 12.286, 2485.006667, 47.75, 45.672]))
#MRLsMatch  = list(map(custom_log, [ (43.401+313.610+30.272), (0.07789+3.162148+0.235625), (796.029+13.168+ 10.341), 1, 1]))
MRLsMatch  = list(map(custom_log, [ (43.401+313.610+30.272), (0.07789+3.162148+0.235625), (271), 6.43, (3.68)]))
MRLsMatch_IH  = list(map(custom_log, [(132260+229873+46914.5)/1000, (0.07786+3.147147+0.232840), (271), 13, 3.68]))
#MRLsMatch_WH  = list(map(custom_log, [0.9708, 0.95733, 0.98761]))
x = np.arange(len(labels))  # the label locations
width = 0.5 # the width of the bars

mscale.register_scale(MercatorLatitudeScale)
axs[0, 2].set_yscale('mercator')
rects3 = axs[0, 2].bar(x  - width/2,(dedoop), width/4,   color = 'whitesmoke', hatch=patterns[0], edgecolor='black', linewidth = 0.1)
rects5 = axs[0, 2].bar(x - width/4, sc, width/4,  color = 'black', hatch=patterns[1], edgecolor='black', linewidth = 0.1)
rects6 = axs[0, 2].bar(x, MRLsMatch, width/4, color = 'darkgrey', edgecolor='black', linewidth = 0.1  )
rects7 = axs[0, 2].bar(x + width/4, MRLsMatch_IH, width/4, label='$MRLsMatch_{IH}$', color = 'silver', hatch=patterns[9], edgecolor='black', linewidth=0.1)
axs[0, 2].set_ylabel('Time (sec.)', fontdict=font1)
axs[0, 2].set_xticks(x)

axs[0, 2].set_title('(c) Efficiency',  fontdict=font1)
axs[0, 2].set_xticklabels(labels, size=1)
axs[0, 2].set_ylim([0, custom_log(5200)])
axs[0, 2].tick_params(labelsize=7.5)
for xtick in axs[0, 2].get_xticklabels():
    xtick.set_rotation(30)
for ytick in axs[0, 2].get_yticklabels():
    ytick.set_rotation(30)
x = [5, 10, 15, 20, 25]
#without_mqo = [ 147.7863, 191.5051, 285.9294, 290.1875, 436.2881]
with_mqo = [ 0.945, 0.94312, 0.94312, 0.94255, 0.94306]
dedoop = [ (1778/7447), (3564/9570), (5325/11729), (7108/13898), (8876/16014)]
sc = [ (1778/7447), (3564/9570), (5325/11729), (7108/13898), (8876/16014)]


#axs[2, 3].plot(x, with_mqo,   marker='o', color='mediumslateblue')
#axs[2, 3].plot(x, dedoop,   marker='+', color='khaki')
axs[0, 1].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5)
axs[0, 1].plot(x, dedoop, marker='+',color='gold', markersize = 11, label='Dedoop')
axs[0, 1].set_ylabel('Recall', fontdict=font1)
axs[0, 1].set_title('(b) Movie: Varying $dup$',  fontdict=font1)
axs[0, 1].set_xlim([5, 25])
axs[0, 1].set_ylim([0.2, 1])
axs[0, 1].tick_params(labelsize=10)
for ytick in axs[3, 3].get_yticklabels():
    ytick.set_rotation(30)



x = [10, 20, 30, 40, 50]
without_mqo = [
12116,
35153,
68588,
108520,
1564922

]

#(132260 + 229873 + 46914.5) / 1000, (142593 + 282142 + 52006.7) / 1000, (104208 + 297850 + 49574.1) / 1000, ( 83953.2 + 340841 + 48299.1) / 1000, (64624 + 250573 + 48079.7) / 1000
with_mqo = [
    14596,14706,14706,42804,935001
]
axs[1, 1].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5)
axs[1, 1].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8)
axs[1, 1].set_ylabel('Maximum Workload', fontdict=font1)
axs[1, 1].set_xlim([10,50])
axs[1, 1].set_title('(f) IMDB: Varying $||\Sigma||$',  fontdict=font1)
axs[1, 1].tick_params(labelsize=10)
for ytick in axs[1, 1].get_yticklabels():
    ytick.set_rotation(30)

x = [16, 20, 24, 28, 32]
without_mqo = [
1819664,
1884652,
1866229,
1728648,
2079616

]

#(132260 + 229873 + 46914.5) / 1000, (142593 + 282142 + 52006.7) / 1000, (104208 + 297850 + 49574.1) / 1000, ( 83953.2 + 340841 + 48299.1) / 1000, (64624 + 250573 + 48079.7) / 1000
with_mqo = [
714868,
779856,
844844,
782592,
974820

]
axs[2, 0].plot(x, with_mqo,   marker='s', color='mediumslateblue', markersize=5)
axs[2, 0].plot(x, without_mqo, marker='+', linestyle = '--',color='hotpink', markersize = 8)
axs[2, 0].set_ylabel('Data Shipment', fontdict=font1)
axs[2, 0].set_xlim([16,32])
axs[2, 0].set_title('(i) IMDB: Varying $n$',  fontdict=font1)
axs[2, 0].tick_params(labelsize=10)
for ytick in axs[2, 0].get_yticklabels():
    ytick.set_rotation(30)


x = np.arange(4)
x_ = x
width = 0.3
y1 = [7.42669, 32.9, 1.4511185, 2.4576912]
def func(count):
    return count + 0.3
x_ = map(func, x_)
x_ = list(x_)
#ax3.bar(x, y1, width)
axs[3, 2].bar(x + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[5]['color'],
             )
axs[3, 2].set_xticks(x + width)
axs[3, 2].plot(x_, y1, color='salmon')
axs[3, 2].set_xticklabels(['Dedoop', 'Dis-Dup', '$MRLsMatch$', '$MRLsMatch_{IH}$'])
axs[3, 2].set_ylabel('Space Rquirements (Gb.)',  fontdict=font1)
axs[3, 2].set_title('(o) Movie: Space Requirements',  fontdict=font1)
axs[3, 2].set_ylim([1, 33])
axs[3, 2].tick_params(labelsize=7.5)
for xtick in axs[3, 2].get_xticklabels():
    xtick.set_rotation(30)


x = np.arange(3)
x_ = x
width = 0.3
y1 = [37.44, 37.52, 42.7395]
def func(count):
    return count + 0.3
x_ = map(func, x_)
x_ = list(x_)
#ax3.bar(x, y1, width)
axs[3, 3].bar(x + width, y1, width,
       color=list(plt.rcParams['axes.prop_cycle'])[5]['color'],
             )
axs[3, 3].set_xticks(x + width)
axs[3, 3].plot(x_, y1, color='salmon')
axs[3, 3].set_xticklabels(['$MRLsMatch$', '$MRLsMatch_{IH}$', '$MRLsMatch_{WH}$'])
axs[3, 3].set_ylabel('Space Requirements (Gb.)',  fontdict=font1)
axs[3, 3].set_title('(p) IMDB: Space Requirements',  fontdict=font1)
axs[3, 3].set_ylim([32, 45])
axs[3, 3].tick_params(labelsize=7.5)
for xtick in axs[3, 3].get_xticklabels():
    xtick.set_rotation(30)

fig.legend(loc='upper center', ncol = 6, fontsize=10)

plt.show()
