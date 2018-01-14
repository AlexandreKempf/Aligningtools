
#### BASED ON A REFERENCE FRAME, PLOT THE IOI MAPS WITH THE RESPECTIVE COORDINATED IN IOI.NPY
####



import numpy as np
from PIL import Image
import glob
from scipy.ndimage import zoom, rotate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.580, 0.300, 0.100])

# dir = "/home/alexandre/data/dev/localisation_fields/thalamus/"
# dir = "/home/alexandre/data/dev/localisation_fields/cortex/"

paths,x,y,w,h,a = np.load(dir +"save/ioi.npy")

paths = paths.astype(str)
x = x.astype(float)
y = y.astype(float)
w = w.astype(float)
h = h.astype(float)
a = a.astype(float)

x -= x[0]
y -= y[0]
x *= 494 / w[0]
y *= 659 / h[0]
w *= 494 / w[0]
h *= 659 / h[0]

vessels = mpimg.imread(paths[0])
if vessels.ndim == 3:
    vessels = vessels.transpose((1,0,2))
else:
    vessels = vessels.transpose((1,0))



fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.imshow(rotate(vessels,90), origin='lower', extent=[x[0], x[0]+h[0], y[0]-w[0], y[0]], alpha = 1)

rects = []
transforms = []
for i in np.arange(len(x)):
    rects.append(patches.Rectangle((x[i],y[i]), w[i], h[i], color="black", fill=False, lw=2))
    transforms.append(mpl.transforms.Affine2D().rotate_deg_around(x[i], y[i], a[i]) + ax.transData)
    rects[i].set_transform(transforms[i])
    ax.add_patch(rects[i])

plt.show()
