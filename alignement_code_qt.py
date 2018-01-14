### ALIGNEMENT DE L'ENSEMBLE DES CHAMPS ET DES IOI



import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PIL import Image
import glob
from scipy.ndimage import zoom


def rotate_xy(xi,yi,xc,yc,a):
    """
    coordinate of on old (xi,yi) points after a rotation of "a" degree around (xc,yc)
    """
    a *= np.pi/180
    x = (xi-xc)*np.cos(a) - (yi-yc)*np.sin(a) + xc
    y = (xi-xc)*np.sin(a) + (yi-yc)*np.cos(a) + yc
    return x, y


dir = "/home/alexandre/data/dev/localisation_fields/"
app = QtGui.QApplication([])
wind = pg.GraphicsWindow(size=(1000,800), border=True)
wind.setWindowTitle('pyqtgraph example: ROI Examples')
w1 = wind.addLayout(row=0, col=0)
v1 = w1.addViewBox(row=0, col=0, lockAspect=True)

def load(dirpath):
    paths,x,y,w,h,a = np.load(dirpath)
    x = x.astype(float)
    y = y.astype(float)
    w = w.astype(float)
    h = h.astype(float)
    a = a.astype(float)
    return paths, x, y, w, h, a

def merge(paths, x, y, w, h, a, paths1, x1, y1, w1, h1, a1):
    paths = np.concatenate([paths,paths1])
    x = np.concatenate([x,x1])
    y = np.concatenate([y,y1])
    w = np.concatenate([w,w1])
    h = np.concatenate([h,h1])
    a = np.concatenate([a,a1])
    return paths, x, y, w, h, a

def small_transf(x, y, a, xc, yc, ac, iref):
    """
    x,y is a coord of images ready to be change,
    xc,yc is the coordinates of the reference frame (ioi)
    iref is the index of the image in x that is similar to xc and yc
    """
    x += xc - x[iref]
    y += yc - y[iref]
    x, y = rotate_xy(x, y, x[iref], y[iref], ac - a[iref])
    a += ac - a[iref]
    return x, y, a

paths,x,y,w,h,a = load(dir + "save/ioi.npy")

dirnames = [dir + "save/c1m2_c1m2.npy",
            dir +"save/c4m1_c4m1.npy",
            dir +"save/c4m2_c4m2.npy",
            dir +"save/c4m3_c4m3.npy",
            dir +"save/c5m2_c5m2.npy",
            dir +"save/c6m1_c6m1.npy",
            dir +"save/c6m2_c6m2.npy"]

for i in np.arange(6):
    pathsc,xc,yc,wc,hc,ac = load(dirnames[i])
    xc, yc, ac = small_transf(xc, yc, ac, x[i], y[i], a[i], 0)
    paths, x, y, w, h, a = merge(paths, x, y, w, h, a, pathsc, xc, yc, wc, hc, ac)

imgs_src = [np.array(Image.open(dir +path), dtype=np.float32) for path in paths]
vesselfile = np.array([str(dir +path).find("vessels") for path in paths]) != -1 # where is the vessel map
ioifile = np.array([str(dir +path).find("4khz") for path in paths]) != -1 # where is the vessel map
zoomfile = vesselfile | ioifile
for i in np.arange(len(zoomfile)):
    if zoomfile[i]:
        if imgs_src[i].ndim == 3:
            imgs_src[i] = zoom(imgs_src[i][:,:],(2.6,2.6,1))
        else:
            imgs_src[i] = zoom(imgs_src[i][:,:],(2.6,2.6))

alphas = np.zeros(len(imgs_src))+1
alphas[0] = 1
imgs = [pg.ImageItem(imgs_src[i], opacity=alphas[i]) for i in np.arange(len(x))]
rects = [pg.RectROI([x[i], y[i]], [w[i], h[i]], angle = a[i], pen=(0,9)) for i in np.arange(len(x))]
[v1.addItem(rect) for rect in rects]

for i in np.arange(len(imgs)):
    imgs[i].setParentItem(rects[i])


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
