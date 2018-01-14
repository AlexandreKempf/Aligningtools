### PROGRAM TO ALIGN THE IMG TOGETHER

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PIL import Image
import glob
from scipy.ndimage import zoom

# import of vessels + 2p maps
dir = "/home/alexandre/data/dev/localisation_fields/thalamus/"
NAME_FILE_SAVE = dir+"save/souris5.npy"
paths = np.delete(np.sort(glob.glob(dir + "*souris5*")),-2)[::-1] # list all the fiels except the tonotopy
imgs_src = [np.array(Image.open(path), dtype=np.float32) for path in paths]
# Pour le 10x c'est 2.6 par rapport a la carte des vaisseaux
# imgs_src[0] = zoom(imgs_src[0][:,:,0],2.6)  # attention certaines images ont 3 channels
# Pour le 20x c'est a peu pret par rapport a la carte des vaisseaux
imgs_src[0] = zoom(imgs_src[0][:,:,0],8.5)  # attention certaines images ont 3 channels

# # import of ioi maps
# dir = "/home/alexandre/data/dev/localisation_fields/thalamus/"
# NAME_FILE_SAVE = dir + "save/ioi.npy"
# paths = np.sort(glob.glob(dir+"*4kHz*"))
# path = paths[0]
# imgs_src = [zoom(np.array(Image.open(path), dtype=np.float32),(2.6,2.6,1)) for path in paths]

app = QtGui.QApplication([])
w = pg.GraphicsWindow(size=(1000,800), border=True)
w.setWindowTitle('pyqtgraph example: ROI Examples')
w1 = w.addLayout(row=0, col=0)
v1 = w1.addViewBox(row=0, col=0, lockAspect=True)

for i in np.arange(len(imgs_src)):
    imgs_src[i] -= np.percentile(imgs_src[i],10)
    imgs_src[i] /= np.percentile(imgs_src[i],90)/255
    imgs_src[i][imgs_src[i]<0] = 0
    imgs_src[i][imgs_src[i]>255] = 255

imgs = [pg.ImageItem(img_src, opacity=0.5) for img_src in imgs_src]
ratios = [float(img_src.shape[0])/float(img_src.shape[1]) for img_src in imgs_src]
alphas = [0.5 for img_src in imgs_src]
rects = [pg.RectROI([np.random.random(1)[0], 0], [img_src.shape[0], img_src.shape[1]], pen=(0,9)) for img_src in imgs_src]
[rect.addRotateHandle([1,0], [0.5, 0.5]) for rect in rects]
[v1.addItem(rect) for rect in rects]




def update(rect):
    global index
    x, y, w, h = rect.pos().x(), rect.pos().y(), rect.size().x(),rect.size().y()
    xsize = np.array([irect.size().x() for irect in rects])
    ysize = np.array([irect.size().y() for irect in rects])
    xpos = np.array([irect.pos().x() for irect in rects])
    ypos = np.array([irect.pos().y() for irect in rects])
    if len(np.unique(xsize)) == len(xsize):
        index = np.where(xsize==w)[0][0]
    elif len(np.unique(ysize)) == len(ysize):
        index = np.where(ysize == h)[0][0]
    elif len(np.unique(xpos)) == len(xpos):
        index = np.where(xpos == x)[0][0]
    elif len(np.unique(ypos)) == len(ypos):
        index = np.where(ypos == h)[0][0]
    else:
        index=0

    imgs[index].setRect(QtCore.QRect(0,0,w,w/ratios[index]))
    rects[index].setSize((w,w/ratios[index]))
    return index


for i in np.arange(len(imgs)):
    imgs[i].setParentItem(rects[i])
    rects[i].sigRegionChanged.connect(update)

v1.autoRange()
v1.disableAutoRange('xy')


export = QtGui.QPushButton("Export transform", parent=w)
export.show()
plus = QtGui.QPushButton("+", parent=w)
plus.move(0,25)
plus.show()
minus = QtGui.QPushButton("-", parent=w)
minus.move(0,50)
minus.show()

def export_function():
    x = np.array([rect.pos().x() for rect in rects])
    y = np.array([rect.pos().y() for rect in rects])
    w = np.array([rect.size().x() for rect in rects])
    h = np.array([rect.size().y() for rect in rects])
    a = np.array([rect.angle() for rect in rects])
    np.save(NAME_FILE_SAVE,(paths,x,y,w,h,a))


def opacity_plus():
    global index
    imgs[index].setImage(opacity=alphas[index]+0.1)
    alphas[index] += 0.1

def opacity_minus():
    global index
    imgs[index].setImage(opacity=alphas[index]-0.1)
    alphas[index] -= 0.1


export.pressed.connect(export_function)
plus.pressed.connect(opacity_plus)
minus.pressed.connect(opacity_minus)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
