import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PIL import Image
import glob
from scipy.ndimage import zoom
from scipy.io import loadmat, savemat
import h5py





IMGS = "" #defaut
final_listdedir = ['/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160513_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160517am_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160517pm_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160518am_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160518pm_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160523_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160524am_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160524pm_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160525_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160525am_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160525pm_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160526am_cage4_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160526am_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160527_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160527_cage4_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160527_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160530_cage1_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160530_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160608_cage4_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160608_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160608_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160609_cage4_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160609_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160609_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160613_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160613_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160615_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160615_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160619_cage4_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160619_cage4_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160619_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160620_cage4_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160620_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160621_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160621pm_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622am_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622pm_cage4_mouse3',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622pm_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160705_cage5_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160705_cage6_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160706_cage4_mouse3',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160707_cage6_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160707_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160711_cage6_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160711_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160711pm_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160712_cage6_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160712_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160712pm_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160713_cage6_mouse1',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160713_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160713pm_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160720am_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160720pm_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160721am_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160721pm_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160722_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160724_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160726_cage6_mouse2',
 '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160727_cage6_mouse2']



def mapping(loc,nx,ny,data):
    """
    data is a vector of size Ncell
    """
    data -= data.min()
    data *= data.max()
    allpix = np.zeros((ny*nx,4))
    for i in np.arange(len(loc)):
        allpix[loc[i]] = np.array(plt.cm.jet(data[i]))*data[i]
    allpixsquare = np.reshape(allpix,(nx,ny,4))
    return allpixsquare

def rotate_xy(xi,yi,xc,yc,a):
    """
    coordinate of on old (xi,yi) points after a rotation of "a" degree around (xc,yc)
    """
    a *= np.pi/180
    x = (xi-xc)*np.cos(a) - (yi-yc)*np.sin(a) + xc
    y = (xi-xc)*np.sin(a) + (yi-yc)*np.cos(a) + yc
    return x, y

def load(dirpath):
    paths,x,y,w,h,a = np.load(dirpath)
    x = x.astype(float)
    y = y.astype(float)
    w = w.astype(float)
    h = h.astype(float)
    a = a.astype(float)
    paths = paths.astype(str)
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



app = QtGui.QApplication([])
wind = pg.GraphicsWindow(size=(1000,800), border=True)
wind.setWindowTitle('AWESOMNESS')
w1 = wind.addLayout(row=0, col=0)
v1 = w1.addViewBox(row=0, col=0, lockAspect=True)

# paths,x,y,w,h,a = load("/home/alexandre/img/image_thibaut/save/ioi.npy")
# dirnames = ["/home/alexandre/img/image_thibaut/save/c1m2_c1m2.npy",
#             "/home/alexandre/img/image_thibaut/save/c4m1_c4m1.npy",
#             "/home/alexandre/img/image_thibaut/save/c4m2_c4m2.npy",
#             "/home/alexandre/img/image_thibaut/save/c4m3_c4m3.npy",
#             "/home/alexandre/img/image_thibaut/save/c5m2_c5m2.npy",
#             "/home/alexandre/img/image_thibaut/save/c6m1_c6m1.npy",
#             "/home/alexandre/img/image_thibaut/save/c6m2_c6m2.npy"]
#
# for i in np.arange(len(dirnames)):
#     pathsc,xc,yc,wc,hc,ac = load(dirnames[i])
#     xc, yc, ac = small_transf(xc, yc, ac, x[i], y[i], a[i], 0)
#     paths, x, y, w, h, a = merge(paths, x, y, w, h, a, pathsc, xc, yc, wc, hc, ac)
#
# selection = np.arange(len(paths))
# vesselfile = np.array([str(path).find("vessels") for path in paths]) != -1 # where is the vessel map
# selection = selection[~vesselfile][6:]
# paths, x, y, w, h, a = paths[selection], x[selection], y[selection], w[selection], h[selection], a[selection]
# s = np.concatenate([[0],np.argsort(paths)[:-1]])
# paths, x, y, w, h, a = paths[s], x[s], y[s], w[s], h[s], a[s]
# np.save("SAVE_FINISH.npy", (paths, x, y, w, h, a))

# stim,stim_order = np.load("stim_order.npy")
# stim_order = np.array(stim_order,dtype=int)
# GLMfiles = [name + "/GLMscore.npy" for name in final_listdedir]
regionsfiles = [name + "/regions.npy" for name in final_listdedir]
dir = "/home/alexandre/data/dev/localisation_fields/cortex/"
# paths, x, y, w, h, a = load("/home/alexandre/data/dev/localisation_fields/cortex/SAVE_FINISH.npy")



paths,x,y,w,h,a = load("/home/alexandre/data/dev/localisation_fields/cortex/SAVE_FINISH_removelast.npy")

IMGS = []
XX = []
YY = []
DD = []
suma = 0
for i in np.arange(len(regionsfiles)):
    loc,depth,ny,nx,idx = np.load(regionsfiles[i])
    # glm = np.load(GLMfiles[i])[stim_order,:]


    # color = 1-glm[33]
    # color -= 0.99
    # color[color<0]=0
    # color *= 1/np.percentile(color,98)
    # color[color>1]=1
    color = np.random.random((len(loc)))
    img = mapping(loc,nx,ny,color)
    IMGS.append(img)
    X = np.array([np.mean(l//ny) for l in loc]) + x[i+1]
    Y = np.array([np.mean(l%ny) for l in loc]) + y[i+1]
    X, Y = rotate_xy(X, Y ,x[i+1], y[i+1], a[i+1])
    XX.extend(X)
    YY.extend(Y)
    DD.extend(np.repeat(depth, len(X)))

# [print(i,stim[i]) for i in np.arange(len(stim))]

np.save("/home/alexandre/data/script/model_clust_loc/cortex/coordinates_points.npy", (XX,YY,DD))

IMGS.insert(0,np.array(Image.open(dir+paths[0]), dtype=np.float32))
imgs_src = IMGS
imgs_src[0] = zoom(imgs_src[0],(2.6,2.6,1))

alphas = np.zeros(len(imgs_src))+0.2
alphas[0] = 0.5
imgs = [pg.ImageItem(imgs_src[i], opacity=alphas[i]) for i in np.arange(len(x))]
rects = [pg.RectROI([x[i], y[i]], [w[i], h[i]], angle = a[i], pen=QtGui.QPen(QtGui.QColor(0,0,0,0))) for i in np.arange(len(x))]
[v1.addItem(rect) for rect in rects]
# points = pg.PlotDataItem(XX, YY)
# v1.addItem(points)

for i in np.arange(len(imgs)):
    imgs[i].setParentItem(rects[i])


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
