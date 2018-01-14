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

final_listdedir = ['/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170620_souris1', # Bof
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170620_souris2', # bof
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170620_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170621_souris2', # bof
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170621_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170622_souris1-2',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170622_souris1', # great
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170622_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170628_souris1',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170628_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170629_souris3-2',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170629_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170630_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170703_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170704_souris3',
    #  '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170704_souris4',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170705_souris3',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170821_souris5',
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170823_souris5',# bof
     '/run/user/1000/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/Alex/Auditory/Axons/170830_souris5'] # bof





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

# paths,x,y,w,h,a = load("/home/alexandre/data/dev/localisation_fields/thalamus/save/ioi.npy")
# dirnames = ["/home/alexandre/data/dev/localisation_fields/thalamus/save/souris1.npy",
#             "/home/alexandre/data/dev/localisation_fields/thalamus/save/souris2.npy",
#             "/home/alexandre/data/dev/localisation_fields/thalamus/save/souris3.npy",
#             "/home/alexandre/data/dev/localisation_fields/thalamus/save/souris5.npy",]
#
# for i in np.arange(len(dirnames)):
#     pathsc,xc,yc,wc,hc,ac = load(dirnames[i])
#     xc, yc, ac = small_transf(xc, yc, ac, x[i], y[i], a[i], 0)
#     paths, x, y, w, h, a = merge(paths, x, y, w, h, a, pathsc, xc, yc, wc, hc, ac)
#
# selection = np.arange(len(paths))
# vesselfile = np.array([str(path).find("vessels") for path in paths]) != -1 # where is the vessel map
# selection = np.concatenate([[0], selection[~vesselfile][4:]]) # A mettre a la main, depends du nombre de souris
# paths, x, y, w, h, a = paths[selection], x[selection], y[selection], w[selection], h[selection], a[selection]
# s = np.concatenate([[0],np.argsort(paths)[:-1]])
# paths, x, y, w, h, a = paths[s], x[s], y[s], w[s], h[s], a[s]
# np.save("/home/alexandre/data/dev/localisation_fields/thalamus/save/SAVE_FINISH.npy", (paths, x, y, w, h, a))


regionsfiles = [name + "/regions.npy" for name in final_listdedir]
dir = "/home/alexandre/data/dev/localisation_fields/thalamus"
paths, x, y, w, h, a = load("/home/alexandre/data/dev/localisation_fields/thalamus/save/SAVE_FINISH.npy")

IMGS = []
XX = []
YY = []
DD = []
suma = 0
for i in np.arange(len(regionsfiles)):
    loc,depth,nx,ny,idx = np.load(regionsfiles[i])
    eqbrice = '/run/user/1000/gvfs/smb-share:server=157.136.60.15,share=eqbrice/Alex/model_clust_loc/thalamus/'
    rm_cell = np.load(eqbrice + final_listdedir[i].split("/")[-1]+"_rmneurons.npy")
    loc = np.delete(loc,rm_cell)
    print(len(loc))
    nx, ny = int(nx), int(ny)
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

np.save("/home/alexandre/data/script/model_clust_loc/thalamus/coordinates_points.npy", (XX,YY,DD))

IMGS.insert(0,np.array(Image.open(paths[0]), dtype=np.float32))
imgs_src = IMGS
imgs_src[0] = zoom(imgs_src[0],(8.5,8.5,1))

alphas = np.ones(len(imgs_src))
alphas[0] = 0.2
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
