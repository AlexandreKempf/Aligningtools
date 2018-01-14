import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/alexandre/docs/code/pkg/imgca/new/imgca')
import imgca as ca
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster,leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
import matplotlib.patches as patches
from sklearn.decomposition import NMF
import glob
# import module
from vispy import gloo, app, scene
from vispy.scene import visuals, canvas
from vispy.scene.visuals import Text
from scipy.ndimage import gaussian_filter



XX,YY,DD = np.load("coordinates_points.npy")
XX, YY, DD = XX.astype(float), YY.astype(float), DD.astype(float)
DD *= -1
DATA = np.load("DATA_all.npy")
STIM, STIM_ORD = np.load("stim_order.npy")
STIM_ORD = np.array(STIM_ORD, dtype=int)
DATA = np.take(DATA,STIM_ORD,1)
DATA /= DATA[:,:,:15].reshape(DATA.shape[0],-1).std(1)[:,np.newaxis, np.newaxis]
DATA = DATA[:,:,5:55]
DATA = DATA.reshape((DATA.shape[0],DATA.shape[1],int(DATA.shape[2]/2),2)).mean(-1)
POS = np.concatenate([[XX],[YY],[DD]],0).T

CUBES = []
Xmin, Xmax = XX.min(), XX.max()
Ymin, Ymax = YY.min(), YY.max()
Dmin, Dmax = DD.min(), DD.max()

def bin3D(DATA, XX, YY, DD, istim, Xmin, Xmax, Ymin, Ymax, Dmin ,Dmax):
    print(istim)
    DAT = DATA[:,istim,:].T
    # DAT = gaussian_filter(DAT, (3,0))
    # DAT = np.maximum(np.percentile(DAT,70), DAT)
    # DAT = np.minimum(np.percentile(DAT,99.9),DAT)
    # DAT -= DAT.min()
    # DAT /= DAT.max()
    CUBE = np.zeros((30,30,5,DAT.shape[0]))
    xborders = np.linspace(Xmin, Xmax, CUBE.shape[0])
    yborders = np.linspace(Ymin, Ymax, CUBE.shape[1])
    dborders = np.linspace(Dmin, Dmax, CUBE.shape[2])
    for x in np.arange(len(xborders)-1):
        for y in np.arange(len(yborders)-1):
            for d in np.arange(len(dborders)-1):
                ixs = (XX > xborders[x]) & (XX < xborders[x+1])
                iys = (YY > yborders[y]) & (YY < yborders[y+1])
                ids = (DD > dborders[d]) & (DD < dborders[d+1])
                cells = ixs & iys & ids
                if np.sum(cells) !=0:
                    CUBE[x,y,d,:] = np.mean(DAT[:,cells],1)
    return CUBE

CUBES = np.array([bin3D(DATA, XX, YY, DD, i, Xmin, Xmax, Ymin, Ymax, Dmin ,Dmax) for i in np.arange(148)])


################################################################################
# Score

def nnh_ratio(classes, idx, radius, XX, YY, DD):
    ix, iy, id = XX[idx], YY[idx], DD[idx]
    cells = np.where(np.sqrt((XX-ix)**2 + (YY-iy)**2 + (DD-id)**2) < radius)[0]
    ratio = (np.sum(classes[cells] == classes[idx])-1)/(len(cells)-1)
    return ratio


def score_class(classes,radius, XX, YY, DD):
    classe_uniq = np.unique(classes)
    mean_ratios = np.zeros(len(classe_uniq))
    std_ratios = np.zeros(len(classe_uniq))

    for iclass in np.arange(classe_uniq.shape[0]):
        cells = np.where(classes == classe_uniq[iclass])[0]
        ratios = [nnh_ratio(classes,i,radius,XX,YY,DD) for i in cells]
        mean_ratios[iclass] = np.mean(ratios)*len(classes)/len(cells)
        std_ratios[iclass] = np.std(ratios)
    return mean_ratios, std_ratios


# classes = np.random.randint(0,6,len(XX))
for isound in np.arange(148):
    classes = np.mean(DATA[:,isound,5:10],1)
    classes = np.digitize(classes, np.percentile(classes,(0,70,80,85,90,95,98,99)))

    # color = plt.cm.jet(classes/np.max(classes))
    # color[:,3]=0.2
    # scatter(POS, color, [0,0,0,0])

    graph = np.array([score_class(classes, radius, XX, YY, DD)[0] for radius in [40,50,60,80,100,150,200,350,500,1000]]).T
    [plt.plot(graph[i], color = plt.cm.jet(np.arange(graph.shape[0])/(graph.shape[0]-1))[i], lw=2) for i in np.arange(graph.shape[0])];
    plt.title(STIM[isound]);
    plt.ylim([0,10])
    plt.savefig("plot_"+str(STIM[isound])+".png")
    plt.close()

################################################################################

1+1

def image(CUBES, STIM, istim, timeborders, clim=[]):
    img = np.nanmean(CUBES[:,:,:,:,:],3)[istim]
    img = gaussian_filter(img,(1.5,1.5,2))
    img = np.nanmean(img[:,:,timeborders[0]:timeborders[1]],2)
    image = plt.imshow(img, interpolation="none", origin="lower")
    if len(clim)!=0:
        plt.clim(clim)
    plt.title(STIM[istim])

image(CUBES, STIM, 0, [10,30], [0,0.25]); plt.show()


def animation(CUBES, STIM, istim, soundborders=[10,41], clim=[], rep = 1):
    img = np.nanmean(CUBES[:,:,:,:,:],3)[istim]
    img = gaussian_filter(img,(1.5,1.5,1.5))

    image = plt.imshow(img[:,:,0], interpolation="none", origin="lower")
    if len(clim)==0:
        plt.clim(np.percentile(img[:,:,:],(3,97)))
    else:
        plt.clim(clim)

    for i in np.tile(np.arange(img.shape[2]),rep):
        image.set_data(img[:,:,i])
        if i>=soundborders[0]:
            if i <=soundborders[1]:
                plt.title(STIM[istim])
            else:
                plt.title("")
        else:
            plt.title("")

        plt.pause(0.01)
    plt.close()

animation(CUBES, STIM, 3, [10,30], clim = [0,0.25], rep=2)


def tonotopy(CUBES, istims, weights, timeborders, clim=[]):
    imgs = np.nanmean(CUBES[:,:,:,:,:],3)[np.array(istims)]
    imgs = np.nanmean(imgs[:,:,:,timeborders[0]:timeborders[1]],3)
    imgs = gaussian_filter(imgs,(0,1.5,1.5))
    imgs = imgs*np.array(weights).reshape([-1,1,1])
    img = np.nanmean(imgs,0)

    image = plt.imshow(img, interpolation="none", origin="lower")
    if len(clim)!=0:
        plt.clim(clim)
    plt.title("Tonotopy")

tonotopy(CUBES, [1,2,3,4,5,6], [-4,-3,-1,+1,+2,+3], [10,30]); plt.show()


################################################################################

def map_corr_complex(CUBES, istims):
    data = CUBES[np.array(istims)]
    CORR = np.zeros((len(istims), len(istims)))
    for i in np.arange(len(istims)):
        print(i)
        for j in np.arange(i+1,len(istims)):
            img0 = data[i]
            img0 = img0.reshape((-1,data.shape[-1]))
            img1 = data[j]
            img1 = img1.reshape((-1,data.shape[-1]))
            cor = np.corrcoef(img0,img1)

            inter = np.nanmean(np.diag(cor[int(cor.shape[0]/2):, :int(cor.shape[0]/2)]))
            CORR[i,j]= inter
            CORR[j,i]= inter
    CORR[np.arange(CORR.shape[0]),np.arange(CORR.shape[0])] = 1
    return CORR


def map_corr(CUBES, istims):
    data = CUBES[np.array(istims)]
    data = data.reshape((data.shape[0],-1))
    cor = np.corrcoef(data)
    return cor

istims = np.arange(148)
cor_map = map_corr(CUBES, istims)
plt.imshow(cor_map, interpolation="none");
plt.yticks(np.arange(cor_map.shape[0]), STIM[istims], rotation='horizontal');
plt.clim(0,0.5);plt.show()


# Bootstraping (take hours)
CORMAP=[]
for j in np.arange(100):
    XX = np.random.choice(XX, len(XX), replace=False)
    YY = np.random.choice(YY, len(YY), replace=False)
    DD = np.random.choice(DD, len(DD), replace=False)
    CUBES = np.array([bin3D(DATA, XX, YY, DD, i, Xmin, Xmax, Ymin, Ymax, Dmin ,Dmax) for i in np.arange(148)])

    istims = np.arange(148)
    cor_map = map_corr(CUBES, istims)
    CORMAP.append(cor_map)
    # plt.imshow(cor_map, interpolation="none");
    # plt.yticks(np.arange(cor_map.shape[0]), STIM[istims], rotation='horizontal');
    # plt.clim(0,0.5);plt.show()

CORMAP = np.array(CORMAP)

# np.save("CORMAP_random.npy", np.array(CORMAP))
# CORMAP = np.load("CORMAP_random.npy")



################################################################################

def activity_corr(DATA, istims):
    data = DATA[:,np.array(istims),:].transpose(1,0,2)
    data = data.reshape(data.shape[0],-1)
    cor = np.corrcoef(data)
    return cor


istims = np.arange(148)
cor_act = activity_corr(DATA, istims)
plt.imshow(cor_act, interpolation="none");
plt.yticks(np.arange(cor_act.shape[0]), STIM[istims], rotation='horizontal');
plt.clim(0,0.3);plt.show()

plt.subplot(121);
plt.imshow(cor_map, interpolation="none");
plt.yticks(np.arange(cor_map.shape[0]), STIM[istims], rotation='horizontal');
plt.clim(0,0.5);

plt.subplot(122);
plt.imshow(cor_act, interpolation="none");
plt.yticks(np.arange(cor_act.shape[0]), STIM[istims], rotation='horizontal');
plt.clim(0,0.3);plt.show()
