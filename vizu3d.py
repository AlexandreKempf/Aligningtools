
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

def scattertime(pos, mfc=[1,1,1,0.8], mec=[0,0,0,0.8], mfs=5, mes=1, bgc=[1,1,1],
            scaling = False, symbol = 'disc', title = 'Vispy Canvas'):
    """ Display a scatter plot in 2D or 3D.

    Parameters
    ----------
    pos : array
        The array of locations to display each symbol.
    mfc : Color | ColorArray
        The color used to draw each symbol interior.
    mec : Color | ColorArray
        The color used to draw each symbol outline.
    mfs : float or array
        The symbol size in px.
    mes : float | None
        The width of the symbol outline in pixels.
    bgc : Color
        The color used for the background.
    scaling : bool
        If set to True, marker scales when rezooming.
    symbol : str
        The style of symbol to draw ('disc', 'arrow', 'ring', 'clobber',
        'square', 'diamond', 'vbar', 'hbar', 'cross', 'tailed_arrow', 'x',
        'triangle_up', 'triangle_down', 'star').
    """
    # Create the Canvas, the Scene and the View
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=bgc, title=title)
    canvas2 = scene.SceneCanvas(keys='interactive', size=(80, 80), show=True, bgcolor=bgc)

    view = canvas.central_widget.add_view()
    # Create the scatter plot
    scatter = visuals.Markers()
    scatter.set_data(pos, face_color=mfc[0,:,:], edge_color=mec, scaling = scaling, size=mfs, edge_width=mes, symbol=symbol)
    view.add(scatter)
    t1 = Text('frame (ms) : 0', parent=view.scene, color='black')
    # 2D Shape
    if pos.shape[1] == 2:
        # Set the camera properties
        view.camera = 'panzoom'

    # 3D Shape
    elif pos.shape[1] == 3:
        view.camera = 'turntable'
        global time
        time=0


    def update(ev):
        global time
        time+=1
        t1.text = 'frame (ms) : ' + str(time)
        time=time%mfc.shape[0]
        scatter.set_data(pos=pos, face_color=mfc[time,:,:], edge_color=mec, scaling = scaling, size=mfs)
        if 10 < time < 25 :
        # if 15 < time < 46 :
            canvas2.bgcolor = (0,0,0)
        else :
            canvas2.bgcolor = (1,1,1)

    timer = app.Timer(.2, connect=update, start=True)
    app.run()



def scatter(pos, mfc=[1,1,1,0.8], mec=[0,0,0,0.8], mfs=5, mes=1, bgc=[1,1,1],
            scaling = False, symbol = 'disc',  title = 'Vispy Canvas'):
    """ Display a scatter plot in 2D or 3D.
    Parameters
    ----------
    pos : array
        The array of locations to display each symbol.
    mfc : Color | ColorArray
        The color used to draw each symbol interior.
    mec : Color | ColorArray
        The color used to draw each symbol outline.
    mfs : float or array
        The symbol size in px.
    mes : float | None
        The width of the symbol outline in pixels.
    bgc : Color
        The color used for the background.
    scaling : bool
        If set to True, marker scales when rezooming.
    symbol : str
        The style of symbol to draw ('disc', 'arrow', 'ring', 'clobber',
        'square', 'diamond', 'vbar', 'hbar', 'cross', 'tailed_arrow', 'x',
        'triangle_up', 'triangle_down', 'star').
    """
    # Create the Canvas, the Scene and the View
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=bgc, title=title)
    view = canvas.central_widget.add_view()
    # Create the scatter plot
    scatter = visuals.Markers()
    scatter.set_data(pos, face_color=mfc, edge_color=mec, scaling = scaling, size=mfs, edge_width=mes, symbol=symbol)
    view.add(scatter)
    view.camera = 'turntable'
    app.run()






################################################################################
##########################         SCRIPT         ##############################
################################################################################

# final_listdedir = ['/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160513_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160517am_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160517pm_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160518am_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160518pm_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160523_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160524am_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160524pm_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160525_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160525am_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160525pm_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160526am_cage4_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160526am_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160527_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160527_cage4_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160527_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160530_cage1_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160530_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160608_cage4_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160608_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160608_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160609_cage4_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160609_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160609_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160613_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160613_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160615_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160615_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160619_cage4_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160619_cage4_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160619_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160620_cage4_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160620_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160621_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160621pm_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622am_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622pm_cage4_mouse3',
#    # '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622pm_cage6_mouse1', # BUG ON THIS ONE
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160622pm_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160705_cage5_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160705_cage6_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160706_cage4_mouse3',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160707_cage6_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160707_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160711_cage6_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160711_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160711pm_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160712_cage6_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160712_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160712pm_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160713_cage6_mouse1',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160713_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160713pm_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160720am_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160720pm_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160721am_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160721pm_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160722_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160724_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160726_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160727_cage6_mouse2',
#  '/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/ANALYSIS/thibault/160728_cage6_mouse2']

#
# DATA = []
# for path in final_listdedir:
#     d = np.load(path + "/data.npy")[0]
#     d -= np.mean(d[:,:,:,:15], 3,keepdims=True)
#     DATA.append(d.mean(2))
#     print(path)
# DATA = np.concatenate(DATA,0)
# np.save("DATA_all.npy", DATA)

colorm = "global"
XX,YY,DD = np.load("coordinates_points.npy")
XX, YY, DD = XX.astype(float), YY.astype(float), DD.astype(float)
DD *= -1
DATA = np.load("DATA_all.npy")
STIM, STIM_ORD = np.load("stim_order.npy")
STIM_ORD = np.array(STIM_ORD, dtype=int)
DATA = np.take(DATA,STIM_ORD,1)
DATA /= DATA[:,:,:15].reshape(DATA.shape[0],-1).std(1)[:,np.newaxis, np.newaxis]
POS = np.concatenate([[XX],[YY],[DD]],0).T


for istim in np.arange(1,2):
# istim = 1
    DAT = DATA[:,istim,5:55].T
    DAT = gaussian_filter(DAT, (2,0))
    if colorm == "local":
        DAT = np.maximum(np.percentile(DAT,80), DAT)
        DAT = np.minimum(np.percentile(DAT,99.9),DAT)
        DAT -= DAT.min()
        DAT /= DAT.max()
        COLOR=plt.cm.jet(DAT)
        COLOR[:,:,3] *= DAT/1.2
        COLOR[DAT<0.7,3] = 0.01
    elif colorm == "global":
        DAT = np.maximum(np.percentile(DATA,70), DAT)
        DAT = np.minimum(np.percentile(DATA,99.9),DAT)
        DAT -= DAT.min()
        DAT /= DAT.max()
        COLOR=plt.cm.jet(DAT)
        COLOR[:,:,3] *= DAT/1.2
        COLOR[DAT<0.7,3] = 0.02


    scattertime(POS, mfc=COLOR, mec=[0,0,0,0.0], mfs=9, mes=0, bgc=[1,1,1], scaling = False, symbol = 'disc', title= STIM[istim] );
