#!/usr/bin/python

import numpy as np

#-------------------------------------------------------------------------------------
def takespread (sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(np.ceil(i * length / num))]

#-------------------------------------------------------------------------------------
def makespread (sequence, num):
    length = float(len(sequence))
    seq = np.array(sequence)
    return seq[np.ceil(np.arange(num) * length / num).astype(int)]

##==============================================================================
def save_fig_to_png (fig, fname, facecolor=None):

    import pylab as plt

    print ("Saving " + fname)
    fig.set_size_inches(22,16)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=600, facecolor=facecolor)
    plt.close(fig)

#-------------------------------------------------------------------------------------
def show_many_slices (vol, vol2=None, volaxis=1, n_slices=[8, 8], slices_idx=None,
                      vol1_colormap=None, vol1_transp_val=None,
                      vol2_colormap=None, vol2_transp_val=0,
                      interpolation='nearest', figtitle='', facecolor='', show_colorbar=True):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

    class ImageObserver:
        'update image in response to changes in clim or cmap on another image'
        def __init__(self, follower):
            self.follower = follower
        def __call__(self, leader):
            self.follower.set_cmap(leader.get_cmap())
            self.follower.set_clim(leader.get_clim())

    if isinstance(vol2, np.ndarray):
        assert vol.shape == vol2.shape, 'vol do not have the same shape as vol2'
        if vol2_colormap == None:
            vol2_colormap = plt.cm.jet

    if vol1_colormap == None:
        vol1_colormap = plt.cm.gray

    size   = vol.shape[volaxis]
    n_rows = len(n_slices)
    n_cols = max(n_slices)

    if not slices_idx:
        slice_idx = makespread (range(size), np.sum(n_slices))

    fig  = plt.figure(figtitle)

    if facecolor:
        fig.set_facecolor(facecolor)

    barargs = {}
    if show_colorbar:
        barargs = dict(cbar_location="right",
                       cbar_mode='single',
                       cbar_size='10%',
                       cbar_pad=0.05)

    grid = ImageGrid(fig, 111,
                     nrows_ncols = (n_rows, n_cols),
                     axes_pad=0,
                     direction='column',
                     **barargs)

    axes   = []
    images = []
    vmin   =  1e40
    vmax   = -1e40

    for g in grid:
        g.axis('off')

    c = 1
    for i in slice_idx[:-2]:
        g = grid[c-1]

        img = vol.take([i], volaxis).squeeze()
        if vol1_transp_val != None:
            img = np.ma.masked_where(img == vol1_transp_val, img)
        g.imshow(np.rot90(img), cmap=vol1_colormap, interpolation=interpolation)

        if isinstance(vol2, np.ndarray):
            img2 = vol2.take([i], volaxis).squeeze()
            img2 = np.ma.masked_where(img2 == vol2_transp_val, img2)
            images.append(g.imshow(np.rot90(img2), cmap=vol2_colormap, interpolation=interpolation))
            dd = np.ravel(img2)
            vmin = min(vmin, np.amin(dd))
            vmax = max(vmax, np.amax(dd))

        #gpos  = g.get_position()
        #gaxis = g.get_axis()
        axes.append(g)
        textx = img.shape[0]/2
        texty = 4
        g.text(textx, texty, str(i), horizontalalignment='center',
               fontsize=12, fontname='Arial', color = '#0055ff')

        c += 1

    if show_colorbar:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for i, im in enumerate(images):
            im.set_norm(norm)
            if i > 0:
                images[0].callbacksSM.connect('changed', ImageObserver(im))

        axes[-1].cax.colorbar(images[-1])

    return fig

#-------------------------------------------------------------------------------------
# Matplotlib option
#-------------------------------------------------------------------------------
def show_3slices (vol, vol2=None, x=None, y=None, z=None, fig=None, 
                       vol2_colormap = None, vol2_transp_val = 0, 
                    interpolation = 'nearest'):

    assert vol.shape == vol2.shape, 'vol do not have the same shape as vol2'

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.widgets import Slider, Button

    if not x:
        x = np.floor(vol.shape[0]/2)
    if not y:
        y = np.floor(vol.shape[1]/2)
    if not z:
        z = np.floor(vol.shape[2]/2)

    if not fig:
        fig = plt.figure()
    else:
        fig = plt.figure(fig.number)

    try:
        if not np.ma.is_masked(vol2):
            vol2 = np.ma.masked_equal(vol2, vol2_transp_val)
    except:
        pass

    if vol2_colormap == None:
        vol2_colormap = plt.cm.hot

    plt.subplot (2,2,1)
    plt.axis('off')
    plt.imshow(np.rot90(vol[x,...]), cmap=plt.cm.gray, interpolation=interpolation)
    try:
        plt.imshow(np.rot90(vol2[x,...]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass
    #plt.imshow(vol[x,...], cmap=plt.cm.gray)
    plt.title ('X:' + str(int(x)))

    plt.subplot (2,2,2)
    plt.axis('off')
    plt.imshow(np.rot90(vol[:,y,:]), cmap=plt.cm.gray, interpolation=interpolation)
    try:
        plt.imshow(np.rot90(vol2[:,y,:]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass

    plt.title ('Y:' + str(int(y)))

    plt.subplot (2,2,3)
    plt.axis ('off')
    plt.imshow(np.rot90(vol[...,z]), cmap=plt.cm.gray, interpolation=interpolation)
    try:
        plt.imshow(np.rot90(vol2[...,z]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass
    plt.title ('Z:' + str(int(z)))

    plt.subplot (2,2,4)

    plt.axis('off')
    axcolor = 'lightgoldenrodyellow'
    xval = plt.axes([0.60, 0.20, 0.20, 0.03], axisbg=axcolor)
    yval = plt.axes([0.60, 0.15, 0.20, 0.03], axisbg=axcolor)
    zval = plt.axes([0.60, 0.10, 0.20, 0.03], axisbg=axcolor)

    xsli = Slider(xval, 'X', 0, vol.shape[0]-1, valinit=x, valfmt='%i')
    ysli = Slider(yval, 'Y', 0, vol.shape[1]-1, valinit=y, valfmt='%i')
    zsli = Slider(zval, 'Z', 0, vol.shape[2]-1, valinit=z, valfmt='%i')

    def update(val):
        x = np.around(xsli.val)
        y = np.around(ysli.val)
        z = np.around(zsli.val)
        #debug_here()
        visualize_3slices (vol, vol2, x, y, z, fig)
        #fig.set_data()

    xsli.on_changed(update)
    ysli.on_changed(update)
    zsli.on_changed(update)

    plt.show()

    return fig
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# Mayavi2 options
#-------------------------------------------------------------------------------------
def show_cutplanes (img, first_idx    = 10, 
                              second_idx   = 10,
                              first_plane  = 'x_axes',
                              second_plane = 'y_axes',
                        ):

    from mayavi import mlab
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(img),
                            plane_orientation=first_plane,
                            slice_index=first_idx,
                            colormap='gray',
                        )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(img),
                            plane_orientation=second_plane,
                            slice_index=second_idx,
                            colormap='gray',
                        )
    mlab.outline()

#-------------------------------------------------------------------------------------
def show_dynplane (img):
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(img)
    mlab.pipeline.iso_surface(src, contours=[img.min()+0.1*img.ptp(), ], opacity=0.1)
    mlab.pipeline.iso_surface(src, contours=[img.max()-0.1*img.ptp(), ],)
    mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=10,
                            colormap='gray',
                        )


#-------------------------------------------------------------------------------------
def show_contour (img):
    from mayavi import mlab
    mlab.contour3d(img, colormap='gray')

#-------------------------------------------------------------------------------------
def show_render (img, vmin=0, vmax=0.8):
    from mayavi import mlab
    mlab.pipeline.volume(mlab.pipeline.scalar_field(img), vmin=vmin, vmax=vmax, colormap='gray')

#-------------------------------------------------------------------------------------
# pyqtgraph option
#-------------------------------------------------------------------------------
#def visualize (vol):

#    import sys
#    import numpy as np
#    import nibabel as nib

#    f = '/old_data/data/vbm_svm/subjects_nifti/control_OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.nii'
#    s = nib.load(f).get_data()
#    vol = s.copy()

#    import numpy as np
#    import scipy
#    from pyqtgraph.Qt import QtCore, QtGui
#    import pyqtgraph as pg

#    app = QtGui.QApplication([])

#    ## Create window with three ImageView widgets
#    win = QtGui.QMainWindow()
#    win.resize(800,800)
#    cw = QtGui.QWidget()
#    win.setCentralWidget(cw)
#    l = QtGui.QGridLayout()
#    cw.setLayout(l)
#    imv1 = pg.ImageView()
#    imv2 = pg.ImageView()
#    imv3 = pg.ImageView()
#    l.addWidget(imv1, 0, 0)
#    l.addWidget(imv2, 1, 0)
#    l.addWidget(imv3, 0, 1)
#    win.show()

#    roi = pg.TestROI(pos=[10, 10], size=pg.Point(2,2), pen='r') 
#    imv1.addItem(roi)

#    def update():
#        global vol, imv1, imv2, imv3
#        d2 = roi.getArrayRegion(vol, imv1.imageItem, axes=(1,2))
#        imv2.setImage(d2)

#    imv1.setImage(vol)

#    volmin = np.min(vol)
#    volmax = np.max(vol)
#    imv1.setHistogramRange(volmin, volmax)

#    lowlvl = 0
#    uprlvl = 1
#    if volmin < 0: 
#        lowlvl = volmin

#    if volmax > 1: 
#        uprlvl = volmax

#    imv1.setLevels(lowlvl, uprlvl)

#    update()
#    
#    if sys.flags.interactive != 1:
#        app.exec_()
















