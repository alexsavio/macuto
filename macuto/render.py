# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------
import os

import numpy as np
import skimage.io as skio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nipy.labs.viz_tools.edge_detect import _edge_detect

from .math import makespread
from .files.names import get_temp_file

#-------------------------------------------------------------------------------------
# Matplotlib-based options
#-------------------------------------------------------------------------------
def show_many_slices(vol, vol2=None, volaxis=1, n_slices=[8, 8], slices_idx=None,
                     vol1_colormap=None, vol1_transp_val=None,
                     vol2_colormap=None, vol2_transp_val=0,
                     interpolation='nearest', figtitle=None, facecolor='',
                     is_red_outline=False, show_colorbar=True):
    """
    @param vol: numpy 3D array
    @param vol2: numpy 3D array
    @param volaxis: int
    @param n_slices: list of int
    @param slices_idx: list of int
    @param vol1_colormap: matplotlib colormap
    @param vol1_transp_val: vol1.dtype scalar
    Volume1 transparent value

    @param vol2_colormap: matplotlib colormap
    @param vol2_transp_val: vol2.dtype scalar
    Volume2 transparent value

    @param interpolation: string, optional
    @param figtitle: string, optional
    @param facecolor: string, optional
    @param is_red_outline: boolean, optional
    @param show_colorbar: boolean, optional

    @return: matplotlib figure
    """

    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import ImageGrid

    class ImageObserver:
        'update image in response to changes in clim or cmap on another image'
        def __init__(self, follower):
            self.follower = follower
        def __call__(self, leader):
            self.follower.set_cmap(leader.get_cmap())
            self.follower.set_clim(leader.get_clim())

    if isinstance(vol2, np.ndarray):
        assert vol.shape == vol2.shape, 'vol do not have the same shape as vol2'
        if vol2_colormap is None:
            vol2_colormap = plt.cm.jet

    if vol1_colormap is None:
        vol1_colormap = plt.cm.gray

    size   = vol.shape[volaxis]
    n_rows = len(n_slices)
    n_cols = max(n_slices)

    if not slices_idx:
        slice_idx = makespread(list(range(size)), np.sum(n_slices))

    fig  = plt.figure(figtitle, frameon=False)

    if facecolor:
        fig.set_facecolor(facecolor)

    barargs = {}
    if show_colorbar:
        barargs = dict(cbar_location="right",
                       cbar_mode='single',
                       cbar_size='10%',
                       cbar_pad=0.05)

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_rows, n_cols),
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
        if vol1_transp_val is not None:
            img = np.ma.masked_where(img == vol1_transp_val, img)
        g.imshow(np.rot90(img), cmap=vol1_colormap,
                 interpolation=interpolation)

        if isinstance(vol2, np.ndarray):
            img2 = vol2.take([i], volaxis).squeeze()
            img2 = np.rot90(img2)

            if is_red_outline:
                _, img2 = _edge_detect(img2)
                img2 = np.ma.masked_where(img2 == 0, img2)

                #change vol2 colormap
                vol2_colormap = plt.cm.autumn_r
                show_colorbar = False

            else:
                img2 = np.ma.masked_where(img2 == vol2_transp_val, img2)

            images.append(g.imshow(img2, cmap=vol2_colormap,
                                   interpolation=interpolation))
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


def create_imglist_html(output_dir, img_files, filename='index.html'):
    """
    :param img_files: list of strings

    :return:
    """
    #import markdown
    #md_indexf = os.path.join(output_dir, 'index.markdown')
    html_indexf = os.path.join(output_dir, filename)

    with open(html_indexf, 'w') as f:
        for imgf in img_files:
                #line = '[![' + imgf + '](' + imgf + ')' + imgf + '](' + imgf + ')'
                #line = '[<img src="' + imgf + '" width=1000/>' + imgf + '](' + imgf + ')'
                line = '<p><a href="' + imgf + '"><img src="' + imgf + '"/>' +\
                       imgf + '</a></p>'
                f.write(line + '\n')

    #markdown.markdownFromFile(md_indexf, html_indexf, encoding="utf-8")


def show_connectivity_matrix(image, cmap=None):
    """
    @param image: 2D ndarray
    @param cmap: colormap
    """
    if cmap is None:
        cmap = cm.jet

    skio.imshow(image, cmap=cmap)


def autocrop_img(image, color=0):
    """
    Crops the borders of the given color of an image.

    @param image: ndarray
    @param color: int
    @return:
    ndarray
    """
    if len(image.shape) == 3:
        img = image[..., 3] if image.shape[2] == 4 else image[..., 0]
    else:
        img = image

    mask = (img != color).astype(int)

    left, right, top, bottom = borders(mask, 0)

    return image[top:bottom, left:right, :]


def imshow(image, **kwargs):
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(image, **kwargs)
    plt.axis('off')
    return fig


def borders(im, color):
    """
    @param im:
    @param color:
    @return:
    """
    #left, right, top, bottom = -1,-1,-1,-1

    non_color_pix = np.where(im != color)
    top = non_color_pix[0].min()
    bottom = non_color_pix[0].max()
    left = non_color_pix[1].min()
    right = non_color_pix[1].max()

    return left, right, top, bottom


def slicesdir_paired_overlays(output_dir, file_list1, file_list2, dpi=150,
                              is_red_outline=False, **kwargs):
    """
    @param output_dir:
    @param file_list1: list of strings
    Paths to the background image, can be either 3D or 4D images.
    If they are 4D images, will pick one of the center.

    @param file_list2: list of strings
    Paths to the overlay images, must be 3D images.

    @param is_red_outline: boolean, optional
    If True will show from the files in list2 a red outline border.

    @param kwargs: arguments to show_many_slices
    See macuto.render.show_many_slices docstring.

    @return:
    """
    assert(len(file_list1) > 0)
    assert(len(file_list1) == len(file_list2))

    from .nifti.read import get_nii_data
    from .files.names import remove_ext, get_temp_file
    #import markdown

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = []

    #CREATE separate images of each file_list2 file
    # on the corresponding file_list1 file
    for idx in list(range(len(file_list1))):
        f1_vol = get_nii_data(file_list1[idx])
        f2_vol = get_nii_data(file_list2[idx])

        if len(f1_vol.shape) > 3:
            f1_vol = f1_vol[..., int(np.floor(f1_vol.shape[3]/2))]

        fig = show_many_slices(f1_vol, f2_vol, is_red_outline=is_red_outline,
                               **kwargs)

        png_fname = os.path.relpath(remove_ext(file_list1[idx]))
        png_fname = png_fname.replace('.', '').replace('/', '_').replace('__', '') + '.png'
        png_path = os.path.join(output_dir, png_fname)

        export_figure(fig, png_path, dpi=dpi)

        plt.close(fig)

        img_files.append(os.path.basename(png_path))

    #Create the index.html file with all images
    create_imglist_html(output_dir, img_files)

    return img_files


def export_figure(fig, filepath, dpi=150):

    tmpf = get_temp_file(suffix='.png').name
    fig.savefig(tmpf, transparent=True, dpi=dpi)
    img = autocrop_img(skio.imread(tmpf))
    skio.imsave(filepath, img)
    return filepath


def slicesdir_oneset(output_dir, file_list1, dpi=150, **kwargs):
    """
    Creates a folder with a html file and png images of slices
    of each of nifti file in file_list1.

    @param output_dir: string
    Path to the output folder

    @param file_list1: list of strings
    Paths to the background image, can be either 3D or 4D images.
    If they are 4D images, will pick one of the center.

    @param file_list2: list of strings
    Paths to the overlay images, must be 3D images.

    @param kwargs: arguments to show_many_slices
    See macuto.render.show_many_slices docstring.

    @return:
    """
    assert(len(file_list1) > 0)

    import os
    import matplotlib.pyplot as plt
    from .nifti.read import get_nii_data
    from .files.names import remove_ext

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = []

    show_colorbar = kwargs.pop('show_colorbar', False)

    #CREATE separate images of each file_list2 file
    # on the corresponding file_list1 file
    for idx in list(range(len(file_list1))):
        f1_vol = get_nii_data(file_list1[idx])

        if len(f1_vol.shape) > 3:
            f1_vol = f1_vol[..., int(np.floor(f1_vol.shape[3]/2))]

        fig = show_many_slices(f1_vol, show_colorbar=show_colorbar, **kwargs)

        png_fname = os.path.relpath(remove_ext(file_list1[idx]))
        png_fname = png_fname.replace('.', '').replace('/', '_').replace('__', '') + '.png'
        png_path = os.path.join(output_dir, png_fname)

        export_figure(fig, png_path, dpi=dpi)

        plt.close(fig)

        img_files.append(os.path.basename(png_path))

    #Create the index.html file with all images
    create_imglist_html(output_dir, img_files)

    return img_files


def show_3slices(vol, vol2=None, x=None, y=None, z=None, fig=None,
                 vol2_colormap=None, vol2_transp_val=0, interpolation='nearest'):
    """
    @param vol: numpy 3D array
    @param vol2: numpy 3D array
    @param x: int
    @param y: int
    @param z: int
    @param fig: matplotlib Figure
    @param vol2_colormap: matplotlib colormap
    @param vol2_transp_val: scalar
    Volume2 transparent value

    @param interpolation: string

    @return: matplotlib figure
    """

    assert vol.shape == vol2.shape, 'vol do not have the same shape as vol2'

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.widgets import Slider

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

    if vol2_colormap is None:
        vol2_colormap = plt.cm.hot

    plt.subplot (2, 2, 1)
    plt.axis('off')
    plt.imshow(np.rot90(vol[x, ...]), cmap=plt.cm.gray, interpolation=interpolation)
    try:
        plt.imshow(np.rot90(vol2[x, ...]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass
    #plt.imshow(vol[x,...], cmap=plt.cm.gray)
    plt.title ('X:' + str(int(x)))

    plt.subplot (2, 2, 2)
    plt.axis('off')
    plt.imshow(np.rot90(vol[:, y, :]), cmap=plt.cm.gray, interpolation=interpolation)
    try:
        plt.imshow(np.rot90(vol2[:, y, :]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass

    plt.title ('Y:' + str(int(y)))

    plt.subplot (2, 2, 3)
    plt.axis ('off')
    plt.imshow(np.rot90(vol[..., z]), cmap=plt.cm.gray, interpolation=interpolation)
    try:
        plt.imshow(np.rot90(vol2[..., z]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass
    plt.title ('Z:' + str(int(z)))

    plt.subplot (2, 2, 4)

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
        show_3slices(vol, vol2, x, y, z, fig)
        #fig.set_data()

    xsli.on_changed(update)
    ysli.on_changed(update)
    zsli.on_changed(update)

    plt.show()

    return fig


def slicesdir_connectivity_matrices(output_dir, cmat_list, dpi=150,
                                    lower_triangle=True, **kwargs):
    """
    @param output_dir:
    @param file_list1: list of ndarrays
    List of connectivity matrices

    @param dpi: int
    Dots per inch resolution of the plot images

    @param lower_triangle: bool
    If true, will plot a lower triangle of the matrix,
    the full matrix otherwise.

    @param kwargs: arguments to show_many_slices
    See draw_square_matrix_channels named arguments.
    """
    import os
    import matplotlib.pyplot as plt
    from nitime.viz import drawmatrix_channels

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = []

    #cmap = kwargs.pop('cmap', plt.cm.rainbow)
    channel_names = kwargs.pop('channel_names', None)
    color_anchor = kwargs.pop('color_anchor', 0)
    size = kwargs.pop('size', [10., 10.])

    #CREATE separate images of each file_list2 file
    # on the corresponding file_list1 file
    for idx in list(range(len(cmat_list))):
        cmat = cmat_list[idx]

        if lower_triangle:
            fig = drawmatrix_channels(cmat, channel_names, size=size,
                                      color_anchor=color_anchor, **kwargs)
        else:
            fig = draw_square_matrix_channels(cmat, channel_names, size=size,
                                              color_anchor=color_anchor,
                                              **kwargs)

        png_fname = 'connectivity_matrix' + str(idx) + '.png'
        png_path = os.path.join(output_dir, png_fname)

        export_figure(fig, png_path, dpi=dpi)

        img_files.append(os.path.basename(png_path))

        plt.close()

    #Create the index.html file with all images
    create_imglist_html(output_dir, img_files)

    return img_files


def draw_square_matrix_channels(in_m, channel_names=None, fig=None,
                                x_tick_rot=None, size=None, cmap=plt.cm.RdBu_r,
                                colorbar=True, color_anchor=None, title=None):
    """
    Copied from nitime.viz import drawmatrix_channels

    Creates a full-matrix or lower-triangle of the matrix of an nxn set of values.
    This is the typical format to show a symmetrical bivariate quantity (such as
    correlation or coherence between two different ROIs).

    @param cmat: nxn array
    with values of relationships between two sets of rois or channels

    @param channel_names (optional): list of strings
    with the labels to be applied to the channels in the input.
    Defaults to '0','1','2', etc.

    @param fig (optional): a matplotlib figure

    @param cmap (optional): a matplotlib colormap
    to be used for displaying the values of the connections on the graph

    @param title (optional): string
    to title the figure (can be like '$\alpha$')

    @param color_anchor (optional): int
    Determine the mapping from values to colormap
        if None, min and max of colormap correspond to min and max of in_m
        if 0, min and max of colormap correspond to max of abs(in_m)
        if (a,b), min and max of colormap correspond to (a,b)

    @return: fig
    a figure object
    """
    import matplotlib.ticker as ticker
    from mpl_toolkits.axes_grid import make_axes_locatable

    N = in_m.shape[0]
    ind = np.arange(N)  # the evenly spaced plot indices

    def channel_formatter(x):
        thisind = np.clip(int(x), 0, N - 1)
        return channel_names[thisind]

    if fig is None:
        fig = plt.figure()

    if size is not None:
        fig.set_figwidth(size[0])
        fig.set_figheight(size[1])

    w = fig.get_figwidth()
    h = fig.get_figheight()

    ax_im = fig.add_subplot(1, 1, 1)

    #If you want to draw the colorbar:
    if colorbar:
        divider = make_axes_locatable(ax_im)
        ax_cb = divider.new_vertical(size="10%", pad=0.1, pack_start=True)
        fig.add_axes(ax_cb)

    #Make a copy of the input, so that you don't make changes to the original
    #data provided
    m = in_m.copy()

    #Extract the minimum and maximum values for scaling of the
    #colormap/colorbar:
    max_val = np.nanmax(m)
    min_val = np.nanmin(m)

    if color_anchor is None:
        color_min = min_val
        color_max = max_val
    elif color_anchor == 0:
        bound = max(abs(max_val), abs(min_val))
        color_min = -bound
        color_max = bound
    else:
        color_min = color_anchor[0]
        color_max = color_anchor[1]

    #The call to imshow produces the matrix plot:
    im = ax_im.imshow(m, origin='upper', interpolation='nearest',
                      vmin=color_min, vmax=color_max, cmap=cmap)

    #Formatting:
    ax = ax_im
    ax.grid(True)
    #Label each of the cells with the row and the column:
    if channel_names is not None:
        if x_tick_rot is None:
            if str(channel_names[0]).isdigit():
                x_tick_rot = 0
            else:
                x_tick_rot = 45

        for i in list(range(m.shape[0])):
            # if i < (m.shape[0] - 1):
            #     ax.text(i, -1, channel_names[i],
            #             rotation=x_tick_rot,
            #             horizontalalignment='right',
            #             verticalalignment='bottom',
            #             fontsize=6)
            if i > 0:
                ax.text(-1, i - 1, channel_names[i],
                        horizontalalignment='right',
                        fontsize=6,
                        linespacing=4.)

        ax.set_axis_off()
        ax.set_xticks(np.arange(N))
        ax.set_xticklabels(channel_names)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(channel_formatter))
        fig.autofmt_xdate(rotation=x_tick_rot)

        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(channel_names)
        #ax.set_ybound([-0.5, N - 0.5])
        #ax.set_xbound([-0.5, N - 1.5])

    #Make the tick-marks invisible:
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(0)

    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0)

    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    #The following produces the colorbar and sets the ticks
    if colorbar:
        #Set the ticks - if 0 is in the interval of values, set that, as well
        #as the maximal and minimal values:
        if min_val < 0:
            ticks = [color_min, min_val, 0, max_val, color_max]
        #Otherwise - only set the minimal and maximal value:
        else:
            ticks = [color_min, min_val, max_val, color_max]

        #This makes the colorbar:
        cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
                          cmap=cmap,
                          norm=im.norm,
                          boundaries=np.linspace(color_min, color_max, 256),
                          ticks=ticks,
                          format='%.2f')

    # Set the current figure active axis to be the top-one, which is the one
    # most likely to be operated on by users later on
    fig.sca(ax)

    return fig


#-------------------------------------------------------------------------------
# Mayavi2 options
#-------------------------------------------------------------------------------
def show_cutplanes (vol, first_idx=10, second_idx=10,
                    first_plane='x_axes', second_plane='y_axes'):
    """
    @param vol: numpy 3d array
    @param first_idx: int
    @param second_idx: int
    @param first_plane: string
    @param second_plane: string
    """
    from mayavi import mlab

    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(vol),
                            plane_orientation=first_plane,
                            slice_index=first_idx,
                            colormap='gray')

    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(vol),
                            plane_orientation=second_plane,
                            slice_index=second_idx,
                            colormap='gray')

    mlab.outline()


def show_dynplane(vol):
    """
    @param vol: numpy 3d array
    """
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(vol)
    mlab.pipeline.iso_surface(src, contours=[vol.min()+0.1*vol.ptp(), ],
                              opacity=0.1)
    #mlab.pipeline.iso_surface(src, contours=[vol.max()-0.1*vol.ptp(), ],)
    mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=10,
                            colormap='gray')


def show_contour(vol):
    """
    @param vol: numpy 3d array
    """
    from mayavi import mlab
    mlab.contour3d(vol, colormap='gray')


def show_render(vol, vmin=0, vmax=0.8):
    """
    @param vol: numpy 3d array
    @param vmin: float
    @param vmax: float
    """
    from mayavi import mlab
    mlab.pipeline.volume(mlab.pipeline.scalar_field(vol), vmin=vmin, vmax=vmax,
                         colormap='gray')
