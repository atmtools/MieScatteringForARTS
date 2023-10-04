#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:04:18 2021

@author: u242031
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

# =============================================================================
#
# =============================================================================

def nearest(x0, x):

    idx = np.argmin((x - x0) ** 2)

    xn = x[idx]

    return xn, idx


def getOverlap(a, b):
    '''
    Function to calculate the overlap between two ranges given
    by the edges of  each range.

    Args:
        a: vector
            Edges of range 1.
        b:  vector
            Edges of range 2.

    Returns:
        overlap: float
            overlap

    '''

    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


#%%============================================================================
# plotting
# =============================================================================

def cmap_matlab_lines():

    cmap=np.array([[0      , 0.44701, 0.74101, 1],
                   [0.85001, 0.32501, 0.09801, 1],
                   [0.92901, 0.69401, 0.12501, 1],
                   [0.49401, 0.18401, 0.55601, 1],
                   [0.46601, 0.67401, 0.18801, 1],
                   [0.30101, 0.74501, 0.93301, 1],
                   [0.63501, 0.07801, 0.18401, 1]])

    return cmap


def subplot_dimensions(nop, ratio=1):
    '''
    function to create automized subplot dimensions

    Args:
        nop (int): Number of subplots.
        ratio (float, optional): Ration between row an cols. Defaults to 1.

    Returns:
        rows (int): Rows of subplots.
        cols (int): Columns of subplots.

    '''

    if ratio >=1:
        cols=np.floor(np.sqrt(nop*ratio))
    else:
        cols=np.ceil(np.sqrt(nop*ratio))

    rows=np.ceil(nop/cols)

    return int(rows), int(cols)


def default_figure(rows,columns,width_in_cm=29.7,height_in_cm=20.9,sharey='all', sharex='all' ):

    fig, ax = plt.subplots(rows,columns,sharey=sharey, sharex=sharex)
    fig.set_size_inches(width_in_cm/2.54,h=height_in_cm/2.54)

    if np.size(ax)==1:
        ax.set_prop_cycle(color=cmap_matlab_lines())

    elif len(np.shape(ax))==1:

        for c in range(columns*rows):
            ax[c].set_prop_cycle(color=cmap_matlab_lines())

    else:

        for r in range(rows):
            for c in range(columns):
                ax[r,c].set_prop_cycle(color=cmap_matlab_lines())

    return fig, ax


def default_figure_gridspec(rows, columns, width_in_cm=29.7, height_in_cm=20.9):
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(width_in_cm / 2.54, h=height_in_cm / 2.54)
    spec = gridspec.GridSpec(ncols=columns, nrows=rows, figure=fig, hspace=0.35)

    return fig, spec


def pcolor_plot(fig,ax,x,y,Z,minZ,maxZ, font_name='cmr10',xlabel=None,ylabel=None,
                cmap=None, title=None, cbar_label=None, shading='auto'):



    if cmap==None:
        cmap=plt.get_cmap("viridis")

    #make plot and add colorbar
    pcm=ax.pcolormesh( x, y, Z, shading=shading,cmap=cmap,vmin=minZ, vmax=maxZ)
    pcm.set_rasterized(True)
    cbar=fig.colorbar(pcm, ax=ax, shrink=1)

    #Set font
    font = FontProperties()
    font.set_name('cmr10')

    #set the Make-Up and writings
    ax.set_title(title,fontproperties=font)
    ax.title.set_fontsize(10)

    ax.set_xlabel(xlabel,fontproperties=font)
    ax.set_ylabel(ylabel,fontproperties=font)

    cbar.set_label(cbar_label,fontproperties=font)
    # set_tick_font(ax,font_name)
    # set_tick_font(cbar.ax,'cmr10')

    ax.grid(which='both',linestyle=':', linewidth=0.25)
    ax.axes.axes.tick_params(direction='in',which='both')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    return fig, ax, pcm, cbar

def scatter_plot(fig, ax, x, y, c=None, font_name=None,xlabel=None,ylabel=None,
                cmap=None, title=None, cbar_label=None, **kwargs):

    if cmap is None:
        cmap=plt.get_cmap('speed',64)

    #Set font
    if font_name is None:
        font_name='Bitstream Vera Sans'

    font = FontProperties()
    font.set_name(font_name)


    sca=ax.scatter(x, y, c=c, cmap=cmap,**kwargs)
    sca.set_rasterized(True)

    if cbar_label is not None:
        cbar=fig.colorbar(sca, ax=ax, shrink=1)
        cbar.set_label(cbar_label,fontproperties=font)
    else:
        cbar=None

    #set the Make-Up and writings
    ax.set_title(title,fontproperties=font)
    ax.title.set_fontsize(8)

    ax.set_xlabel(xlabel,fontproperties=font)
    ax.set_ylabel(ylabel,fontproperties=font)


    # set_tick_font(ax,font_name)
    # set_tick_font(cbar.ax,'cmr10')

    ax.grid(which='both',linestyle=':', linewidth=0.25)
    ax.axes.axes.tick_params(direction='in',which='both')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    return fig, ax, sca, cbar


def scatter3d_plot(fig, ax, x, y, z, c=None, font_name='cmr10',
                   xlabel=None,ylabel=None,zlabel=None,
                cmap=None, title=None, cbar_label=None, alpha=1.):

    if cmap is None:
        cmap=plt.get_cmap('speed',64)

    sca=ax.scatter(x, y, z, c=c, cmap=cmap, alpha = alpha)
    sca.set_rasterized(True)
    cbar=fig.colorbar(sca, ax=ax, shrink=1)

    #Set font
    font = FontProperties()
    font.set_name('cmr10')

    #set the Make-Up and writings
    ax.set_title(title,fontproperties=font)
    ax.title.set_fontsize(10)

    ax.set_xlabel(xlabel,fontproperties=font)
    ax.set_ylabel(ylabel,fontproperties=font)
    ax.set_zlabel(zlabel,fontproperties=font)

    cbar.set_label(cbar_label,fontproperties=font)

    maxis=np.max(np.array([np.max(np.abs(x)),np.max(np.abs(y)),np.max(np.abs(z))]))

    ax.set_xlim(xmin=-maxis,xmax=maxis)
    ax.set_ylim(ymin=-maxis,ymax=maxis)
    ax.set_zlim(zmin=-maxis,zmax=maxis)


    return fig, ax, sca, cbar


def default_plot_format(ax, font_name=None):
    '''
    simple function to define basic properties of a plot

    Args:
        ax: matplotlib axis object
            axis object

        font_name: str
            font name

    Returns:
        ax: matplotlib axis object
            axis object

        font: font properties object
            font properties

    '''

    font = FontProperties()
    if font_name is not None:
        font.set_name(font_name)

    # ax.set_prop_cycle(color=cmap_matlab_lines())

    ax.grid(which='both', linestyle=':', linewidth=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axes.tick_params(direction='in', which='both')

    return ax, font