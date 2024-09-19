#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: aux_function

This module provides utility functions for creating plots with customized
settings and color maps, similar to MATLAB's 'lines' colormap. It includes
functions for determining subplot dimensions, setting default figure formats,
and applying a specific color cycle to plots.

Functions:
- cmap_matlab_lines: Generates a colormap similar to MATLAB's 'lines'.
- subplot_dimensions: Calculates the number of rows and columns for subplots.
- default_figure: Creates a default figure with specified dimensions and properties.
- default_plot_format: Sets basic properties for a given matplotlib axis.

Created on Wed Apr 21 14:04:18 2021
@author: Manfred Brath


"""


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



#%%============================================================================
# plotting
# =============================================================================

def cmap_matlab_lines():
    """
    Generates a colormap similar to the MATLAB 'lines' colormap.
    Returns:
        numpy.ndarray: A 2D array representing the RGBA values of the colormap.
    """


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