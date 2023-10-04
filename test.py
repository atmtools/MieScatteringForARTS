#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 23:58:57 2022

@author: u242031
"""

import numpy as np

import matplotlib.pyplot as plt

import generate_miescattering_functions as gmf

import aux_function as af

# =============================================================================
# definitions
# =============================================================================

freq=500e12
x=50000

radius=gmf.mie_size_parameter2radius(x,freq)
nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1

#hires grid
za_grid_hr=gmf.create_angular_grid(180001, k=5, unit='deg')
za_rad_hr=za_grid_hr*np.pi/180

#lowres grid
za_grid_lr=gmf.create_angular_grid(1441, k=10, unit='deg')
za_rad_lr=za_grid_lr*np.pi/180


m=1.3-1j*0.0000001

P_hr, sigma_ext, sigma_abs, P_coeffs = gmf.calc_mie_scattering(radius, freq, za_grid_hr, m,
                                                         oversampling=1, verbose=True)


P_lr, _, _, _ = gmf.calc_mie_scattering(radius, freq, za_grid_lr, m,
                                                         oversampling=1, verbose=True)


# =============================================================================
#% show phase matrix
# =============================================================================
rows, cols = af.subplot_dimensions(np.size(P_hr,axis=1),4/3)

fig, ax=af.default_figure(rows, cols, sharey=False)

cnt=-1
for row in range(rows):

    for col in range(cols):

        cnt+=1

        if cnt==0:
            X=P_hr[:,cnt]
            ax[row,col].set_ylabel(f'{P_coeffs[cnt]} / m$^2$')
            ax[row,col].semilogy(za_grid_hr,X)
        else:
            X=P_hr[:,cnt]/P_hr[:,0]
            ax[row,col].set_ylabel(f'{P_coeffs[cnt]}/{P_coeffs[0]}')
            ax[row,col].plot(za_grid_hr,X)


        ax[row,col].set_xlabel('$\Theta$ / $^\circ$')
        ax[row,col],_=af.default_plot_format(ax[row,col])



# =============================================================================
# reconstruct
# =============================================================================


P_int=np.zeros((len(za_grid_hr),6))


cnt=-1
for row in range(rows):

    for col in range(cols):

        cnt+=1

        F_int=gmf.interp1d(za_grid_lr, P_lr[:,cnt],kind='linear')
        P_int[:,cnt]=F_int(za_grid_hr)

        if cnt==0:
            X=P_int[:,cnt]
        else:
            X=P_int[:,cnt]/P_int[:,0]
        ax[row,col].plot(za_grid_hr,X)



Csca_hr=np.trapz(np.squeeze(P_hr[:,0])*np.sin(za_rad_hr),za_rad_hr)*2*np.pi
Csca_lr=np.trapz(np.squeeze(P_lr[:,0])*np.sin(za_rad_lr),za_rad_lr)*2*np.pi
Csca_int=np.trapz(np.squeeze(P_int[:,0])*np.sin(za_rad_hr),za_rad_hr)*2*np.pi
