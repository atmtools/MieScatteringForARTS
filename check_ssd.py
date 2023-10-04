#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:54:10 2022

@author: u242031
"""

import os
import numpy as np
import glob

from pyarts import xml

import generate_miescattering_data as gmd





# =============================================================================
# paths/constants
# =============================================================================

material='H2O_liquid'

datafolder='scattering_data/visible/'


# =============================================================================
#
# =============================================================================

list_scatt=glob.glob(os.path.join(datafolder, material,'*um.xml'))

radii=[float(os.path.basename(entry)[11:22]) for entry in list_scatt]

sort_idx=np.argsort(radii)

list_scatt=[list_scatt[idx] for idx in sort_idx]
radii=np.array([radii[idx] for idx in sort_idx])



for file_i in list_scatt:


    ssd=xml.load(file_i)
    smd=xml.load(file_i[0:-3]+'meta.xml')

    extmat=ssd.ext_mat_data
    absvec=ssd.abs_vec_data
    phamat=ssd.pha_mat_data
    za_grid=ssd.za_grid
    f_grid=ssd.f_grid


    for i, f_i in enumerate(f_grid):


        Cext=extmat[i,0,0,0,0]
        Cabs=absvec[i,0,0,0,0]
        Csca_data=extmat[i,0,0,0,0]-absvec[i,0,0,0,0]
        Csca=np.trapz(np.squeeze(phamat[i, 0, :, 0, 0, 0, 0]),-np.cos(za_grid*np.pi/180))*2*np.pi
        Csca2=np.trapz(np.squeeze(phamat[i, 0, :, 0, 0, 0, 0])*np.sin(za_grid*np.pi/180),za_grid*np.pi/180)*2*np.pi

        dev=abs(Csca2 - Csca_data) / extmat[i,0,0,0,0]

        print(f'{file_i}')
        print(f'frequency: {f_i/1e12}THz')
        print(f'Csca_data {Csca_data}')
        print(f'Csca {Csca}')
        print(f'Csca2 {Csca2}')
        print(f'albedo deviation {dev}')
        print(f'Cabs {Cabs}')
        print(f'Cext {Cext}\n')
        print('==============================================================')



    print('bla')






