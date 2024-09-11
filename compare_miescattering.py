#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:20:57 2022

@author: u242031
"""
import os
import numpy as np
import glob

import pyarts.arts as arts
import pyarts.xml as xml

import refractive_index_of_H2O_segelstein as refs
import refractive_index_of_H2O_Liebe93 as refl
import aux_function as af
import generate_miescattering_functions as gmd


# =============================================================================
#%% paths/constants
# =============================================================================

material='H2O_liquid'

home=os.environ["HOME"]

arts_scat_data_folder=f'{home}/.cache/arts/arts-xml-data-2.6.6/scattering/'





# =============================================================================
#%% go through data
# =============================================================================


list_scatt=glob.glob(os.path.join(arts_scat_data_folder, material,'*um.xml'))

radii=[float(os.path.basename(entry)[11:22]) for entry in list_scatt]

sort_idx=np.argsort(radii)

list_scatt=[list_scatt[idx] for idx in sort_idx]
radii=np.array([radii[idx] for idx in sort_idx])



print('\nParticle list --------------------------------------------------\n')
for idx, entry in enumerate(list_scatt):

    name=os.path.basename(entry)

    print(f'{name} == {idx}')


part_idx=int(input('\nchoose particle ==>>'))

print('\n selected particle:')
print(os.path.basename(list_scatt[part_idx]))

pid=os.path.basename(list_scatt[part_idx][:-3])


# =============================================================================
#%% load data and calculate
# =============================================================================


ssd_arts=xml.load(list_scatt[part_idx])
smd_arts=xml.load(list_scatt[part_idx][:-3]+'meta.xml')



f_grid=ssd_arts.f_grid[0:]
t_grid=np.array([np.max(ssd_arts.T_grid)])
za_grid=ssd_arts.za_grid
droplet_radius=smd_arts.diameter_volume_equ/2
rho_water=1000.

#speed of light
c0=arts.constants.c #[m/s]

#size parameter
x=2*np.pi*droplet_radius*f_grid/c0


#refractive index
m_rs,m_is = refs.refactive_index_water_segelstein(f_grid)
ms=m_rs-m_is*1j
ml=np.sqrt(refl.eps_water_liebe93(f_grid, t_grid))
ml=np.conj(ml)


ssd_mie, smd_mie, P_coeffs = gmd.calc_arts_scattering_data(f_grid,t_grid,za_grid, droplet_radius, [1], ml, rho_water )

# =============================================================================
#%% evaluate
# =============================================================================

#calculate scattering coefficient
sigma_sca_arts_from_int=np.zeros(len(f_grid))
sigma_sca_mie_from_int=np.zeros(len(f_grid))

for i in range(len(f_grid)):
    sigma_sca_arts_from_int[i]=np.trapz(ssd_arts.pha_mat_data[i,-1,:,0,0,0,0], -np.cos(za_grid*np.pi/180))*2*np.pi
    sigma_sca_mie_from_int[i] =np.trapz(ssd_mie.pha_mat_data[i,0,:,0,0,0,0] , -np.cos(za_grid*np.pi/180))*2*np.pi


sigma_sca_arts=ssd_arts.ext_mat_data[:,-1,0,0,0]-ssd_arts.abs_vec_data[:,-1,0,0,0]
sigma_sca_mie=ssd_mie.ext_mat_data[:,0,0,0,0]-ssd_mie.abs_vec_data[:,0,0,0,0]

# =============================================================================
#%% plot cross sections
# =============================================================================
fig, ax=af.default_figure(1, 3, sharey=True)
ax[0].loglog(f_grid*1e-9,ssd_mie.ext_mat_data[:,0,0,0,0],'+-',label='Miepython SSD')
ax[0].loglog(f_grid*1e-9,ssd_arts.ext_mat_data[:,-1,0,0,0],'x--',label='ARTS SSD')
ax[0].set_xlabel('frequency / GHz')
ax[0].set_ylabel('extinction cross section / m$^2$')
ax[0].legend()
ax[0],_=af.default_plot_format(ax[0])

ax[1].loglog(f_grid*1e-9,sigma_sca_mie_from_int,'-',label='Miepython SSD integ.')
ax[1].loglog(f_grid*1e-9,sigma_sca_arts_from_int,'--',label='ARTS SSD integ.')
ax[1].loglog(f_grid*1e-9,sigma_sca_mie,'o',label='Miepython SSD')
ax[1].loglog(f_grid*1e-9,sigma_sca_arts,'s',label='ARTS SSD')
ax[1].set_xlabel('frequency / GHz')
ax[1].set_ylabel('scattering cross section / m$^2$')
ax[1].legend()
ax[1],_=af.default_plot_format(ax[1])


ax[2].loglog(f_grid*1e-9,ssd_mie.abs_vec_data[:,0,0,0,0],'+-',label='Miepython SSD')
ax[2].loglog(f_grid*1e-9,ssd_arts.abs_vec_data[:,-1,0,0,0],'x--',label='ARTS SSD')
ax[2].set_xlabel('frequency / GHz')
ax[2].set_ylabel('absorption cross section / m$^2$')
ax[2],_=af.default_plot_format(ax[2])

fig.suptitle(rf'{pid} {material}')


# =============================================================================
#%% plot phase matrix elements
# =============================================================================

#Number of frequencies to plot
nof=6

#number of phase matrix elements
nop=len(P_coeffs)

freq_idx=np.linspace(0, len(f_grid)-1, nof)
freq_idx=freq_idx.astype('int')


fig1, ax1=af.default_figure(nop, nof, sharey=False)

for i,fidx_i in enumerate(freq_idx):
    for j,coeff in enumerate(P_coeffs):
        
        if j==0:
            X_mie=ssd_mie.pha_mat_data[fidx_i,0,:,0,0,0,0]
            X_arts=ssd_arts.pha_mat_data[fidx_i,-1,:,0,0,0,0]
            ax1[j,i].set_ylabel(f'{P_coeffs[j]} / m$^2$')
            ax1[j,i].semilogy(za_grid,X_mie,label='Miepython SSD')            
            ax1[j,i].semilogy(za_grid,X_arts,'--',label='ARTS SSD')                        
            if i==0 and j==0:
                ax1[j,i].legend()
            ax1[j,i].set_title(f'x={x[fidx_i]:.4g} -- {f_grid[fidx_i]/1e9:.4g}GHz')               
            
        else:
            X_mie=ssd_mie.pha_mat_data[fidx_i,0,:,0,0,0,j]/ssd_mie.pha_mat_data[fidx_i,0,:,0,0,0,0]
            X_arts=ssd_arts.pha_mat_data[fidx_i,-1,:,0,0,0,j]/ssd_arts.pha_mat_data[fidx_i,-1,:,0,0,0,0]
            ax1[j,i].set_ylabel(f'{P_coeffs[j]}/{P_coeffs[0]}')
            ax1[j,i].plot(za_grid,X_mie,label='Miepython SSD')
            ax1[j,i].plot(za_grid,X_arts,'--',label='ARTS SSD')

        ax1[j,i],_=af.default_plot_format(ax1[j,i])
        
        if j==nop-1:
            ax1[j,i].set_xlabel('$\Theta$ / $^\circ$')


# =============================================================================
# %% plot phase function integral
# =============================================================================



phfct_integral_mie,phfct_integral_mie_test= gmd.integrate_phasefunction_for_testing(ssd_mie)
phfct_integral_arts,phfct_integral_arts_test = gmd.integrate_phasefunction_for_testing(ssd_arts, t_index=-1)


fig, ax=af.default_figure(1, 1)
ax.semilogx(f_grid*1e-9,(phfct_integral_mie/2-1)*100,'-',label='Miepython SSD')
ax.semilogx(f_grid*1e-9,(phfct_integral_arts/2-1)*100,'--',label='ARTS SSD')
ax.semilogx(f_grid*1e-9,(phfct_integral_mie_test/2-1)*100,'o',label='Miepython SSD test')
ax.semilogx(f_grid*1e-9,(phfct_integral_arts_test/2-1)*100,'s',label='ARTS SSD test')
ax.set_xlabel('frequency / GHz')
ax.set_ylabel('Normalization derivation / %')
ax.legend()
ax,_=af.default_plot_format(ax)
