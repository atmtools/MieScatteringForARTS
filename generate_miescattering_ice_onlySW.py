#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:20:57 2022

@author: u242031
"""
import os
import numpy as np
import time

import pyarts.arts as arts
from pyarts import xml

import generate_miescattering_functions as gmf
import refractive_index_of_ice_warren08 as ref
import aux_function as af



# =============================================================================
#     Definitions
# =============================================================================

#speed of light
c0=arts.constant.c #[m/s]

#angular grid
# za_grid = np.linspace(0,180,721)
za_grid = gmf.create_angular_grid(1441,k=5)

#frequency range
f_min=1.5e14 # Hz =>30 cm Wavelength
f_max=1e15 # Hz => 300 nm Wavelength

#frequency grid
N_f=21
f_grid=np.logspace(np.log10(f_min),np.log10(f_max), N_f) #[Hz]

#temperature grid
t_grid=[303.15] #[K]

#droplet radius
a=np.ceil(10**np.arange(0,1,0.1)*4)/4
b=10**(np.arange(-7,-2.))
droplet_radii=np.sort(np.ravel(np.outer(a,b)))

#remove droplets greater than 5 mm
droplet_radii=droplet_radii[droplet_radii<5e-3]

#material
material='H2O_ice'

#density of ice
rho_water=1000. #kg m^-3

#refractive index
m_r,m_i = ref.refactive_index_ice_warren08(f_grid)
m=m_r-m_i*1j

ref_index_text=('Warren, S. G., and R. E. Brandt\n'
                +'Optical constants of ice from the ultraviolet to the microwave:A revised compilation..\n'
                +'J. Geophys. Res., 113, D14220, doi:10.1029/2007JD009744.\n'
                +'\n'
                +'No temperature dependency. Nominal temperature: 266 K\n')

#plot results ?
plotting=False

#Samples per subdomain
N_sub=1


datafolder=f'scattering_data/shortwave/{material}/'
datafolder_arrayformat=f'scattering_data/shortwave/ArrayFormat/{material}/'
plotfolder=f'plots/shortwave/{material}/'



os.makedirs(datafolder, exist_ok=True)
os.makedirs(datafolder_arrayformat, exist_ok=True)
os.makedirs(plotfolder, exist_ok=True)



# =============================================================================
#   the actual calculation
# =============================================================================


dlog_r=np.mean(np.diff(np.log10(droplet_radii)))
r_sub_fac=10**((np.linspace(1/(2*N_sub),1-1/(2*N_sub), N_sub)-0.5)*dlog_r)

ssd_array=arts.ArrayOfSingleScatteringData()
smd_array=arts.ArrayOfScatteringMetaData()

for i, r_i in enumerate(droplet_radii):

    pid=f'MieSphere_R{r_i*1e6:1.5e}um'

    ssd, smd, P_coeffs = gmf.calc_arts_scattering_data(f_grid,t_grid,za_grid,
                                                       r_i, r_sub_fac, m,
                                                       rho_water,
                                                       ignore_limit=True,
                                                       ref_index_text=ref_index_text)

    f_grid_ssd=ssd.f_grid.value

    if plotting:

        for k, f_k in enumerate(f_grid_ssd):

            if k%10 >0 :
                continue

            identifier=f'{pid}_F{f_k/1e12:.3f}THz_L{c0/f_k*1e9:.3f}nm'

            #calculate size parameter
            x=2*np.pi*r_i*f_k/c0

            ## plot phase matrix
            rows,cols=af.subplot_dimensions(6, ratio=1)
            fig, ax=af.default_figure(rows, cols, sharey=False)

            cnt=-1
            for row in range(rows):

                for col in range(cols):

                    cnt+=1

                    if cnt==0:
                        X=ssd.pha_mat_data[k,0,:,0,0,0,cnt]
                        ax[row,col].set_ylabel(f'{P_coeffs[cnt]} / m$^2$')
                        ax[row,col].semilogy(za_grid,X, linewidth=1.)
                    else:
                        X=ssd.pha_mat_data[k,0,:,0,0,0,cnt]/ssd.pha_mat_data[k,0,:,0,0,0,0]
                        ax[row,col].set_ylabel(f'{P_coeffs[cnt]}/{P_coeffs[0]}')
                        ax[row,col].plot(za_grid,X, linewidth=1.)


                    ax[row,col].set_xlabel('$\Theta$ / $^\circ$')
                    ax[row,col],_=af.default_plot_format(ax[row,col])


            fig.suptitle(f'{identifier}\n {material} --- m = {m[k]:.3g} --- x = {x:.1f}')

            pha_mat_folder=os.path.join(plotfolder,f'PhaMat_{pid}')
            os.makedirs(pha_mat_folder, exist_ok=True)

            plotfilename=(f'PhaMat_{identifier}.pdf')
            fig.savefig(os.path.join(pha_mat_folder,plotfilename))

            af.plt.close(fig)
            time.sleep(0.1)

        print('plotting phase mat done')



        #plot extinction and absorption vector as function of frequency
        fig, ax=af.default_figure(2, 2, sharey=False)
        ax[0,0].loglog(c0/f_grid_ssd*1e9,ssd.ext_mat_data[:,0,0,0,0],'s-')
        ax[0,0].set_xlabel('wavelength / nm')
        ax[0,0].set_ylabel('extinction cross section / m$^2$')
        ax[0,0],_=af.default_plot_format(ax[0,0])

        ax[0,1].loglog(c0/f_grid_ssd*1e9,ssd.abs_vec_data[:,0,0,0,0],'s-')
        ax[0,1].set_xlabel('wavelength / nm')
        ax[0,1].set_ylabel('absorption cross section / m$^2$')
        ax[0,1],_=af.default_plot_format(ax[0,1])

        ax[1,0].loglog(c0/f_grid*1e9,m.real,'s-')
        ax[1,0].set_xlabel('wavelength / nm')
        ax[1,0].set_ylabel('refraction index real part')
        ax[1,0],_=af.default_plot_format(ax[1,0])

        ax[1,1].loglog(c0/f_grid*1e9,abs(m.imag),'s-')
        ax[1,1].set_xlabel('wavelength / nm')
        ax[1,1].set_ylabel('refraction index imaginary part')
        ax[1,1],_=af.default_plot_format(ax[1,1])

        fig.suptitle(rf'{pid} {material}')
        plotfilename=(f'optical_properties_{pid}.pdf')
        fig.savefig(os.path.join(plotfolder,plotfilename))

        af.plt.close(fig)
        time.sleep(0.1)

        print('plotting crossections done')



        ##plot phase function normalization derivation
        phfct_integral_mie, _ = gmf.integrate_phasefunction_for_testing(ssd)

        fig, ax=af.default_figure(1, 1)
        ax.semilogx(c0/f_grid_ssd*1e9,(phfct_integral_mie/2-1)*100,'+-',label='Miepython SSD')
        ax.set_xlabel('wavelength / nm')
        ax.set_ylabel('Normalization derivation / %')
        ax.legend()
        ax,_=af.default_plot_format(ax)

        fig.suptitle(rf'{pid} {material}')
        plotfilename=(f'phasefunction_derivation_{pid}.pdf')
        fig.savefig(os.path.join(plotfolder,plotfilename))

        af.plt.close(fig)
        time.sleep(0.1)

        print('plotting normalization mat done')


    #save scattering data
    print(pid)

    ssd.savexml(os.path.join(datafolder,pid+'.xml'), format='binary' )
    smd.savexml(os.path.join(datafolder,pid+'.meta.xml'))

    ssd_array.append(ssd)
    smd_array.append(smd)


    print(f'done with radius {r_i*1e6} µm')


#save it as ArrayOfArrayOfSingleScatteringData...
ssd_array_sqr=arts.ArrayOfArrayOfSingleScatteringData()
smd_array_sqr=arts.ArrayOfArrayOfScatteringMetaData()

ssd_array_sqr.append(ssd_array)
smd_array_sqr.append(smd_array)


xml.save(ssd_array_sqr, os.path.join(datafolder_arrayformat,f'MieSpheres_{material}.data.xml'),format='binary')
xml.save(smd_array_sqr, os.path.join(datafolder_arrayformat,f'MieSpheres_{material}.meta.xml'),format='binary')






