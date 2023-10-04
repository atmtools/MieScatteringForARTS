#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:20:57 2022

@author: u242031
"""
import os
import warnings
import numpy as np
import miepython

from scipy.signal import convolve
from scipy.interpolate import interp1d
from datetime import date
import time

import pyarts.arts as arts

import refractive_index_of_H2O_segelstein as ref
import aux_function as af


# =============================================================================
# define functions
# =============================================================================

def create_angular_grid(N,k=0.1, unit='deg'):
    '''
    Creates an angular grid for different units

    Parameters
    ----------
    N : int
        Number of grid points.
    k : float, optional
        shapefactor. For 0<k<<1, the resulting grid is quasi linear.
                     Otherwise, the grid is dtiributed like a sigmoid. With
                     increasing k the start and the end region are higher sampled.
                     The default is 0.1.
    unit : str, optional
        Define the unit of the grid. The default is 'deg'.


    Returns
    -------
    y : ndarray
        The angular grid.

    '''

    if k==0:
        raise ValueError('k must be >0')

    x=np.linspace(-1,1,N)

    y=(1+np.exp(-k*x))**(-1)

    #Normalize so that y is from zero to one
    y=y-y.min()
    y/=y.max()


    if unit=='deg':
        y*=180
    elif unit=='rad':
        y*=np.pi
    elif unit=='cosTheta':
        y=-(y-0.5)*2
    else:
        raise ValueError('desired unit is not supported.\n'+
                         'Only "deg", "rad" and "cosTheta" are supported.')

    return y


def S1S2ToPhaseMatSphere(S11,S22):
    '''
    Calculate phase matrix in ARTS format from scattering amplitude matrix
    for a spherical particle. We use Mishchenko's notation.

    Args:
        S11 (1d array): scattering matrix amplitude S11.
        S22 (1d array): scattering matrix amplitude S22.

    Returns:
        P (ndarray): Phase matrix for spherically symmetric particles in ARTS format.
        P_coeffs (Array of strings): Corresponding matrix element names.

    '''

    assert len(S11)==len(S22),"S11, S22 must be of same length"

    P=np.zeros((len(S11),6))

    P[:,0]=(np.abs(S11)**2+np.abs(S22)**2)/2.
    P[:,1]=(np.abs(S11)**2-np.abs(S22)**2)/2.
    P[:,2]=P[:,0]
    P[:,3]=np.real(S11*np.conj(S22))
    P[:,4]=-np.imag(S11*np.conj(S22))
    P[:,5]=P[:,3]

    P_coeffs=['F11','F12','F22','F33','F34','F44']

    return P, P_coeffs



def calc_mie_scattering(radius, frequency, za_grid, m, smoothing_window_size=0.,
                        oversampling=10, verbose=False ):
    '''
    Calculates Mie scattering phase matrix (ARTS convention), extinction and
    absorption cross section.

    Args:
        radius (Float): Particle radius in m.
        frequency (Float): Frequency in Hz.
        za_grid (1d array): Desired angular grid.
        m (complex128): Complex refraction index.
        smoothing_window_size (float): Smoothing window size in deg.
        oversampling (int): Oversampling, intened for use in combination with smooting.
        verbose (TYPE, optional): Verbosity. Defaults to False.

    Returns:
        P (ndarray): Phase matrix using ARTS .
        sigma_ext (float): Extinction cross section.
        sigma_abs (TYPE): Absorption cross section.
        P_coeffs (Array of strings): Corresponding matrix element names.
        x (float): Size parameter.

    '''

    #speed of light
    c0=arts.constant.c #[m/s]

    #Create calculation grid
    index=np.arange(0,len(za_grid))
    F_za_int=interp1d(index,za_grid)
    index_os=np.linspace(index.min(),index.max(),len(za_grid)*oversampling)
    za_grid_hr=F_za_int(index_os)

    za_rad_hr=za_grid_hr*np.pi/180
    za_rad=za_grid*np.pi/180


    #cosine of angular grids
    mu = np.cos(za_rad)
    mu_hr = np.cos(za_rad_hr)

    #Size parameter
    x=2*np.pi*radius*frequency/c0

    #geeometric cross section
    sigma_geo=np.pi*radius**2 #[m^2]


    #check if imag(m) <=0
    if np.imag(m)>0:
        m=np.conj(m)
        warnings.warn('\nImaginary part of m needs to be <=0, but it is >0.\n'+
                      'So, I changed the sign of it. Maybe you should check\n'+
                      'your refractive index')

    #scattering amplitude and efficiencies
    #here we need to distinguish between small and big spheres.
    #miepython doc :"...because in the small particle or Rayleigh limit
    #the Mie formulas become ill-conditioned. The method was taken from
    #Wiscombe’s paper and has been tested for several complex indices of refraction.
    #Wiscombe uses this when np.abs(m)*x<=0.1."
    if np.abs(m)*x<=0.1:
        S1_hr,S2_hr=miepython.miepython._small_mie_S1_S2(m, x, mu_hr)
        qext, qsca, qback, g = miepython.miepython._small_mie(m, x)
    else:
        S1_hr,S2_hr=miepython.mie_S1_S2(m, x, mu_hr)
        qext, qsca, qback, g = miepython.mie(m,x)



    #map scattering amplitudes to ARTS phase mat of total random particles
    #S1 of Bohren and Huffman is S22 of Mishchenko
    #S2 of Bohren and Huffman is S11 of Mishchenko
    P_hr, P_coeffs =S1S2ToPhaseMatSphere(S2_hr,S1_hr)

    #normalize
    P_hr*=(qext/qsca)



    if smoothing_window_size>0:
        #smooth scattering amplitude to get rid of ripples
        window_half_size=smoothing_window_size/2 #[deg]
        dza=np.mean(np.diff(za_grid_hr)) #[deg]
        N_win=int(2*np.ceil(window_half_size/dza)+1)

        kernel=np.ones(N_win)
        kernel=kernel/np.sum(kernel)

        shift=int((len(kernel)-1)/2)

        P_sm=np.zeros(np.shape(P_hr))
        for i in range(np.size(P_hr,axis=1)):
            p_temp=convolve(P_hr[:,i], kernel, mode='full', method='auto')

            p_temp=p_temp[shift:-shift]

            #at the forward and backward peak the smoothed version should match the unsmoothed version
            w=np.linspace(1,0,N_win)

            #make a smooth transition
            p_temp[0:N_win]=w*P_hr[0:N_win,i]+(1-w)*p_temp[0:N_win]
            p_temp[-N_win:]=(1-w)*P_hr[-N_win:,i]+w*p_temp[-N_win:]

            P_sm[:,i]=p_temp

        #Interpolate on input grid
        F_int=interp1d(za_grid_hr, P_sm, axis=0)



    else:

        #Interpolate on input grid
        F_int=interp1d(za_grid_hr, P_hr, axis=0)

    P=F_int(za_grid)

    #assure that the integral of the phase function (P[:,0]) over 4*pi is one.
    # P/=np.trapz(P[:,0], x=-mu)*2*np.pi
    P/=np.trapz(P[:,0]*np.sin(za_rad), x=za_rad)


    #Normalize it to ARTS convention
    P=P*qsca*sigma_geo

    sigma_sca_int=np.trapz(P[:,0]*np.sin(za_rad), x=za_rad)*2*np.pi


    if verbose:
        test=np.trapz((np.abs(S1_hr)**2+np.abs(S2_hr)**2)*np.sin(za_rad_hr), x=za_rad_hr)*np.pi*qext/qsca
        print(f'Int(|S1|^2+|S2|^2 dOmega = {test}')

        P_test=np.trapz(P[:,0], x=-mu)*2*np.pi
        print(f'Int P dOmega = {P_test}')
        print(f'sigma_sca = {qsca*sigma_geo}\n')


    #absorption cross section
    sigma_abs=(qext-qsca)*sigma_geo
    # sigma_abs=sigma_ext-sigma_sca_int

    #extinction cross section
    #As the scattering cross section from the phase function is not exactly qsca*sigm_geo
    #due to the finite angular resolution, we need to calculate the extinction cross section,
    #from the scattering cross section of the phase function and the absorption cross section.
    sigma_ext=sigma_sca_int+sigma_abs


    if sigma_abs<0:
        raise ValueError('negative absorption')
    if sigma_ext<0:
        raise ValueError('negative extinction')

    return P, sigma_ext, sigma_abs, P_coeffs



def calc_arts_scattering_data(f_grid,t_grid,za_grid, droplet_radius, r_sub_fac, m, rho, ignore_limit=False ):


    description=("Spherical particle of liquid water generated using\n"+
                 f"miepython and python by Manfred Brath, {date.today()}\n")
    source='miepython 2.2.3: https://pypi.org/project/miepython/'

    ref_index_text=('Segelstein, David J.\n'
               +'The complex refractive index of water / by David J. Segelstein.  1981.\n'
               +'ix, 167 leaves : ill. ; 29 cm.\n'
               +'Thesis (M.S.)--Department of Physics. University of\n'
               +'Missouri-Kansas City, 1981.\n'
               +'\n')

    #speed of light
    c0=arts.constant.c #[m/s]

    #calculate size parameter
    x=2*np.pi*droplet_radius*f_grid/c0

    #select only size parameter x < 10000
    if ignore_limit:
        logic_x=np.ones(len(x),dtype=bool)
    else:
        logic_x=x<10000

    #angular grid in rad
    za_rad=za_grid*np.pi/180


    pid=f'MieSphere_R{droplet_radius*1e6:1.5e}um'

    pha_mat=np.zeros((len(f_grid[logic_x]),len(t_grid), len(za_grid), 1, 1, 1, 6 ))
    ext_mat=np.zeros((len(f_grid[logic_x]),len(t_grid),  1, 1, 1 ))
    abs_vec=np.zeros((len(f_grid[logic_x]),len(t_grid),  1, 1, 1 ))

    refr_index=''

    N_sub=len(r_sub_fac)

    for k, f_k in enumerate(f_grid[logic_x]):

        if np.isnan(m[k]):
            P_coeffs=[]
            continue

        refr_index+=f'm={m[k]}\n'


        if x[k]<10000 or ignore_limit:
        # if x[k]>0:

            print(pid+'\n-------------------------------------------------------')
            print(f'size parameter x ={x[k]}')

            #allocate
            P=np.zeros((len(za_grid),6))
            sigma_ext=0.
            sigma_abs=0.


            #average over subdomain
            for j in range(N_sub):

                r_ij=droplet_radius*r_sub_fac[j]

                P_j, sigma_ext_j, sigma_abs_j, P_coeffs = calc_mie_scattering(r_ij, f_k,
                                                                           za_grid,
                                                                           m[k],
                                                                           smoothing_window_size=0.,
                                                                           oversampling=1,
                                                                           verbose=True)

                P+=P_j
                sigma_ext+=sigma_ext_j
                sigma_abs+=sigma_abs_j

            P/=N_sub
            sigma_ext/=N_sub
            sigma_abs/=N_sub

            P=P.reshape((np.shape(pha_mat[k, 0, :, 0, 0, 0, :])))

            pha_mat[k, 0, :, 0, 0, 0, :] =P
            ext_mat[k, 0, 0, 0, 0] = sigma_ext
            abs_vec[k, 0, 0, 0, 0] = sigma_abs


            ##tests
            Csca_data=sigma_ext-sigma_abs
            Csca=np.trapz(np.squeeze(pha_mat[k, 0, :, 0, 0, 0, 0])*np.sin(za_rad),za_rad)*2*np.pi
            dev=abs(Csca - Csca_data) / sigma_ext

            print(f'Csca_data {Csca_data}')
            print(f'Csca {Csca}')
            print(f'albedo deviation {dev}\n')


            print(f'done with frequency {f_k/1e12} THz')
            print(f'done with wavelength {c0/f_k*1e9} nm')
            print('-------------------------------------------------------\n')

    #size description
    min_size=droplet_radius*r_sub_fac[0]
    max_size=droplet_radius*r_sub_fac[-1]
    size_description=(f'Geometric mean of the scattering properties of {N_sub} \n'
                     +'uniformly distributed liquid water spheres between \n'
                     +f'{min_size*1e6:.2g} um and {max_size*1e6:.2g} um')

    #single scattering data
    ssd=arts.SingleScatteringData()
    ssd.T_grid=t_grid
    ssd.aa_grid=[]
    ssd.f_grid=f_grid[logic_x]
    ssd.ptype=arts.PType(100)
    ssd.za_grid=za_grid
    ssd.description=description+f'mean particle size:{droplet_radius*1e6:.5f}µm\n'+size_description


    #scattering meta data
    smd=arts.ScatteringMetaData()
    smd.description=description+f'mean particle size:{droplet_radius*1e6:.5f}µm\n'+size_description
    smd.diameter_area_equ_aerodynamical=2*droplet_radius
    smd.diameter_max=2*droplet_radius
    smd.diameter_volume_equ=2*droplet_radius
    smd.mass=4/3*np.pi*droplet_radius**3*rho
    smd.refr_index=ref_index_text
    smd.source=source


    #store data in objects
    ssd.pha_mat_data=pha_mat
    ssd.ext_mat_data=ext_mat
    ssd.abs_vec_data=abs_vec

    return ssd, smd, P_coeffs


def integrate_phasefunction(ssd,t_index=0):

    phfct_integral=np.ones(len(ssd.f_grid))*2
    phfct_integral_test=np.ones(len(ssd.f_grid))*2

    for i,f_i in enumerate(ssd.f_grid):



        sca=ssd.ext_mat_data[i,t_index,0,0,0]-ssd.abs_vec_data[i,t_index,0,0,0]

        if sca>0:



            phfct=ssd.pha_mat_data[i,t_index,:,0,0,0,0]*4*np.pi/sca
            za_rad=ssd.za_grid*np.pi/180

            kernel=(phfct[0:-1]*np.sin(za_rad[0:-1])+phfct[1:]*np.sin(za_rad[1:]))/2*(za_rad[1:]-za_rad[0:-1])


            phfct_integral[i]=np.trapz(phfct*np.sin(za_rad),za_rad)
            phfct_integral_test[i]=np.sum(kernel)

        print(f'f({i})={f_i/1e12}THz')

    return phfct_integral, phfct_integral_test


# =============================================================================
#
# =============================================================================

if __name__ == "__main__":


# =============================================================================
#     Definitions
# =============================================================================

    #speed of light
    c0=arts.constant.c #[m/s]

    #angular grid
    # za_grid = np.linspace(0,180,721)
    za_grid = create_angular_grid(1441,k=5)

    #wavelength
    # lamb=np.linspace(2000,400,17)*1e-9 #[m]
    # lamb=np.linspace(3e-1,3e-4,17) #[m]

    # lamb_min=3e-4 #[m]
    # lamb_max=3e-1 #[m]
    lamb_min=400e-9 #[m]
    lamb_max=2000e-9 #[m]

    f_min=c0/lamb_max
    f_max=c0/lamb_min

    #frequency grid
    N_f=35
    f_grid=np.linspace(f_min,f_max, N_f) #[Hz]

    #temperature grid
    t_grid=[303.15] #[K]

    #droplet radius
    # droplet_radii=np.logspace(-6,-2,5)
    a=np.array([1.,2.,4.,8.])
    b=10**(np.arange(-7,-2.))
    droplet_radii=np.sort(np.ravel(np.outer(a,b)))

    #material
    material='H2O_liquid'

    #density of water
    rho_water=1000. #kg m^-3

    #refractive index
    m_r,m_i = ref.refactive_index_water_segelstein(f_grid)
    m=m_r-m_i*1j

    #smooth data
    smoothing_window_size=2.5

    #plot results ?
    plotting=False

    #Samples per subdomain
    N_sub=50


    datafolder=f'scattering_data/visible/{material}/'
    plotfolder=f'plots/visible/{material}/'



    os.makedirs(datafolder, exist_ok=True)
    os.makedirs(plotfolder, exist_ok=True)



# =============================================================================
#   the actual calculation
# =============================================================================


    dlog_r=np.mean(np.diff(np.log10(droplet_radii)))
    r_sub_fac=10**((np.linspace(1/(2*N_sub),1-1/(2*N_sub), N_sub)-0.5)*dlog_r)


    for i, r_i in enumerate(droplet_radii):

        pid=f'MieSphere_R{r_i*1e6:1.5e}um'

        ssd, smd, P_coeffs = calc_arts_scattering_data(f_grid,t_grid,za_grid,
                                                       r_i, r_sub_fac, m,
                                                       rho_water,
                                                       ignore_limit=True)

        f_grid_ssd=ssd.f_grid.value

        if plotting:

            for k, f_k in enumerate(f_grid_ssd):

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
                            ax[row,col].semilogy(za_grid,X)
                        else:
                            X=ssd.pha_mat_data[k,0,:,0,0,0,cnt]/ssd.pha_mat_data[k,0,:,0,0,0,0]
                            ax[row,col].set_ylabel(f'{P_coeffs[cnt]}/{P_coeffs[0]}')
                            ax[row,col].plot(za_grid,X)


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
            phfct_integral_mie, _ = integrate_phasefunction(ssd)

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

        ssd.savexml(os.path.join(datafolder,pid+'.xml'))
        smd.savexml(os.path.join(datafolder,pid+'.meta.xml'))


        print(f'done with radius {r_i*1e6} µm')











