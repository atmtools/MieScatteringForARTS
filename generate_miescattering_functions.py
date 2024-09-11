#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:20:57 2022

@author: u242031
"""

import warnings
import numpy as np
import miepython

from scipy.signal import convolve
from scipy.interpolate import interp1d
from datetime import date


import pyarts.arts as arts

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


def small_mie_S1_S2(m, x, mu):
    """
    Calculate the scattering amplitude functions for small spheres (x<0.1).

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    THIS FUNCTION IS TAKEN FROM MIEPYTHON VERSION 2.2.3
    AS IT IS NOT AVAILABLE IN THE CURRENT VERSION 2.5.4

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate scattering amplitudes

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    m2 = m * m
    m4 = m2 * m2
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    D = m2 + 2 + (1 - 0.7 * m2) * x2
    D -= (8 * m4 - 385 * m2 + 350) * x4 / 1400.0
    D += 2j * (m2 - 1) * x3 * (1 - 0.1 * x2) / 3
    ahat1 = 2j * (m2 - 1) / 3 * (1 - 0.1 * x2 + (4 * m2 + 5) * x4 / 1400) / D
    bhat1 = 1j * x2 * (m2 - 1) / 45 * (1 + (2 * m2 - 5) / 70 * x2)
    bhat1 /= 1 - (2 * m2 - 5) / 30 * x2
    ahat2 = 1j * x2 * (m2 - 1) / 15 * (1 - x2 / 14)
    ahat2 /= 2 * m2 + 3 - (2 * m2 - 7) / 14 * x2

    S1 = 1.5 * x3 * (ahat1 + bhat1 * mu + 5 / 3 * ahat2 * mu)
    S2 = 1.5 * x3 * (bhat1 + ahat1 * mu + 5 / 3 * ahat2 * (2 * mu**2 - 1))

    # norm = sqrt(qext*pi*x**2)
    norm = np.sqrt(np.pi * 6 * x**3 * (ahat1 + bhat1 + 5 * ahat2 / 3).real)
    S1 /= norm
    S2 /= norm

    return [S1, S2]


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
    c0=arts.constants.c #[m/s]

    #Create calculation grid
    index=np.arange(0,len(za_grid))
    F_za_int=interp1d(index,za_grid)
    index_os=np.linspace(index.min(),index.max(),len(za_grid)*oversampling)
    za_grid_hr=F_za_int(index_os)

    za_rad_hr=za_grid_hr*np.pi/180
    za_rad=za_grid*np.pi/180


    #cosine of angular grids
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
        qext, qsca, qback, g = miepython.miepython._small_mie(m, x)

        #Here we use the small sphere approximation for the scattering amplitudes taken from
        #MiePython version 2.2.3
        S1_hr,S2_hr=small_mie_S1_S2(m, x, mu_hr)
    else:        
        qext, qsca, qback, g = miepython.mie(m,x)
        S1_hr,S2_hr=miepython.mie_S1_S2(m, x, mu_hr)


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
    P/=np.trapz(P[:,0]*np.sin(za_rad), x=za_rad)*2*np.pi

   
    #Normalize it to ARTS convention
    P=P*qsca*sigma_geo

    sigma_sca_int=np.trapz(P[:,0]*np.sin(za_rad), x=za_rad)*2*np.pi


    if verbose:
        test=np.trapz((np.abs(S1_hr)**2+np.abs(S2_hr)**2)*np.sin(za_rad_hr), x=za_rad_hr)*np.pi*qext/qsca
        print(f'Int(|S1|^2+|S2|^2 dOmega = {test}')

        P_test=np.trapz(P[:,0]*np.sin(za_rad), x=za_rad)*2*np.pi
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



def calc_arts_scattering_data(f_grid,t_grid,za_grid, droplet_radius, r_sub_fac, m, rho, ignore_limit=False, ref_index_text=''):                              
    """
    Calculate the scattering data for ARTS (Atmospheric Radiative Transfer Simulator) using miepython for spherical particles.

    Parameters:
    - f_grid (array-like): Frequency grid.
    - t_grid (array-like): Temperature grid.
    - za_grid (array-like): Zenith angle grid.
    - droplet_radius (float): Radius of the droplet.
    - r_sub_fac (array-like): Subdomain factor.
    - m (array-like): Refractive index.
    - rho (float): Density of the droplet.
    - ignore_limit (bool, optional): Flag to ignore the size parameter limit. Defaults to False.
    - ref_index_text (str, optional): Reference index text. Defaults to ''.

    Returns:
    - ssd (arts.SingleScatteringData): Single scattering data.
    - smd (arts.ScatteringMetaData): Scattering meta data.
    - P_coeffs (list): List of P coefficients.

    Raises:
    - None

    Notes:
    - This function calculates the scattering data for ARTS using Mie scattering theory. It generates the single scattering data (ssd) and scattering meta data (smd) objects, as well as the list of P coefficients (P_coeffs).
    - The size parameter limit is set to x < 10000 by default, but can be ignored by setting ignore_limit to True.
    - The refractive index is calculated based on the given refractive index values (m).
    - The function prints information about the size parameter, Csca data, Csca, and albedo deviation for each frequency.
    - The size description is generated based on the droplet radius and subdomain factor.
    - The scattering meta data includes the description, diameter area equivalent aerodynamical, maximum diameter, diameter volume equivalent, mass, refractive index, and source.

    References:
    - Mie scattering theory: https://en.wikipedia.org/wiki/Mie_scattering
    - ARTS: https://www.radiativetransfer.org/
    - miepython: https://github.com/scottprahl/miepython/
    """

    description=("Spherical particle of liquid water generated using\n"+
                 f"miepython and python by Manfred Brath, {date.today()}\n")
    
    source=f'miepython {miepython.__version__}: https://github.com/scottprahl/miepython/'

    #speed of light
    c0=arts.constants.c #[m/s]

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


            print(f'done with frequency {f_k/1e12:.4f} THz')
            print(f'done with wavelength {c0/f_k*1e9:.4f} nm')
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


def integrate_phasefunction_for_testing(ssd,t_index=0):
    """
    Calculates the integral of the phase function over the zenith angle grid.

    IMPORTANT: This function is intended for testing purposes only.
    As there is no integration over azimuth the integral should close to 2 and not 4*pi.

    Parameters:
        ssd (object): The scattering data object.
        t_index (int, optional): The index of the time step. Default is 0.
    Returns:
        tuple: A tuple containing two arrays. The first array represents the integral of the phase function
                over the zenith angle grid for each frequency, and the second array represents the integral
                calculated using a numerical approximation.

    """


    phfct_integral=np.ones(len(ssd.f_grid))*np.nan
    phfct_integral_test=np.ones(len(ssd.f_grid))*np.nan

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



def mie_size_parameter(radius, quantity, qtype='frequency'):
    """
    Calculates mie size parameter either from frequency or from wavelength

    Args:
        radius (float or ndarray): Particle radius in m.
        quantity (ndarray float): Frequency in Hz or wavelength in m.
        qtype (string, optional): Define the quantity type. Defaults to 'frequency'.

    Raises:
        ValueError: qtype must either "frequency" or "feom "wavelength".

    Returns:
        TYPE: Size parameter.

    """

    if qtype=='frequency':
        #speed of light
        c0=arts.constants.c #[m/s]

        inv_wavelength=quantity/c0

    elif qtype=='wavelength':
        inv_wavelength=1/quantity
    else:
        raise ValueError('qtype must either "frequency" or "wavelength"')

    return 2*np.pi*radius*inv_wavelength


def mie_size_parameter2radius(x, quantity, qtype='frequency'):
    """
    Calculates radius from mie size parameter and either from frequency or from wavelength

    Args:
        x (float or ndarray): Size parameter.
        quantity (ndarray float): Frequency in Hz or wavelength in m.
        qtype (string, optional): Define the quantity type. Defaults to 'frequency'.

    Raises:
        ValueError: qtype must either "frequency" or "feom "wavelength".

    Returns:
        TYPE: Particle radius in m.

    """

    if qtype=='frequency':
        #speed of light
        c0=arts.constants.c #[m/s]

        wavelength=c0/quantity

    elif qtype=='wavelength':
        wavelength=quantity
    else:
        raise ValueError('qtype must either "frequency" or "wavelength"')

    return x*wavelength/(2*np.pi)


