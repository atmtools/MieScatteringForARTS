#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: refractive_index_of_H2O_vis

Module for calculating the refractive index of water and steam as a 
function of wavelength, temperature, and density. This module provides a 
function to compute the complex refractive index of water based on the 
revised formulation from the Journal of Physical and Chemical Reference 
Data. It checks the validity of input parameters and returns the 
refractive index along with a boolean array indicating valid input 
ranges.

Functions
---------
refractive_index_water(wavelength, T, rho_water):

Created on Fri Mar 11 17:24:39 2022
Author: Manfred Brath
"""


import numpy as np
import warnings


def refractive_index_water(wavelength, T, rho_water):
    """
    Refractive index of water and steam for the optical

    Revised formulation for the Refractive
    Index of Water and Steam as a Function of
    Wavelength, Temperature and Density

    Journal of Physical and Chemical Reference Data 27, 761 (1998);
    https://doi.org/10.1063/1.556029 27, 761


    Parameters
    ----------
    wavelength : ndarray
        Wavelength in µm.
    T : ndarray
        Temperature in K.
    rho_water : ndarray
        Density in kg/m^3.

    Returns
    -------
    n : ndarray
        Complex refractive index.
    n_valid : boolean array
        Indicates where input values are inside fittig range
    """

    # Reference values
    T_star = 273.15  # [K]
    rho_star = 1000  # [kg/m^3]
    lambda_star = 0.589  # [µm]

    # coefficients
    a0 = 0.244257733
    a1 = 9.74634476e-3
    a2 = -3.73234996e-3
    a3 = 2.68678472e-4
    a4 = 1.58920570e-3
    a5 = 2.45934259e-3
    a6 = 0.900704920
    a7 = -1.66626219e-2

    lambda_uv = 0.2292020
    lambda_ir = 5.432937

    # check input
    T_ok = (T > T_star - 12) * (T < T_star + 500)
    rho_ok = (rho_water > 0) * (rho_water < 1060)
    wvl_ok = (wavelength > 0.2) * (wavelength < 1.9)

    if np.sum(np.logical_not(T_ok)) > 0:
        warnings.warn("One or more temperature values outside of fit range")

    if np.sum(np.logical_not(rho_ok)) > 0:
        warnings.warn("One or more density values outside of fit range")

    if np.sum(np.logical_not(wvl_ok)) > 0:
        warnings.warn("One or more wavelength values outside of fit range")

    n_valid = np.asarray(T_ok * rho_ok * wvl_ok, dtype=bool)

    # normalize input variables
    T_bar = T / T_star
    rho_bar = rho_water / rho_star
    lambda_bar = wavelength / lambda_star

    # set up right hands side of eq A1
    term1 = a0
    term2 = a1 * rho_bar
    term3 = a2 * T_bar
    term4 = a3 * lambda_bar**2 * T_bar
    term5 = a4 / lambda_bar**2
    term6 = a5 / (lambda_bar**2 - lambda_uv**2)
    term7 = a6 / (lambda_bar**2 - lambda_ir**2)
    term8 = a7 * rho_bar**2

    rhs = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

    # Multiply rhs with normalize density
    a = rho_bar * rhs
    a = a.astype(complex)

    # solve Eq A1 after n
    num = np.sqrt(-2 * a - 1)
    denom = np.sqrt(a - 1)

    # refractive index
    n = num / denom

    return n, n_valid


if __name__ == "__main__":

    T = np.arange(225, 280)  # [K]
    wavelength = np.linspace(0.150, 0.600, 16)  # [µm]
    rho_water = 1000  # [kg/m^3]

    WW, TT = np.meshgrid(wavelength, T)

    n, n_valid = refractive_index_water(WW, TT, rho_water)
