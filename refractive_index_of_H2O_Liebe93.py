#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: refractive_index_of_H2O_Liebe93

This module provides a function to calculate the complex permittivity of water 
using the Liebe 1993 model. It allows for the evaluation of permittivity over a 
range of frequencies and temperatures, with considerations for temperature limits.
Function:
    eps_water_liebe93(f, t, modT=0)

Created on Wed Jun  8 15:08:18 2022
Author: Manfred Brath


"""

import numpy as np


def eps_water_liebe93(f, t, modT=0):
    """
    Calculate the complex permittivity of water using the Liebe 1993 model.

    The actual limits of the parameterisation are not known. Studies
    indicated that it degenerates for T<248K (T. Kuhn, WATS study;
    J.Mendrok, ESA planet toolbox study). Hence, the following
    limits are here applied:
     f: [ 10 MHz, 1000 GHz]
     t: [ 248 K, 374 K ]
    The modT flags allows a workaround at lower temperatures. Then, epsilon is
    set to the one of the lowest allowed temperature (T=248K), i.e. assuming
    epsilon to be constant below 248K.

    Function was converted from atmlab (matlab code) to Python by Manfred Brath.
    Originally this function was created by Patrick Eriksson for the atmlab package
    and modiefied by Jana Mendrok.

    Parameters:
        f (float or array-like): Frequency in Hz. Valid range is 0.01 to 1000 GHz.
        t (float or array-like): Temperature in Kelvin. Valid range is 248 to 374 K.
        modT (bool, optional): If True, modifies the temperature for values below 248 K. Default is 0 (False).
    Returns:
        complex: The complex permittivity of water at the given frequency and temperature.
    Raises:
        ValueError: If frequency is outside the valid range or if temperature is below 248 K.
    """

    # Expressions directly from the Liebe 93 paper

    fghz = f / 1e9
    # if np.sum(np.logical_or(fghz<0.01,fghz>1000))>0:
    #   raise ValueError('Valid range for frequency is 0.01-1000 GHz')

    # if np.sum(t>374)>0:
    #   raise ValueError('Temperature above valid range (248-374 K)')

    logic_t = t < 248
    theta = 300.0 / t

    if modT:
        theta[logic_t] = 300.0 / 248.0

    elif np.sum(logic_t) > 0:
        raise ValueError("Temperature below 248")

    e0 = 77.66 + 103.3 * (theta - 1)
    e1 = 0.0671 * e0
    e2 = 3.52

    g1 = 20.2 - 146 * (theta - 1) + 316 * (theta - 1) ** 2
    g2 = 39.8 * g1

    # epswater93 uses 146.4, but the 1993 paper says just 146
    # With 146.4 exactly the same results are obtained.

    return e0 - fghz * ((e0 - e1) / (fghz + 1j * g1) + (e1 - e2) / (fghz + 1j * g2))
