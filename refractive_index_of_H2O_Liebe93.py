#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:08:18 2022

@author: u242031
"""

import numpy as np

# EPS_WATER_LIEBE93   Dielectric constant for pure water according to Liebe 93
#
#    Provides the complex dielectric constant following the Liebe 1993
#    paper.
#
#    Note that the function *epswater93* solves the same task. This function
#    was primarily implemented to compare if different versions of the 
#    mathematical expressions give the same result (which they did). In fact,
#    this function contains two parallel versions, giving identical
#    results. This function uses just SI units for input arguments, in 
#    contrast to *epswater93*.
#
#    The actual limits of the parameterisation are not known. Studies
#    indicated that it degenerates for T<248K (T. Kuhn, WATS study;
#    J.Mendrok, ESA planet toolbox study). Hence, the following
#    limits are here applied:
#      f: [ 10 MHz, 1000 GHz] 
#      t: [ 248 K, 374 K ]
#    The modT flags allows a workaround at lower temperatures. Then, epsilon is
#    set to the one of the lowest allowed temperature (T=248K), i.e. assuming
#    epsilon to be constant below 248K. 
#
# FORMAT    e = eps_water_liebe93( f, t [, modT] )
#        
# OUT   e    Complex dielectric constant
# IN    f    Frequency
#       t    Temperature
# OPT   modT flag, whether to use modified t if t<248K (liebe formula 
#            degenerates at these temperatures). Default is false.

# 2004-10-22   Created by Patrick Eriksson
# 2013-10-04   Jana Mendrok introduced modT flag


def eps_water_liebe93( f, t, modT=0 ):

# Expressions directly from the Liebe 93 paper


    fghz  = f/1e9
    # if np.sum(np.logical_or(fghz<0.01,fghz>1000))>0:
    #   raise ValueError('Valid range for frequency is 0.01-1000 GHz')
    
    
    # if np.sum(t>374)>0:
    #   raise ValueError('Temperature above valid range (248-374 K)') 
    
    logic_t=t<248
    theta = 300./t
    
    if modT:
        theta[logic_t] = 300./248.
        
    elif np.sum(logic_t)>0:
        raise ValueError('Temperature below 248')
    
    
    e0 = 77.66 + 103.3 * ( theta - 1 )
    e1 = 0.0671 * e0
    e2 = 3.52
    
    g1 = 20.2 - 146 * ( theta -1 ) + 316 * ( theta -1 )**2
    g2 = 39.8 * g1
    
    # epswater93 uses 146.4, but the 1993 paper says just 146
    # With 146.4 exactly the same results are obtained.
    
    return e0 - fghz * ( (e0-e1)/(fghz+1j*g1) + (e1-e2)/(fghz+1j*g2) )