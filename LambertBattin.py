# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:12:57 2024

@author: alespall
"""
"""
import numpy as np
v1=[1,2,3]
v2=[3,4,5]

print(np.dot(v1,v2))
print(np.cross(v1,v2))
print(np.linalg.norm(v1))
"""

import numpy as np
import math as mt

def seebatt(v):

    #-------------------------  implementation   -------------------------
    c=np.zeros(21)

    c[0] =    0.2
    c[1] =    9.0 /  35.0
    c[2] =   16.0 /  63.0
    c[3] =   25.0 /  99.0
    c[4] =   36.0 / 143.0
    c[5] =   49.0 / 195.0
    c[6] =   64.0 / 255.0
    c[7] =   81.0 / 323.0
    c[8] =  100.0 / 399.0
    c[9]=  121.0 / 483.0
    c[10]=  144.0 / 575.0
    c[11]=  169.0 / 675.0
    c[12]=  196.0 / 783.0
    c[13]=  225.0 / 899.0
    c[14]=  256.0 /1023.0
    c[15]=  289.0 /1155.0
    c[16]=  324.0 /1295.0
    c[17]=  361.0 /1443.0
    c[18]=  400.0 /1599.0
    c[19]=  441.0 /1763.0
    c[20]=  484.0 /1935.0

    sqrtopv = np.sqrt(1.0 + v)
    eta = v / (1.0 + sqrtopv)**2

    #------------------- process forwards ----------------------
    delold = 1.0
    termold = c[0] * eta
    sum1 = termold
    i = 0
    while (i <= 20) and (abs(termold) > 0.00000001):
        delnew = 1.0 / (1.0 + c[i + 1] * eta * delold)
        term = termold * (delnew - 1.0)
        sum1 += term
        i += 1
        delold = delnew
        termold = term

    seebat = 1.0 / ((1.0 / (8.0 * (1.0 + sqrtopv))) * (3.0 + sum1 / (1.0 + eta * sum1)))
    return seebat



def seebattk(v):
     # ----------------------------- Lodals ----------------------------
    d=np.zeros(21)

    d[0] =     1.00 /    3.00;
    d[1] =     4.00 /   27.00;
    d[2] =     8.00 /   27.00;
    d[3] =     2.00 /    9.00;
    d[4] =    22.00 /   81.00;
    d[5] =   208.00 /  891.00;
    d[6] =   340.00 / 1287.00;
    d[7] =   418.00 / 1755.00;
    d[8] =   598.00 / 2295.00;
    d[9] =   700.00 / 2907.00;
    d[10]=   928.00 / 3591.00;
    d[11]=  1054.00 / 4347.00;
    d[12]=  1330.00 / 5175.00;
    d[13]=  1480.00 / 6075.00;
    d[14]=  1804.00 / 7047.00;
    d[15]=  1978.00 / 8091.00;
    d[16]=  2350.00 / 9207.00;
    d[17]=  2548.00 /10395.00;
    d[18]=  2968.00 /11655.00;
    d[19]=  3190.00 /12987.00;
    d[20]=  3658.00 /14391.00;

    # ----------------- Prodess Forwards ------------------------
    sum1 = d[0]
    delold = 1.0
    termold = d[0]
    i = 0
    while True:
        delnew = 1.0 / (1.0 + d[i + 1] * v * delold)
        term = termold * (delnew - 1.0)
        sum1 += term
        i += 1
        delold = delnew
        termold = term
        if not (i < 20 and abs(termold) > 0.000001):
            break

    return sum1

#------------------------------------------------------------------------------
#
#                          LAMBERBATTIN
#
#   this subroutine solves Lambert's problem using Battins method. The method
#   is developed in Battin (1987). It uses contiNued fractions to speed the
#   solution and has several parameters that are defined differently than
#   the traditional Gaussian technique.
#
# Inputs:         Description                    Range/Units
#   ro          - IJK Position vector 1          m
#   r           - IJK Position vector 2          m
#   dm          - direction of motion            'pro','retro'
#   Dtsec       - Time between ro and r          s
#
# OutPuts:
#   vo          - IJK Velocity vector            m/s
#   v           - IJK Velocity vector            m/s
#
# Reference:
# Vallado D. A; Fundamentals of Astrodynamics and Applications; McGraw-Hill
# , New York; 3rd edition(2007).
#
# Last modified:   2015/08/12   M. Mahooti
#
#------------------------------------------------------------------------------

def LAMBERTBATTIN(ro,r,dm, Dtsec):
    small = 1e-6
    mu = 3.986004418e14  # m^3/s^2
    y1 = 0
    magr = np.linalg.norm(r)
    magro = np.linalg.norm(ro)
    CosDeltaNu = np.dot(ro, r) / (magro * magr)
    rcrossr = np.cross(ro, r)
    magrcrossr = np.linalg.norm(rcrossr)
    if dm == 'pro':
        SinDeltaNu = magrcrossr / (magro * magr)
    else:
        SinDeltaNu = -magrcrossr / (magro * magr)
    DNu = np.arctan2(SinDeltaNu, CosDeltaNu)

    # the angle needs to be positive to work for the long way
    if DNu < 0.0:
        DNu += 2.0 * np.pi

    RoR = magr / magro
    eps = RoR - 1.0
    tan2w = 0.25 * eps**2 / (np.sqrt(RoR) + RoR * (2.0 + np.sqrt(RoR)))
    rp = np.sqrt(magro * magr) * (np.cos(DNu * 0.25)**2 + tan2w)
    if DNu < np.pi:
        L = (np.sin(DNu * 0.25)**2 + tan2w) / (np.sin(DNu * 0.25)**2 + tan2w + np.cos(DNu * 0.5))
    else:
        L = (np.cos(DNu * 0.25)**2 + tan2w - np.cos(DNu * 0.5)) / (np.cos(DNu * 0.25)**2 + tan2w)

    m = mu * Dtsec**2 / (8.0 * rp**3)
    x = 10.0
    xn = L
    chord = np.sqrt(magro**2 + magr**2 - 2.0 * magro * magr * np.cos(DNu))
    s = (magro + magr + chord) * 0.5
    lim1 = np.sqrt(m / L)

    Loops = 1
    while True:
        x = xn
        tempx = seebatt(x)
        Denom = 1.0 / ((1.0 + 2.0 * x + L) * (4.0 * x + tempx * (3.0 + x)))
        h1 = (L + x)**2 * (1.0 + 3.0 * x + tempx) * Denom
        h2 = m * (x - L + tempx) * Denom

        # ----------------------- Evaluate CUBIC ------------------
        b = 0.25 * 27.0 * h2 / ((1.0 + h1)**3)
        if b < -1.0:  # reset the initial condition
            xn = 1.0 - 2.0 * L
        else:
            if y1 > lim1:
                xn *= lim1 / y1
            else:
                u = 0.5 * b / (1.0 + np.sqrt(1.0 + b))
                k2 = seebattk(u)
                y = ((1.0 + h1) / 3.0) * (2.0 + np.sqrt(1.0 + b) / (1.0 + 2.0 * u * k2**2))
                xn = np.sqrt(((1.0 - L) * 0.5)**2 + m / (y**2)) - (1.0 + L) * 0.5

        Loops += 1
        y1 = np.sqrt(m / ((L + x) * (1.0 + x)))
        if abs(xn - x) < small and Loops > 30:
            break

    a = mu * Dtsec**2 / (16.0 * rp**2 * xn * y**2)

    # ------------------ Find Eccentric anomalies -----------------
    # ------------------------ Hyperbolic -------------------------
    if a < -small:
        arg1 = np.sqrt(s / (-2.0 * a))
        arg2 = np.sqrt((s - chord) / (-2.0 * a))
        #  ------- Evaluate f and g functions --------
        AlpH = 2.0 * np.arcsinh(arg1)
        BetH = 2.0 * np.arcsinh(arg2)
        DH = AlpH - BetH
        F = 1.0 - (a / magro) * (1.0 - np.cosh(DH))
        GDot = 1.0 - (a / magr) * (1.0 - np.cosh(DH))
        G = Dtsec - np.sqrt(-a**3 / mu) * (np.sinh(DH) - DH)
    # ------------------------ Elliptical ---------------------
    elif a > small:
        arg1 = np.sqrt(s / (2.0 * a))
        arg2 = np.sqrt((s - chord) / (2.0 * a))
        Sinv = arg2
        Cosv = np.sqrt(1.0 - (magro + magr - chord) / (4.0 * a))
        BetE = 2.0 * np.arccos(Cosv)
        BetE = 2.0 * np.arcsin(Sinv)
        if DNu > np.pi:
            BetE = -BetE
        Cosv = np.sqrt(1.0 - s / (2.0 * a))
        Sinv = arg1
        am = s * 0.5
        ae = np.pi
        be = 2.0 * np.arcsin(np.sqrt((s - chord) / s))
        tm = np.sqrt(am**3 / mu) * (ae - (be - np.sin(be)))
        if Dtsec > tm:
            AlpE = 2.0 * np.pi - 2.0 * np.arcsin(Sinv)
        else:
            AlpE = 2.0 * np.arcsin(Sinv)
        DE = AlpE - BetE
        F = 1.0 - (a / magro) * (1.0 - np.cos(DE))
        GDot = 1.0 - (a / magr) * (1.0 - np.cos(DE))
        G = Dtsec - np.sqrt(a**3 / mu) * (DE - np.sin(DE))
    #   --------------------- Parabolic ---------------------
    else:
        raise ValueError('a parabolic orbit')

    vo = (r - F * ro) / G
    v = (GDot * r - ro) / G
    return vo, v