import scipy.io as sio
import numpy as np
import colormath
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import colour as cls
import scipy.integrate as scinteg
import scipy.interpolate as scinterp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import copy

def powerfunction(x,a,b,g):
    return a + (b*(x**g))

def invpowerfunction(x,a,b,g):
    x = copy.copy(x)
    for i, num in enumerate(x):
        if num < a:
            x[i] = a

    return ((x-a)/b)**(1/g)

def genGammaCurves(gunvals,XYZs):
    """
    genGammaCurves takes an array (form 1xM) of of gun values (only the gun values of the color of interest) ranging
    from 0 to 255 as well as an array of XYZ values (form MxN) and computes the luminance gamma curves
    :param gunVals:
    :param XYZs:
    :return: LUT
    """
    assert np.shape(gunvals)[0] == np.shape(XYZs)[0], "gunvals length must match M dimension of XYZs"
    lumvals = XYZs[:,1] # Get the luminance values
    prop_lum = lumvals/np.max(lumvals)
    prop_gunvals = gunvals/255
    popt, pcov = curve_fit(powerfunction,prop_gunvals,prop_lum)
    gunrange = np.linspace(0,1,256)
    lumfitted = invpowerfunction(gunrange, *popt)
    LUT = lumfitted * 256

    fig, ax = plt.subplots()
    ax.plot(gunrange, powerfunction(gunrange, *popt))
    ax.scatter(prop_gunvals, prop_lum)
    ax.plot(gunrange, lumfitted)
    ax.plot(gunrange,powerfunction(lumfitted,*popt))
    return LUT



def spectra_to_XYZ(spectra, lambdas, cmfs, integration='Trapezoidal'):
    """
    spectra_to_XYZ takes an array of spectra (or a single spectrum) and returns XYZ values.
    :param spectra: An array of spectra, where each row is a new spectrum.
    :param lambdas: An array representing the wavelengths at which the spectra were measured. Don't pass a 2d array unless
    all the wavelengths are the same.
    :param cmfs: An array with all three color matching functions and their wavelengths. The columns should represent
     [lambda, X, Y,Z]. The CMFs should cover wavelengths that are shared by the spectral measurements, as I have not
     written interpolation functionality yet.
    :param integration: Type of integration. Will accept 'Trapezoidal', which will use np.trapz for integration of the
    Spectra*CMF curve, 'Spline', which will use scipy.interpolate UnivariateSpline integration method, or 'Sum', which
    uses the rectangular method--assumes the first rectangular block has same length
    :return: An array of arrays of the XYZ measurements for each color, where each row represents a different color.
    """
    assert isinstance(spectra, (np.ndarray, list, tuple)), 'Parameter input spectra not of <class (list, tuple, np.ndarray)>'
    assert isinstance(lambdas, (np.ndarray, list, tuple)), "Parameter input lambdas not of <class (list, tuple, np.ndarray)>"
    #assert isinstance(white, (np.ndarray, list, tuple)), "Parameter input white not of <class (list, tuple, np.ndarray)>"
    #assert isinstance(black, (np.ndarray, list, tuple)), "Parameter input black not of <class (list, tuple, np.ndarray)>"
    assert isinstance(cmfs, (np.ndarray, list, tuple)), "Parameter input cmfs not of <class (list, tuple, np.ndarray)>"
    assert integration == 'Trapezoidal' or integration == 'Spline' or integration == 'Sum', "Parameter input " \
                                                                    "integration=={} not an accepted " \
                                                                    "value. Accepted values are " \
                                                                    "('Trapezoidal','Spline','Sum)".format(integration)
    if np.diff(cmfs[:,0])[0] > 1:
        lambdas_interp = np.arange(360,801,1)
        Xs_new = np.interp(lambdas_interp,cmfs[:,0],cmfs[:,1])
        Ys_new = np.interp(lambdas_interp,cmfs[:,0],cmfs[:,2])
        Zs_new = np.interp(lambdas_interp,cmfs[:,0],cmfs[:,3])
        cmfs_new = np.column_stack((lambdas_interp,Xs_new,Ys_new,Zs_new))
        cmfs = cmfs_new




    cmf_inds = np.intersect1d(cmfs[:,0],lambdas,return_indices=True)[1]
    X_cmf = cmfs[cmf_inds, 1]
    Y_cmf = cmfs[cmf_inds, 2]
    Z_cmf = cmfs[cmf_inds, 3]
    X_Curve = []
    Y_Curve = []
    Z_Curve = []
    X = []
    Y = []
    Z = []
    #if isinstance(spectra[0], (np.ndarray, list, tuple)):
    #    for spectrum in spectra:
    #        X_Curve.append(X_cmf * spectrum)
    #        Y_Curve.append(Y_cmf * spectrum)
    #        Z_Curve.append(Z_cmf * spectrum)
    #        X_raw.append(np.trapz(X_Curve[-1], dx=lambdas))
    #        Y_raw.append(np.trapz(Y_Curve[-1], dx=lambdas))
    #        Z_raw.append(np.trapz(Z_Curve[-1], dx=lambdas))
    X_Curve = (X_cmf * spectra)
    Y_Curve = (Y_cmf * spectra)
    Z_Curve = (Z_cmf * spectra)
    if integration == 'Trapezoidal':
        X = np.trapz(X_Curve, x=lambdas)
        Y = np.trapz(Y_Curve, x=lambdas)
        Z = np.trapz(Z_Curve, x=lambdas)
    elif integration == 'Spline':
        int_start = min(lambdas)
        int_end = max(lambdas)
        for x_curve,y_curve,z_curve in zip(X_Curve,Y_Curve,Z_Curve):
            spl_x = UnivariateSpline(lambdas,x_curve)
            spl_y = UnivariateSpline(lambdas,y_curve)
            spl_z = UnivariateSpline(lambdas,z_curve)
            X.append(spl_x.integral(int_start,int_end))
            Y.append(spl_y.integral(int_start,int_end))
            Z.append(spl_z.integral(int_start,int_end))
    elif integration == 'Sum': # this cannot handle a single spectrum
        diffs = np.diff(lambdas)
        diffs = np.insert(diffs, 0, diffs[0])
        X = np.sum((X_Curve * diffs), axis = 1)
        Y = np.sum((Y_Curve * diffs), axis = 1)
        Z = np.sum((Z_Curve * diffs), axis = 1)

    #X_Curve_White = X_cmf * white
    #Y_Curve_White = Y_cmf * white
    #Z_Curve_White = Z_cmf * white
    #X_White = np.trapz(X_Curve_White, x=lambdas)
    #Y_White = np.trapz(Y_Curve_White, x=lambdas)
    #Z_White = np.trapz(Z_Curve_White, x=lambdas)

    #X_Curve_Black = X_cmf * black
    #Y_Curve_Black = Y_cmf * black
    #Z_Curve_Black = Z_cmf * black
    #X_Black = np.trapz(X_Curve_Black, x=lambdas)
    #Y_Black = np.trapz(Y_Curve_Black, x=lambdas)
    #Z_Black = np.trapz(Z_Curve_Black, x=lambdas)

    #X = (X_raw - X_Black)/(X_White - X_Black)
    #Y = (Y_raw - Y_Black)/(Y_White - Y_Black)
    #Z = (Z_raw - Z_Black)/(Z_White - Z_Black)

    XYZ = np.stack((X,Y,Z), axis = 1) # The raw values might be what we want anyway
    return XYZ

def normalize_XYZ(XYZ, axis = 1):
    """
    normalize_XYZ takes XYZ coordintes and normalizes them to X' Y' Z' coordinates for use in plotting on a chromaticity
     diagram. It will also remove any blacks from the array.
    :param XYZ: An 3xM or Nx3 array of XYZ coordinates.
    :param axis: [0,1] representing the axis along which the XYZ coordinates are laid out. Default is 1.
    :return: XYZ_norm
    """
    assert isinstance(XYZ, (list, tuple, np.ndarray)), "Parameter input XYZ is not of <class list tuple np.ndarray"
    assert axis == 0 or axis == 1, "Parameter input axis is not an appropriate value. Must be either 0 or 1"
    assert XYZ.shape[0] == 3 or XYZ.shape[1] == 3, "Parameter input XYZ is not of length 3 along either axis."
    assert len(XYZ.shape) <= 2, "Parameter input XYZ must be a 1D or 2D array."

    if axis == 0: # Transpose array if each column is a different color
        XYZ = XYZ.T

    blacks = np.where(np.sum(XYZ, axis = 1) == 0)[0] # Find indices of black colors (0, 0, 0)
    XYZ_NB = np.delete(XYZ,blacks, axis = 0) # Remove black colors from array.
    abc = np.sum(XYZ_NB, axis=1)
    XYZ_norm = np.divide(XYZ_NB, abc[:,None])
    return XYZ_norm

def LUV_to_XYZ(LUV, white = np.ones(3), axis = 1):
    """
    LUV_to_XYZ: Takes an array of LUV values and converts them to XYZ values.
    :param LUV: An array of LUV values of the form 3xM or Nx3.
    :param white: A reference white value in X,Y,Z coordinates.
    :param axis: (Default 1) The axis along which the LUV values are arranged.
    :return: XYZ
    """
    assert isinstance(LUV, (list, tuple, np.ndarray)), "Parameter input LUV is not of <class list tuple np.ndarray"
    assert axis == 0 or axis == 1, "Parameter input axis is not an appropriate value. Must be either 0 or 1"
    assert LUV.shape[0] == 3 or LUV.shape[1] == 3, "Parameter input LUV is not of length 3 along either axis."
    assert len(LUV.shape) <= 2, "Parameter input LUV must be a 1D or 2D array."
    assert len(white) == 3, "Parameter input white must be of size 3" # fix this to allow lists to be passed
    if axis == 0:
        LUV = LUV.T

    L = LUV[:,0]
    U = LUV[:,1]
    V = LUV[:,2]
    Xr = white[0]
    Yr = white[1]
    Zr = white[2]
    epsilon = 216/24389
    kappa = 24389/27
    v0 = 9*Yr/(Xr+15*Yr+3*Zr)
    u0 = 4*Xr/(Xr+15*Yr+3*Zr)
    Y = np.copy(LUV[:,0])
    for index,l in enumerate(Y):
        if l > (kappa * epsilon):
            Y[index] = pow(((l+16)/116),3)
        else:
            Y[index] = l/kappa
    d = Y*(((39*L)/(V+(13*L*v0)))-5)
    c = -1/3
    b = -5*Y
    a = (1/3)*(((52*L)/(U+(13*L*u0)))-1)
    X = (d-b)/(a-c)
    Z = (X*a)+b
    XYZ = np.stack((X, Y, Z), axis=1)
    return XYZ

def XYZ_to_LUV(XYZ, white = np.ones(3), axis = 1):
    """
    XYZ_to_LUV: Takes a matrix of XYZ values and a white point and converts the XYZ values to LUV values.
    XYZ: Matrix of XYZ values of shape either 3xN or Mx3.
    white: Whitepoint of the form [Xw,Yw,Zw]
    axis: Axis along which to read the X,Y,and Z coordinates.
    Returns: LUV
    """
    assert isinstance(XYZ, (list, tuple, np.ndarray)), "Parameter input XYZ is not of <class list tuple np.ndarray"
    assert axis == 0 or axis == 1, "Parameter input axis is not an appropriate value. Must be either 0 or 1"
    assert XYZ.shape[0] == 3 or XYZ.shape[1] == 3, "Parameter input XYZ is not of length 3 along either axis."
    assert len(XYZ.shape) <= 2, "Parameter input XYZ must be a 1D or 2D array."
    assert len(white) == 3, "Parameter input white must be of size 3"

    if axis == 0:
        XYZ = XYZ.T

    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]
    Xr = white[0]
    Yr = white[1]
    Zr = white[2]
    epsilon = 216/24389
    kappa = 24389/27
    vpr = (9*Yr)/(Xr+(15*Yr)+(3*Zr))
    upr = (4*Xr)/(Xr+(15*Yr)+(3*Zr))
    vp = (9*Y)/(X+(15*Y)+(3*Z))
    up = (4 * X) / (X + (15 * Y) + (3 * Z))
    small_yr = Y/Yr
    L = []
    for y in small_yr:
        if y > epsilon:
            L.append((116 * (y ** (1./3))) - 16)
        else:
            L.append(kappa * small_yr) # This may need fixing
    L = np.array(L)
    v = 13 * L * (vp - vpr)
    u = 13 * L * (up - upr)
    LUV = np.stack((L, u, v), axis=1)
    upvp = np.stack((up,vp),axis=1)
    return LUV, upvp

def RGBXYZMatrix(red, green, blue, refwhite=None ,calc='XYZ'):
    '''
    RGBXYZMatrix: Computes the matrix M and the inverse matrix M-1 to convert between an RGB system
    and XYZ.
    :param red: A tuple, list, or array of length two containing the chromaticity coordinates xr and yr, or the three
    chromaticity coordinates Xr Yr Zr
    :param green: A tuple, list, or array of length two containing the chromaticity coordinates xg and yg, or the three
    chromaticity coordinates Xg Yg Zg
    :param blue: A tuple, list, or array of length two containing the chromaticity coordinates xb and yb, or the three
    chromaticity coordinates Xb Yb Zb
    :param refwhite: A tuple, list, or array of length three containing Xw, Yw, and Zw.
    :calc: Either 'xyY' or 'XYZ', indicating whether you are providing x and y values or XYZ values
    :return: M and M-1
    '''

    assert isinstance(red, (list, tuple, np.ndarray)), "Parameter red must be of class <tuple, list, np.ndarray"
    assert isinstance(green, (list, tuple, np.ndarray)), "Parameter green must be of class <tuple, list, np.ndarray"
    assert isinstance(blue, (list, tuple, np.ndarray)), "Parameter blue must be of class <tuple, list, np.ndarray"
    if not refwhite is None:
        assert isinstance(refwhite, (list, tuple, np.ndarray)), "Parameter refwhite must be of class <tuple, list, np.ndarray, none>"
    if calc == 'xyY':
        assert len(red) == 2, "Paramter red must of of length 2"
        assert len(green) == 2, "Paramter green must of of length 2"
        assert len(blue) == 2, "Paramter blue must of of length 2"
    elif calc == 'XYZ':
        assert len(red) == 3, "Paramter red must of of length 3"
        assert len(green) == 3, "Paramter green must of of length 3"
        assert len(blue) == 3, "Paramter blue must of of length 3"

    if calc == 'xyY':
        xr = red[0]
        yr = red[1]
        xg = green[0]
        yg = green[1]
        xb = blue[0]
        yb = blue[1]
        Xr = xr/yr
        Yr = 1
        Zr = (1-xr-yr)/yr
        Xg = xg/yg
        Yg = 1
        Zg = (1-xg-yg)/yg
        Xb = xb/yb
        Yb = 1
        Zb = (1-xb-yb)/yb

        XYZMAT = np.array([[Xr, Xg, Xb],[Yr,Yg,Yb],[Zr,Zg,Zb]])

    elif calc == 'XYZ':
        XYZMAT = np.transpose(np.vstack((red, green, blue)))
    XYZMATINV = np.linalg.inv(XYZMAT)
    if refwhite is None:
        refwhite = np.sum(XYZMAT, axis=1)
    WHITEMAT = np.reshape(np.asarray(refwhite), [3,1])
    S = np.reshape(np.matmul(XYZMATINV,WHITEMAT),[3])


    M = np.multiply(XYZMAT, S)

    MINV = np.linalg.inv(M)

    return M, MINV

def ChromaticAdaptation(source,sourcewhite,destinationwhite,method='bradford'):
    '''
    ChromaticAdaptation computes a linear transformation of a source color to a destination color based on a source reference white and a destination reference white.
    :param source: A 3xN matrix of the source XYZ values, with the first row representing the X values, the second the Y values, and the third the Z values.
    :param sourcewhite: An array, tuple, or list of length 3 of the source color reference white (Xw,Yw,Zw)
    :param destinationwhite: An array, tuple, or list of length 3 of the destination source color reference white (Xd,Yd,Zd).
    :param method: One of either 'xyzscaling', 'bradford', or 'vonkries' (default). This determines the content of Ma and MaINV
    :return: A 3xN matrix of destination XYZ values.
    '''

    if not type(source) == np.ndarray:
        try:
            source = np.array(source)
        except BaseException as err:
            print("Unexpected {}, {}".format(err, type(err)))


    assert source.shape[0] == 3, "Source must be of the shape 3xN"
    assert len(source.shape) == 2, "Source must only have 2 dimensions"
    assert isinstance(sourcewhite, (np.ndarray,list,tuple)), "Sourcewhite must be of class <np.ndarray, list, tuple>"
    assert isinstance(destinationwhite, (np.ndarray, list, tuple)), "Destinationwhite must be of class <np.ndarray, list, tuple>"
    assert len(sourcewhite) == 3, "Sourcewhite must have three elements (X,Y,Z)"
    assert len(destinationwhite) == 3, "Destinationwhite must have three elements (X,Y,Z)"
    assert method in ['xyzscaling','bradford','vonkries'], "Method must be one of 'xyzscaling', 'bradford', 'vonkries'"

    if not type(destinationwhite) == np.ndarray:
        destinationwhite = np.array(destinationwhite)
    if not type(sourcewhite) == np.ndarray:
        sourcewhite = np.array(sourcewhite)

    if method == 'xyzscaling':
        Ma = np.array([[1,0,0],[0,1,0],[0,0,1]])
        MaINV = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif method == 'bradford':
        Ma = np.array(
            [[0.8951000,0.2664000,-0.1614000],[-0.7502000,1.7135000,0.0367000],[0.0389000,-0.0685000,1.0296000]])
        MaINV = np.array(
            [[0.9869929,-0.1470543,0.1599627], [0.4323053,0.5183603,0.0492912], [-0.0085287,0.0400428,0.9684867]])
    elif method == 'vonkries':
        Ma = np.array([[0.4002400,0.7076000,-0.0808100], [-0.2263000,1.1653200,0.0457000],
                       [0.0000000,0.0000000,0.9182200]])
        MaINV = np.array(
            [[1.8599364,-1.1293816,0.2198974], [0.3611914,0.6388125,-0.0000064], [0.0000000,0.0000000,1.0890636]])

    cs = np.matmul(Ma,sourcewhite)
    cd = np.matmul(Ma,destinationwhite)

    conemat = np.array([[cd[0]/cs[0],0,0],[0,cd[1]/cs[1],0],[0,0,cd[2]/cs[2]]])
    M = np.matmul(MaINV,np.matmul(conemat,Ma))
    destination = np.matmul(M,source)
    return destination

def xyY_to_XYZ(xyY):
    '''

    :param xyY: xyY values.
    :return: XYZ values.
    '''
    assert np.shape(xyY)[1] == 3, "xyY must be an array with 3 columns"
    x = xyY[:,0]
    y = xyY[:,1]
    Y = xyY[:,2]
    X = (x*Y)/y
    Z = ((1 - x - y)*Y)/y
    XYZ = np.column_stack((X,Y,Z))
    return XYZ










