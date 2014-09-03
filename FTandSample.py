# -*- coding: utf-8 -*-
import os, sys
lib_path = os.path.abspath('/home/ec511/aipy-0.8.5')
sys.path.append(lib_path)

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
import math
import aipy
import pylab
import scipy.ndimage
import numpy.fft as fft
import Cosmology
cosm = Cosmology.Cosmology()
import cosmoconstants

"""to run:

for an interferometer with fewer than 100 antennas: run small_interferometer(size of uv plane, radius of disc)
for an interferometer with more than 100 antennas run large_interferometer()"""


#move towards more general antennae and image size arrays. Understanding how a given physical base
#line on the group maps to an angular scale on the sky. This will help get to grips with images that have
#pixels smaller than the resolution of the telescope.



def create_disc(j, k,a, b, r, n):    
    array = np.zeros((j,k))
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array[mask]= 1
    return array


def openbox(filename, dim):
   
    size = dim*dim*dim
    dtype = 'float32'
    fd = open(filename,'rb')
    read_data = np.fromfile(fd,dtype)
    fd.close()

    print 'openBox-> Number of pixels in box should be:', size, 'data is in fact', len(read_data)

    if not size==len(read_data):
        print 'tocm.py(openBox) Error: Read box does not match expected size=',dim,'...'
        sys.exit(1)

        
    else:
        data = np.array(read_data)
        newshape = np.reshape(data, (200,200,200))
        return newshape
        #returns the array as a numpy array

def slices(array, z):

    image = array[0:,0:,z]
    return image

def comovingdist(z):
    comov = cD(z)
    return comov



def scale_uv(antennas, z, nuv):
    bl = discrete_uv(antennas, nuv)
    bmax = max(bl)
    bmax = np.linalg.norm(bmax)
    umax = bmax / redshifted_lambda(z)
    umin = umax/ nuv
    nmin = 1/ umax #is it in rads
    nmax = 1/umin #is it in rads
    N = nmax/nmin

    return nmin, nmax, N

def scalebox(z, nuv):
    angle = 100/ cosm.comovingDistanceSingle(z)
    resampled_angle = (angle / nuv) *200
    return resampled_angle 


def match_scales(nuv, z, antennas, galaxy):
    nmin, nmax, N = scale_uv(antennas, z, nuv)
    resampledang = scalebox(z,nuv)
    zoomfactor = nmax/resampledang
    scaledsky = scipy.ndimage.interpolation.zoom(galaxy, zoomfactor)
    return scaledsky



def make_scaled(z):
    Dc = cosm.comovingDistanceSingle(z)
    angle = 100 / Dc
    nmax = angle
    nmin = nmax/200
    umax = 1/nmin
    umin = 1/ nmax
    nuv = umax/ umin
    return nuv
    

def field_of_view(antennas, nuv, z):
    bl = discrete_uv(antennas, nuv)
    bmin = min(bl)
    bmin = np.linalg.norm(bmin)
    angfov = bmin / redshifted_lambda(z)
    return angfov
    

def ang_of_res(antennas, nuv):
    bl = discrete_uv(antennas, nuv)
    bmax = bl.max
    bmax = np.linalg.norm(bmax)
    angres = bmax / redshifted_lambda(z)
    return angres


def make_image(array):
    array = np.real(array)
    imgplot = plt.imshow(array)
    plt.show()

def final_image(array1, array2, array3, array4):

    array1 = np.real(array1)
    array2 = np.real(array2)
    array3 = np.real(array3)
    array4 = np.real(array4)
    
    
    fig = plt.figure()
    fig.add_subplot(221, title = "True Image")    #top left
    plt.imshow(array1)
    plt.axis([0,100,0,100])
    plt.xlabel("Mpc")
    plt.ylabel("Mpc")
    plt.colorbar()
    fig.add_subplot(222, title = "Dirty Image")   #top right
    plt.imshow(array2)
    plt.axis([0,100,0,100])
    plt.xlabel("Mpc")
    plt.ylabel("Mpc")
    plt.colorbar()
    fig.add_subplot(223, title = "Observed Image")  #bottom left
    plt.imshow(array3)
    plt.axis([0,100,0,100])
    plt.xlabel("Mpc")
    plt.ylabel("Mpc")
    plt.colorbar()
    fig.add_subplot(224, title = "Difference between True and Observed")   #bottom right
    plt.imshow(array4)
    plt.axis([0,100,0,100])
    plt.xlabel("Mpc")
    plt.ylabel("Mpc")
    plt.colorbar()
    plt.savefig("clean.png")
    plt.show()


def sample(array1,array2):
    array1 = np.real(array1)
    array2 = np.real(array2)
    return np.multiply(array1,array2)

def continuous_uv_plane(array1, array2):
    uv = sig.convolve(array1, array2, "same")
    return uv

def dirty_beam(array):
    dirty_beam = fft.fft2(array)
    return dirty_beam
                

def discrete_uv(antennas, nuv): #a method to get the discrete uv sampling distribution for the VLA telescope. will generalise for other telescopes soon
    antennas = antennas()
    basel = []
    for x,y in antennas:
        for x2, y2 in antennas:
            u = (x - x2)/redshifted_lambda(1)
            v = (y - y2)/redshifted_lambda(1)
            basel.append((u,v))

    return basel


def grid(points,nuv):   
    points = np.array(points)
    klist =[]
    llist =[]
    maxx, maxy = np.max(points[:,0]), np.max(points[:,1])
    minx, miny = np.min(points[:,0]), np.min(points[:,1])
    lengthx = maxx - minx
    lengthy = maxy-miny

    
    array = np.zeros((nuv,nuv))
    
    for (u,v) in points:

        k = (nuv/2 -1) + (u/(lengthx/nuv))
        l = (nuv/2 -1) + (v/(lengthy/nuv))
           

        array[(k,l)] = 1


        
    return array


def earth_rotation_synthesis(bl,nuv):

    points = []
    Hs = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    d = np.pi/2
    for (u,v) in bl:
        skypoint = (u,v,0)
        
        for H in Hs:
            rotation_array = np.array([[np.sin(H), np.cos(H), 0], [-np.sin(d)*np.cos(H), np.sin(d)*np.sin(H), np.cos(d)], [np.cos(H)*np.cos(d), (-1*np.cos(d)*np.sin(H)), np.sin(d)]])
            rotated = np.dot(rotation_array, skypoint)
            points.append((rotated[0],rotated[1]))

    return points

def dirty(array):

    beam = fft.ifft2(array)
    return beam

def size_of_pixel(theta, nuv):
    size = theta/nuv
    return size

def difference_plot(array1, array2):
    y_mod = np.subtract(array1, array2)   
    
    return y_mod


def small_interferometer(z): 

    """
    small interferometer is defined as having less than 100 antennas. Currently using VLA's D configuration.
    nuv: number of pixels per side in array
    r: radius of galaxy"""
    nuv = make_scaled(z)


    galaxy = openbox("/home/ec511/21cmFAST/Boxes/delta_T_v2_no_halos_nf0.538078_z10.00_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb12.41_200_100Mpc", 200)

    galaxy = slices(galaxy, z)

    telescope = ASKAP_configuration

    ftgal = fft.fftshift(fft.fft2(galaxy))

    uvplane = discrete_uv(telescope, nuv)

    rotated = earth_rotation_synthesis(uvplane,nuv)

    gridded = grid(rotated,nuv)

    #scaledsky = match_scales(nuv, z, telescope, galaxy)

    sampled_sky = sample(ftgal,gridded)

    dirty_image =fft.ifft2(fft.ifftshift(sampled_sky))

    dirtybeam =fft.ifftshift(dirty(gridded))

    size_of_beam(uvplane,nuv)

    deconvolved = aipy.deconv.lsq(dirty_image,dirtybeam, gain=.1, tol=1e-5, maxiter=500)

    deconvolved =  np.abs(deconvolved[0])**2

    difference = difference_plot(galaxy, deconvolved)

    final_image(galaxy, dirty_image, deconvolved, difference)

    


def size_of_beam(uvplane, nuv):
#fourier transform of the antennas makes beam
# so FT of width of antenna array is the size in Fspace of the beam

    real_space= grid(uvplane,nuv)

    real_space = fft.fft2(real_space)

    maxx, maxy = np.max(real_space[:,0]), np.max(real_space[:,1])
    minx, miny = np.min(real_space[:,0]), np.min(real_space[:,1])
    lengthx = np.abs(maxx - minx)
    lengthy = np.abs(maxy-miny)


    angular_width = (lengthx / 1e6) * (180/ np.pi) * (3600)
    
    
    print "The size of the beam is ", angular_width, " arcseconds"

   

    

    


def large_interferometer(filename): #large >100 antennae

    galaxy =create_disc(filename)
    
    ftgal = twod_FT(galaxy) 

    telescope = VLA_D_config

    uvplane = continuous_uv_plane(telescope, telescope) # take output of make_antennas

    sampled_sky = sample(ftgal, uvplane) 

    dirty_image = inv_twod_ft(sampled_sky)

    clean_image = hogbom(dirtyimage, dirtybeam, True, 0.3, 0.1, 100)

    make_image(clean_image)


def redshifted_lambda(z):
    lambda_obs = (1+z)* 0.2112
    return lambda_obs


def VLA_D_config():
    b = [] #1 = 1m
    north_antennas = [0.9, 54.9, 94.9, 134.9, 194.8, 266.4, 347.1, 436.4, 527.6]
    east_antennas = [39.0, 44.8, 89.9, 147.3, 216.0, 295.4, 384.9, 484.0, 592.4]
    west_antennas = [39.0, 44.9, 89.9, 147.4, 216.1, 295.5, 384.9, 484.0, 592.4]

    for i in north_antennas:
        x = 0
        y = i
        b.append((x,y))

    for i in east_antennas:
        x = i * np.cos(np.pi/6)
        y = i* np.sin(np.pi/6) * -1
        b.append((x,y))

    for i in west_antennas:
        x = i * np.cos(np.pi/6)* -1
        y = i* np.sin(np.pi/6) * -1
        b.append((x,y))


    return b

def ASKAP_configuration():
    b = [(-175, -1673), (261, -797), (-29, -744), (-289, -587), (-157, -816), (-521, -755), (-1061, -841), (-922, -998), (-818, -1142), (-532, -851), (81, -790), (31, -1209), (-1165, -317), (-686, -590), (-499, -506), (-182, -365), (421, -811), (804, -1273), (-463, -236), (-450, -15), (14, -111), (-426, 182), (-333, 504), (-1495, -1416), (-1039, -1128), (-207, -956), (-389, -482), (-434, -510), (-398, -462), (-425, -475), (-400, -3664), (1796, -1468), (600, 1532), (-400, 2336), (-3400, 1532), (-2596, -1468)]
    return b

def angofres():

    bl_max = np.sqrt((527.6 + 592.4*(np.sin(np.pi/6)))**2 + (np.sin(np.pi/6))**2)

    angle_of_resolution = 0.21/ bl_max

    convert_to_arcseconds = (180 * 3600* angle_of_resolution)/(np.pi)

    print "The angle of resolution of this interferometer is ", convert_to_arcseconds, "arcseconds"



def overlapIndices(a1, a2, 
                   shiftx, shifty):
    if shiftx >=0:
        a1xbeg=shiftx
        a2xbeg=0
        a1xend=a1.shape[0]
        a2xend=a1.shape[0]-shiftx
    else:
        a1xbeg=0
        a2xbeg=-shiftx
        a1xend=a1.shape[0]+shiftx
        a2xend=a1.shape[0]

    if shifty >=0:
        a1ybeg=shifty
        a2ybeg=0
        a1yend=a1.shape[1]
        a2yend=a1.shape[1]-shifty
    else:
        a1ybeg=0
        a2ybeg=-shifty
        a1yend=a1.shape[1]+shifty
        a2yend=a1.shape[1]

    return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)

        

def hogbom(dirty,
           psf,
           window,
           gain,
           thresh,
           niter):
    """
    Hogbom clean

    :param dirty: The dirty image, i.e., the image to be deconvolved

    :param psf: The point spread-function

    :param window: Regions where clean components are allowed. If
    True, thank all of the dirty image is assumed to be allowed for
    clean components

    :param gain: The "loop gain", i.e., the fraction of the brightest
    pixel that is removed in each iteration

    :param thresh: Cleaning stops when the maximum of the absolute
    deviation of the residual is less than this value

    :param niter: Maximum number of components to make if the
    threshold "thresh" is not hit
    """
    comps=np.zeros(dirty.shape)
    res= dirty
    if window is True:
        window=np.ones(dirty.shape,
                          np.bool)
    for i in range(niter):
        mx, my=np.unravel_index(np.abs(res[window]).argmax(), res.shape)
        mval=res[mx, my]*gain
        comps[mx, my] += mval
        a1o, a2o=overlapIndices(dirty, psf,
                                (mx-dirty.shape[0])/2,
                                (my-dirty.shape[1])/2)
        res[a1o[0]:a1o[1],a1o[2]:a1o[3]]-=psf[a2o[0]:a2o[1],a2o[2]:a2o[3]]*mval
        if np.abs(res).max() < thresh:
            continue
    
    return comps
