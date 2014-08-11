import os, sys
lib_path = os.path.abspath('/home/ec511/aipy-0.8.5')
sys.path.append(lib_path)

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.misc as misc
import scipy.fftpack as fft
import math
import aipy
import pylab


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


def image_input(filename):
    img = mpimg.imread(filename)
    return img

    

def make_image(array):
    array = np.real(array)
    imgplot = plt.imshow(array, cmap = "gist_yarg")
    imgplot.set_clim(0.001, 0.002)
    plt.show()

def make_final_image(array):
    array = np.real(array)
    imgplot = plt.imshow(array)
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

"""def discrete_uv_plane(array):

    dim = array.shape

    b =np.zeros(dim)



    
        if array[x1,y1] == 0:
            b[x1,y1] = 0
        else:
            for (x2,y2), value in np.ndenumerate(array):
                if array[x2,y2] == 0:
                    b[x2,y2] =0
                else :
                    u = (x1 - x2) / redshifted_lambda(1)
                    v = (y1 - y2) / redshifted_lambda(1)
                    if u < 8 and v<8 and u>= 0 and v >= 0:
                        b[u,v] = 1
                    else:
                        pass
                    
    return b"""
                


def discrete_uv_VLA(antennas, nuv): #a method to get the discrete uv sampling distribution for the VLA telescope. will generalise for other telescopes soon
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

def dirty_beam(array):


    beam = fft.fftshift(fft.ifft2(array))

    return beam
    




def small_interferometer(nuv,r): #<100 antennae

    centre = nuv/2
    
    galaxy = create_disc(nuv,nuv,centre,centre,r,nuv)

    telescope = VLA_D_config

    ftgal = fft.fftshift(fft.fft2(galaxy))

    make_image(ftgal)

    uvplane = discrete_uv_VLA(telescope, nuv)

    rotated = earth_rotation_synthesis(uvplane,nuv)

    gridded = grid(rotated,nuv)

    sampled_sky = sample(ftgal,gridded)

    dirty_image =fft.ifft2(fft.ifftshift(sampled_sky))

    make_image(dirty_image)

    dirtybeam = dirty_beam(gridded)

    make_image(dirtybeam)

    deconvolved = aipy.deconv.lsq(dirty_image,dirtybeam, gain=.1, tol=1e-3, maxiter=500)

    deconvolved =  np.abs(deconvolved[0])**2

    imgplot = plt.imshow(deconvolved, cmap = "gist_yarg")

    plt.show()
    






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
    lambda_obs = (1+z)* 0.21

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

def angofres():

    bl_max = np.sqrt((527.6 + 592.4*(np.sin(np.pi/6)))**2 + (np.sin(np.pi/6))**2)

    angle_of_resolution = redshifted_lambda(1)/ bl_max

    return angle_of_resolution



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
