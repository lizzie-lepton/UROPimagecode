import numpy as np
import numpy.fft as fft
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.misc as misc

#move towards more general antennae and image size arrays. Understanding how a given physical base
#line on the group maps to an angular scale on the sky. This will help get to grips with images that have
#pixels smaller than the resolution of the telescope.

# take an image and turn into a bit map that can be processed.

#make an antenna array more easily.





def create_disc(j, k, a, b, r, n):
    array = np.zeros((j,k))
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array[mask]= 1
    return array

#def make_antennas():

#def input_signal():
    

def make_image(array):
    array = np.real(array)
    imgplot = plt.imshow(array)
    plt.show()
    return imgplot

def oned_FT(x):
    FT = fft.fft(x)
    return FT

def twod_FT(array):
    twoFT = fft.fft2(array)
    return twoFT

def inv_FT(array):
    invft = fft.ifft(array)
    return invft

def inv_twod_ft(array):
    invfttwo = fft.ifft2(array)
    return invfttwo

def sample(array1,array2):
    return np.multiply(array1,array2)

def continuous_uv_plane(array1, array2):
    uv = sig.convolve2d(array1, array2, "same")
    return uv

def dirty_beam(array):
    dirty_beam = fft.fft2(array)
    return dirty_beam

def discrete_uv_plane(array1, array2):

    n= len(array1)

    b =np.zeros(n)

    for element in xrange(n):
        if array1[element] == 0:
            b[element] = 0
            
        elif array1[element] == 1:
            for i in xrange(n):
                if array2[i] ==1:
                    #how to define "point"?
                    # append the new array with point = (x1-x2/0.21, y1-y2/0.21)
                    b[point] = 1
                
         
                elif i ==0:
                    b[element] = 0

def basic_interferometer():

    #take input from wherever
    
    galaxy = create_disc(8,8,4,5,2,15) #basic example
    
    ftgal = twod_FT(galaxy) # or input

    #make_antennas

    uvplane = continuous_uv_plane(VLA, VLA) # take output of make_antennas

    #uvplane = discrete_uv_plane(VLA,VLA)

    sampled_sky = sample(ftgal, uvplane)

    dirty_image = inv_twod_ft(sampled_sky)

    #dirty_image_view = make_image(dirty_image)

    #clean_image = deconvolve(dirty_image)

    
#def deconvolve(array):
    
#def Earth_rotation_synthesis(array): DO I ADD INTO THE UV PLANE METHODS?






    






    

    
