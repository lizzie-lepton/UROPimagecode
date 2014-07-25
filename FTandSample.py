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



<<<<<<< HEAD
def create_disc(j, k, a, b, r, n):
    array = np.zeros((j,k))
=======


def create_disc(j, k, l, a, b, r, n):
    array = np.zeros((j,k,l))
>>>>>>> 9fb3d7058ef7b3f4913efa3949d421ecdc862b32
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    array[mask]= 1
    return array

<<<<<<< HEAD
def make_antennas(n,m, p, q, r, z):
    array = np.zeros((n,m))
    y,x = np.ogrid[-p:z-p, -p:z-p]
    mask1 = mask = x*x + y*y <= r*r
    
    array[mask1]= 1
=======
def make_antennas(n,m,o, p, q):
    array = np.zeros((n,m,o))
    y,x = np.ogrid[0:n, 0:n]
    mask1 = y == p
    mask2 = x==q
    array[mask1]= 1
    array[mask2] = 1
>>>>>>> 9fb3d7058ef7b3f4913efa3949d421ecdc862b32
    return array
    

def image_input(filename):
    img = mpimg.imread(filename)
    return img

    

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
<<<<<<< HEAD
    array1 = np.real(array1)
    array2 = np.real(array2)
=======
>>>>>>> 9fb3d7058ef7b3f4913efa3949d421ecdc862b32
    return np.multiply(array1,array2)

def continuous_uv_plane(array1, array2):
    uv = sig.convolve(array1, array2, "same")
    return uv

def dirty_beam(array):
    dirty_beam = fft.fft2(array)
    return dirty_beam

<<<<<<< HEAD
def discrete_uv_plane(array):

    dim = array.shape

    b =np.zeros(dim)

    for (x1,y1), value in np.ndenumerate(array):
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
                    
    return b
                
            
    
                    
                

def small_interferometer(): #<100 antennae
    galaxy = create_disc(8,9,4,5,2,15)
    
    telescope = make_antennas(8,9,4,5)
    
    #ftgal = twod_FT(galaxy)
    
    #uvplane = discrete_uv_plane(telescope)
    
    #sampled_sky = sample(ftgal, uvplane)
    
    #dirty_image = inv_twod_ft(sampled_sky)

   # dirty_image_view = make_image(dirty_image)
=======
def discrete_uv_plane(array1, array2):

    n= len(array1)

    b =np.zeros(n)

    for element in xrange(n):
        x1,y1 = np.ogrid[0:n,0:n]
        if array1[element] == 0:
            b[element] = 0
            
        elif array1[element] == 1:
            x2,y2 = np.ogrid[0:n,0:n]
            for i in xrange(n):
                if array2[i] ==1:
                    x2 = i
                    point = (x1[element]-x2[i])/redshifted_lambda(1), (y1[element]-y2[i])/redshifted_lambda(1)
                    b[point] = 1
                elif i ==0:
                    b[element] = 0



def small_interferometer(): #<100 antennae
    galaxy = create_disc(8,8,4,5,2,15)
    
    telescope = make_antennas
    
    ftgal = twod_FT(galaxy)
    
    uvplane = discrete_uv_plane(telescope, telescope)
    
    sampled_sky = sample(ftgal, uvplane)
    
    dirty_image = inv_twod_ft(sampled_sky)

    dirty_image_view = make_image(dirty_image)

>>>>>>> 9fb3d7058ef7b3f4913efa3949d421ecdc862b32
    #clean image


def large_interferometer(filename): #large >100 antennae

    galaxy = make_image(filename)
    
    ftgal = twod_FT(galaxy) 

    telescope = make_antennas

    uvplane = continuous_uv_plane(telescope, telescope) # take output of make_antennas

<<<<<<< HEAD
    sampled_sky = sample(ftgal, uvplane) 
=======
    sampled_sky = sample(ftgal, uvplane)
>>>>>>> 9fb3d7058ef7b3f4913efa3949d421ecdc862b32

    dirty_image = inv_twod_ft(sampled_sky)

    dirty_image_view = make_image(dirty_image)

    #clean_image = deconvolve(dirty_image)

def redshifted_lambda(z):
    lambda_obs = (1+z)* 0.21

    return lambda_obs

    
#def deconvolve(array):
    
#def Earth_rotation_synthesis(array): DO I ADD INTO THE UV PLANE METHODS?






    






    

    
