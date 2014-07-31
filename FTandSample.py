import numpy as np
import numpy.fft as fft
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.misc as misc


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
    imgplot = plt.imshow(np.absolute(array), cmap = "gist_yarg")
    imgplot.set_clim(0.001, 0.002)
    plt.colorbar()
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
                    
    return b"""
                


def discrete_uv_VLA(antennas, nuv):  #a method to get the discrete uv sampling distribution for the VLA telescope. will generalise for other telescopes soon
    antennas = antennas()
    bl = []
    for x,y in antennas:
        for x2, y2 in antennas:
            u = (x - x2)/redshifted_lambda(1)
            v = (y - y2)/redshifted_lambda(1)
            bl.append((u,v))
            
    bl = np.array(bl)

    maxx, maxy = np.max(bl[:,0]), np.max(bl[:,1])
    minx, miny =  np.min(bl[:,0]), np.min(bl[:,1])

    length = maxx - minx
    array = np.zeros((nuv,nuv))

    for (u,v) in bl:
        skypoint = np.array([[u],[v]])

        for x in range(0,3,1):
            rotation_array = np.array([[np.sin(x), np.cos(x)], [np.cos(x), (-1*np.sin(x))]])
            rotated = np.dot(rotation_array, skypoint)
            p = np.absolute(rotated[0])
            q = np.absolute(rotated[1])
            k = int((nuv/2) -1 + (p/(length/nuv)))
            l = (nuv/2) -1 + (q/(length/nuv)

          
            
           array[k,l] = 1

    return array


    
        
    

    

            
         
    
    
    

def small_interferometer(nuv,r): #<100 antennae

    centre = nuv/2
    
    galaxy = create_disc(nuv,nuv,centre,centre,r,nuv)

    telescope = VLA_D_config

    ftgal = fft.fft2(galaxy)
    
    uvplane = discrete_uv_VLA(telescope, nuv)
    
    sampled_sky = sample(ftgal,uvplane)
    
    dirty_image = fft.ifft2(sampled_sky)

    dirty_image_view = make_image(ers)


def large_interferometer(filename): #large >100 antennae

    galaxy = make_image(filename)
    
    ftgal = twod_FT(galaxy) 

    telescope = VLA_D_config

    uvplane = continuous_uv_plane(telescope, telescope) # take output of make_antennas

    sampled_sky = sample(ftgal, uvplane) 


    dirty_image = inv_twod_ft(sampled_sky)

    dirty_image_view = make_image(dirty_image)

    #clean_image = deconvolve(dirty_image)


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

    
#def deconvolve(array):
    
#def Earth_rotation_synthesis(array): DO I ADD INTO THE UV PLANE METHODS?






    






    

    
