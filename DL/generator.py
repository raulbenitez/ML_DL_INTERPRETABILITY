from cgitb import small
from skimage import data, measure
import skimage as sk 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.transform import rotate

def generate_small_blobs(length = 64, blob_size_fraction = 0.1,
                   n_dim = 2,
                   volume_fraction = 0.2, seed = None):
  rs = np.random.default_rng(seed)
  shape = tuple([length] * n_dim)
  mask = np.zeros(shape)
  n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
  points = (length * rs.random((n_dim, n_pts))).astype(int)
  mask[tuple(indices for indices in points[:,:])] = 1

  mask = gaussian(mask, sigma=0.25 * length * blob_size_fraction,
                  preserve_range=False)

  threshold = np.percentile(mask, 100 * (1 - volume_fraction))
  return np.logical_not(mask < threshold)

def generate_big_blob(img_size = 64, avg_size = 10, random_size_range = 0):
  img = np.zeros((img_size, img_size))
  posx = np.random.randint(img_size)
  posy = np.random.randint(img_size)
  blob_size = np.random.randint(avg_size-random_size_range, avg_size+random_size_range+1)
  rr, cc = sk.draw.disk((posy, posx), blob_size, shape=(img_size, img_size))

  img[rr,cc] = 1
  return img

def generate_blob_img(big = True, length = 64, blob_size_fraction = 0.08,
                      n_dim = 2, volume_fraction = 0.3, seed = None,
                      big_blob_size = 8, big_blob_range=2):
  small_blobs = generate_small_blobs(length, blob_size_fraction, 
                                     n_dim, volume_fraction, seed) 
  if big:
    big_blob = generate_big_blob(length, big_blob_size, big_blob_range)
    return np.logical_or(small_blobs, big_blob)
  else:
    return small_blobs

def remove_small_blobs(img, small_size = 9):
    return remove_small_objects(img, small_size)

def generate_big_blobs(num_blobs = 1, img_size = 64, avg_size = 10, random_size_range = 0):
  img = np.zeros((img_size, img_size))
  for i in range(num_blobs):
    posx = np.random.randint(img_size)
    posy = np.random.randint(img_size)
    blob_size = np.random.randint(avg_size-random_size_range, avg_size+random_size_range+1)
    rr, cc = sk.draw.disk((posy, posx), blob_size, shape=(img_size, img_size))

    img[rr,cc] = 1
  return img

def generate_big_blob2(img_size = 64, maj_axis=15, min_axis=5):
  img = np.zeros((img_size, img_size))
  #size_maj = np.random.randint(avg_size, avg_size+random_size_range+1)
  #size_min = np.random.randint(avg_size-random_size_range, avg_size)
  #posx = np.random.randint(np.max([size_min, size_maj]), img_size - np.max([size_maj, size_min]))
  #posy = np.random.randint(np.max([size_min, size_maj]), img_size - np.max([size_maj, size_min]))
  posx = np.random.randint(maj_axis, img_size-maj_axis)
  posy = np.random.randint(maj_axis, img_size-maj_axis)
  rr, cc = sk.draw.ellipse(posy, posx, maj_axis, min_axis, shape=(img_size, img_size), rotation=np.random.randint(-15, 15)/10)

  img[rr,cc] = 1
  return img


def generate_small_blobs2(length = 64, blob_size_fraction = 0.1,
                   n_dim = 2,
                   volume_fraction = 0.2, seed = None, randomize_sigma = True, rotation = True):
  rs = np.random.default_rng(seed)
  shape = length
  n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
  mask = np.zeros((shape, shape, n_pts))
  points = (length * rs.random((n_dim, n_pts))).astype(int)

  sigma_factor_x = 0.25
  sigma_factor_y = 0.25

  for i in range(n_pts):
    mask[points[:,i][0], points[:,i][1], i] = 1
    if randomize_sigma:
      sigma_factor_x = (np.random.randint(low = 10, high = 50)/100)
      sigma_factor_y = (np.random.randint(low = 10, high = 50)/100)

    mask[:,:,i] = gaussian(mask[:,:,i], sigma=[sigma_factor_x * length * blob_size_fraction, sigma_factor_y * length * blob_size_fraction],
                    preserve_range=False)
  
    angle = np.random.randint(low = 0, high = 180)
    if rotation:
      mask[:,:,i] = rotate(mask[:,:,i], angle)
  mask = mask.sum(axis = -1)

  threshold = np.percentile(mask, 100 * (1 - volume_fraction))
  return np.logical_not(mask < threshold)

def generate_new_blob_img(ellipse = True, size = 64, maj_axis = 10, min_axis = 3, num_big_blobs = 5):
    r = np.floor(np.sqrt(maj_axis*min_axis))
    x = generate_circles_and_ellipse(ellipse, num_big_blobs, size, maj_axis, min_axis)
    y = generate_small_blobs(size, blob_size_fraction = 0.06,
                      n_dim = 2, volume_fraction = 0.2)

    return np.logical_or(x,y)

def generate_circles_and_ellipse(ellipse = True, num_blobs = 10, img_size = 64, maj_axis=15, min_axis=5):
  img = np.zeros((img_size, img_size))
  r = np.ceil(np.sqrt(maj_axis*min_axis))
  posx = np.random.randint(r, img_size-r)
  posy = np.random.randint(r, img_size-r)
  if ellipse:
    rr, cc = sk.draw.ellipse(posy, posx, maj_axis, min_axis, shape=(img_size, img_size), rotation=np.random.randint(-15, 15)/10)
    img[rr,cc] = 1
    num_blobs-=1
  for i in range(num_blobs):
      #print(posx-r, posx+r, posy-r, posy+r)
      while 1 in img[int(posy-r):int(posy+r), int(posx-r):int(posx+r)]:
        #print(img[int(posx-r):int(posx+r), int(posy-r):int(posy+r)])
        posx = np.random.randint(r, img_size-r)
        posy = np.random.randint(r, img_size-r)
      rr, cc = sk.draw.ellipse(posy, posx, r, r, shape=(img_size, img_size))
      img[rr,cc] = 1
  # while 1 in img[int(posy-maj_axis):int(posy+maj_axis), int(posx-maj_axis):int(posx+maj_axis)]:    
  #   posx = np.random.randint(maj_axis, img_size-maj_axis)
  #   posy = np.random.randint(maj_axis, img_size-maj_axis)
  return img
