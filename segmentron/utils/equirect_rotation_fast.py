import os
import sys
import time
import numpy as np
from PIL import Image
import math
from numba import jit


def Rot_Matrix(rotation, unit='degree'):
    rotation = np.array(rotation, dtype=float)

    if(unit == 'degree'):
        # input's unit of cos, sin function is radian
        rotation = np.deg2rad(rotation)
    elif(unit != 'rad'):
        print("ParameterError: "+unit+"is wrong unit!")
        return
    Rx = np.array([[1, 0, 0], [0, math.cos(rotation[0]), -math.sin(rotation[0])],
                   [0, math.sin(rotation[0]), math.cos(rotation[0])]])
    Ry = np.array([[math.cos(rotation[1]), 0, math.sin(rotation[1])],
                   [0, 1, 0], [-math.sin(rotation[1]), 0, math.cos(rotation[1])]])
    Rz = np.array([[math.cos(rotation[2]), -math.sin(rotation[2]), 0],
                   [math.sin(rotation[2]), math.cos(rotation[2]), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rx, Ry), Rz)
    return R


def Pixel2LonLat(equirect):
    # LongLat - shape = (N, 2N, (Long, Lat))
    W = equirect.shape[1]
    H = equirect.shape[0]
    Lon = np.array([2*(x/W-0.5)*math.pi for x in range(W)])
    Lat = np.array([(0.5-y/H)*math.pi for y in range(H)])

    Lon = np.tile(Lon, (H, 1))
    Lat = np.tile(Lat.reshape(H, 1), (W))

    LonLat = np.dstack((Lon, Lat))
    return LonLat


def LonLat2Sphere(LonLat):
    x = np.cos(LonLat[:, :, 1])*np.cos(LonLat[:, :, 0])
    y = np.cos(LonLat[:, :, 1])*np.sin(LonLat[:, :, 0])
    z = np.sin(LonLat[:, :, 1])

    xyz = np.dstack((x, y, z))
    return xyz


def Sphere2LonLat(xyz):
    Lon = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
    Lat = math.pi/2 - np.arccos(xyz[:, :, 2])

    LonLat = np.dstack((Lon, Lat))
    return LonLat


def LonLat2Pixel(LonLat):
    width = LonLat.shape[1]
    height = LonLat.shape[0]
    j = (width*(LonLat[:, :, 0]/(2*np.pi)+0.5)) % width
    i = (height*(0.5-(LonLat[:, :, 1]/np.pi))) % height

    ij = np.dstack((i, j)).astype('int')
    return ij


@jit
def proccesing(src, src_Pixel):
    out = np.zeros_like(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            pixel = src_Pixel[i][j]
            out[i][j] = src[pixel[0]][pixel[1]]
    return out


def Rot_Equirect(src, rotation=(0, 0, 0)):
    src = np.array(src)
    R = Rot_Matrix(rotation)

    out = np.zeros_like(src)
    out_LonLat = Pixel2LonLat(out)
    out_xyz = LonLat2Sphere(out_LonLat)

    src_xyz = np.zeros_like(out_xyz)
    src_xyz = np.einsum("ka,ijk->ija", R, out_xyz)

    src_LonLat = Sphere2LonLat(src_xyz)
    src_Pixel = LonLat2Pixel(src_LonLat)

    out = proccesing(src, src_Pixel)
    return out


if __name__ == "__main__":
    start = time.time()
    src = Image.open(sys.argv[1])
    out = Rot_Equirect(src, (sys.argv[2], sys.argv[3], sys.argv[4]))
    print(time.time()-start)
    Image.fromarray(out).save(sys.argv[5])

# python euiqrect_rotate_fast.py img X Y Z out
