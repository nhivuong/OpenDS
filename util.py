
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter

from featureMap import CFMcompute, CSFcompute
from GaborKern import getGaborKernels

def calculate(mat, thresh):
    (w,h) = mat.shape
    sum_local_max = mat[0][0]
    count_local_max = 0
    global_max = mat[0][0]
    for i in range(1, w-1):
        for j in range(1, h-1):
            if mat[i][j] > max(mat[i-1][j-1],mat[i-1][j],mat[i-1][j+1],
                               mat[i][j-1],              mat[i][j+1],
                               mat[i+1][j-1],mat[i+1][j],mat[i+1][j+1]) and mat[i,j]>thresh:
                if mat[i][j] > global_max:
                    global_max = mat[i][j]

                sum_local_max += mat[i][j]
                count_local_max +=1

    if count_local_max > 0:
        local_max_avg = float(sum_local_max)/float(count_local_max)
    else:
        local_max_avg = 0.0
    return global_max, count_local_max, local_max_avg


def processNormalization(mat):
    #M = 10
    M=90
    thresh  = M/10
    mat = cv2.normalize(mat, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mat = mat*M
    g_max, c_max, l_max_avg = calculate(mat, thresh)
 
    if c_max>1:
        res = mat* (M-l_max_avg)**2
    elif c_max == 1:
        res = mat * M**2
    else:
        res = mat

    return res

def process2(mat):
    #M = 8.0 # an arbitrary global maxima for which the image is scaled
    M = 8.0
    mat = cv2.convertScaleAbs(mat, alpha=M/mat.max(), beta = 0.0)
    w, h = mat.shape
    maxima = maximum_filter(mat, size=(1, 1))
    maxima = (mat == maxima)
    mnum = maxima.sum()
    maxima = np.multiply(maxima, mat)
    mbar = float(maxima.sum()) / mnum
    return mat * (M-mbar)**2


def OFMcompute(L, gaborparams, thetas):
    '''
        Orientation Feature Map
        L = Intensity Map
    '''
    # L = np.maximum(np.maximum(r, g), b)

    kernels = getGaborKernels(gaborparams, thetas)
    featMaps = []
    for th in thetas:
        kernel_0  = kernels[th]['0']
        kernel_45 = kernels[th]['45']
        kernel_90 = kernels[th]['90']
        kernel_135 = kernels[th]['135']
        
        # orientations of 0, 45, 90 and 135 degrees
        o1 = cv2.filter2D(L, -1, kernel_0, borderType=cv2.BORDER_REPLICATE)
        o2 = cv2.filter2D(L, -1, kernel_45, borderType=cv2.BORDER_REPLICATE)
        o3 = cv2.filter2D(L, -1, kernel_90, borderType=cv2.BORDER_REPLICATE)
        o4 = cv2.filter2D(L, -1, kernel_135, borderType=cv2.BORDER_REPLICATE)
        p1 = np.add(abs(o1), abs(o2))
        p2 = np.add(abs(o3), abs(o4))
        o = np.add(abs(p1), abs(p2))
        featMaps.append(o)

    return featMaps

def norm01(mat):
    return cv2.normalize(mat, None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def calculateFeatureMaps(r, g, b, L, params):
    colorMaps = CFMcompute(r, g, b, L)
    orientationMaps = OFMcompute(L, params['gaborparams'] , params['thetas'])
    allFeatureMaps = {
        0: colorMaps[0],
        1: colorMaps[1],
        2: colorMaps[2],
        3: orientationMaps
    }
    return allFeatureMaps

def getPyramid(image, max_level):
    imagePyramid = {
        0: image
    } # scale zero = 1:1

    for i in range(1, max_level):
        imagePyramid[i] = cv2.pyrDown(imagePyramid[i-1])

    return imagePyramid