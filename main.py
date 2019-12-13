import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
import math
from sklearn.feature_selection import mutual_info_classif

from util import processNormalization, calculateFeatureMaps, norm01, getPyramid
from featureMap import CFMcompute, CSFcompute

def run(image, params):
    b = image[:,:,0]/255.
    g = image[:,:,1]/255.
    r = image[:,:,2]/255.
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.

    b_pyr = getPyramid(b, params['max_level'])
    g_pyr = getPyramid(g, params['max_level'])
    r_pyr = getPyramid(r, params['max_level'])
    I_pyr = getPyramid(I, params['max_level'])

    # calculating scale-wise feature maps
    scaledFeaturePyramids = {}

    for i in range(2, len(b_pyr)):
        p_r = r_pyr[i]
        p_g = g_pyr[i]
        p_b = b_pyr[i]
        p_L = I_pyr[i]

        maps = calculateFeatureMaps(p_r, p_g, p_b, p_L, params)

        scaledFeaturePyramids[i] = maps


    # calculating center surround feature maps
    centerSurroundFeatureMaps = CSFcompute(scaledFeaturePyramids)


    # normalizing activation maps
    normalised_maps =[]
    norm_maps = centerSurroundFeatureMaps.copy()
    for i in range(0,4):
        for mat in norm_maps[i]:
            # Resizing to sigma = 4 maps
            nmap = processNormalization(mat)
            nmap = cv2.resize(nmap, (b_pyr[4].shape[1], b_pyr[4].shape[0]), interpolation=cv2.INTER_CUBIC)
            normalised_maps.append(nmap)


    # combine normalised maps
    comb_maps = []
    cfn = len(norm_maps[0])+len(norm_maps[1])
    ifn = len(norm_maps[2])
    ofn = len(norm_maps[3])

    comb_maps.append(normalised_maps[0])
    for i in range(1, cfn):
        comb_maps[0] = np.add(comb_maps[0], normalised_maps[i])

    comb_maps.append(normalised_maps[cfn])
    for i in range(cfn+1, cfn + ifn):
        comb_maps[1] = np.add(comb_maps[1], normalised_maps[i])

    comb_maps.append(normalised_maps[cfn + ifn])
    for i in range(cfn + ifn + 1, cfn + ifn + ofn):
        comb_maps[2] = np.add(comb_maps[2], normalised_maps[i])


    # normalise top channle maps
    ntcmaps = [None]*3
    for i in range(0,3):
        ntcmaps[i] = processNormalization(comb_maps[i])

    # add all of them
    mastermap = (ntcmaps[0] + ntcmaps[1] + ntcmaps[2])/3.0

    #post processing
    gray = norm01(mastermap)
    # blurred = cv2.GaussianBlur(gray,(3,3), 4)
    # gray = norm01(blurred)
    mastermap_res = cv2.resize(gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    return mastermap_res

def setupParams():
    #'stddev': 2,
    #'elongation': 2,
    #'filterSize': -1,
    gaborparams = {
        'stddev': 4,
        'elongation': 1.0,
        'filterSize': -1,
        'filterPeriod': np.pi
    }
    
    #'sigma_frac_act': 0.15, 
    #'sigma_frac_norm': 0.06,
    params = {
        'gaborparams': gaborparams,
        'sigma_frac_act': 0.16, 
        'sigma_frac_norm': 0.08,
        'max_level': 9,
        'thetas': [0, 45, 90, 135]
    }

    return params


if __name__ == '__main__':
    # All rights belong to: https://github.com/shreelock/gbvs/blob/master/ittikochneibur.py 
    params = setupParams()
   
    img = cv2.imread("./testimages/test3.png", 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # process image to correct channel order

    saliency_map = run(img, params)*255.0

    # Save saliency map into current directory
    cv2.imwrite( "./sal_map.jpg", saliency_map )
 
