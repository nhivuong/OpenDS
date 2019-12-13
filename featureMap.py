
import cv2
import numpy as np

def CFMcompute(r, g, b, I):
    ''' 
        Itty Koch Feature Decomposing
        Implementation based on the formula provided in Itti-koch paper.
        Separate into R, G, B color channels as well as Intensity, and Y
        ( combination of R and G)
    '''

    max_I = I.max()
    # normalisation, for decoupling hue from intensity #ittikoch98pami
    r = np.divide(r, I, out=np.zeros_like(r), where=I>max_I/10.)
    g = np.divide(g, I, out=np.zeros_like(g), where=I>max_I/10.)
    b = np.divide(b, I, out=np.zeros_like(b), where=I>max_I/10.)

    # calculating broadly-tuned color channels
    R = r-(g+b)/2.
    R = R*(R>=0)

    G = g-(r+b)/2.
    G = G*(G>=0)

    B = b-(r+g)/2.
    B = B*(B>=0)

    Y = (r+g)/2 - cv2.absdiff(r, g)/2. - b
    Y = Y*(Y>=0)

    RG = cv2.absdiff(R, G)
    BY = cv2.absdiff(B, Y)

    featMaps = {
        0: RG,
        1: BY,
        2: I
    }
    return featMaps

def CSFcompute(feature_pyramids):
    '''
        Center Surround Map
    '''
    center_levels = [2, 3, 4]
    delta = [2, 3]
    Ccs_array = { 0:[], 1:[] }
    Ics_array =[]
    Ocs_array = []
    
    for c in center_levels:
        for d in delta:
            s = c+d
            for i in range(0,2):      
                # For calculating RG and BY channels
                Cc = feature_pyramids[c][i]
                Cs = -feature_pyramids[s][i]
                # to allow for chromatic opponency
                Cs_scaled = cv2.resize(Cs, (Cc.shape[1], Cc.shape[0]), interpolation=cv2.INTER_CUBIC)
                Ccs = (Cc - Cs_scaled) ** 2
                Ccs_array[i].append(Ccs)

            Ic = feature_pyramids[c][2]
            Is = feature_pyramids[s][2]

            Is_scaled = cv2.resize(Is, (Ic.shape[1], Ic.shape[0]), interpolation=cv2.INTER_CUBIC)
            Ics = (Ic - Is_scaled)**2
            Ics_array.append(Ics)

            for idx in range(0, len(feature_pyramids[c][3])):
                Oc = feature_pyramids[c][3][idx]
                Os = feature_pyramids[s][3][idx]

                Os_scaled = cv2.resize(Os, (Oc.shape[1], Oc.shape[0]), interpolation=cv2.INTER_CUBIC)
                Ocs = (Oc - Os_scaled) ** 2
                Ocs_array.append(Ocs)

    final_Feature_Array = {
        0:Ccs_array[0],
        1:Ccs_array[1],
        2:Ics_array,
        3:Ocs_array
    }
    return final_Feature_Array