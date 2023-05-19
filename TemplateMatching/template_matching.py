import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
import time

from utils import Match, CONFIG

def templateMatching(fileName, imgName, mode, templates):
    """ This function reads all jpg from the img folder and runs multi-template matching on it.
    The "coordinates" for teeth (top-left pixel of it) is outputted in a CSV file. Copies of the 
    images labelled with the suspected teeth are also generated for reference.
    
    Note:
    1. This function detects objects similar in size to the list of templates provided. It does not scale the template
    so make sure that the target object is similar in size to how it appears in the image
    2. Possible algorithms for template matching: [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, 
    cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]. The current code is taylored 
    to cv2.TM_CCOEFF_NORMED
    3. The folder structure assumed is an img and template folder in the same directory,
    where the img folder has all the images, and template folder all the templates

    Useful Links:
    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_templateMatching.html (a tutorial for template matching)
    https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5be00b45a4d99b5e42625b4400bfde65 (equations for each algorithm)
    """

    if mode == Match.TWO_D:
        threshold = CONFIG.THRESHOLD
        iouThreshold = CONFIG.iouThreshold
    else:
        threshold = CONFIG.THRESHOLD_1D
        iouThreshold = CONFIG.IOU_THRESHOLD_1D


    teeth = []
    for template in templates:
        # load template
        if mode == Match.TWO_D:
            t = cv2.imread(os.path.join("template", template),cv2.IMREAD_GRAYSCALE)
            if os.path.isfile(os.path.join("img", fileName)):
                img = cv2.imread(os.path.join("img", fileName), cv2.IMREAD_GRAYSCALE)
            else: 
                raise RuntimeError(f"{fileName} was not found in /img.")
        else: 
            t = cv2.imread(os.path.join("template 1D", template),cv2.IMREAD_GRAYSCALE)
            if os.path.isfile(os.path.join("processed", "projection", fileName)):
                img = cv2.imread(os.path.join("processed", "projection", fileName), cv2.IMREAD_GRAYSCALE)
            else: 
                raise RuntimeError(f"{fileName} was not found in /processed/projection. Possibly a Hyperbola was not interpolated, or did you run fitProject first? ")

        # load images and dimensions 
        h, w = t.shape

        for method in CONFIG.METHODS:
            img2 = img.copy()
            result = cv2.matchTemplate(img2, t, method)

            #returns locations where result is bigger than THRESHOLD
            loc = np.where(result >= threshold)

            # for each (x,y)
            for pt in zip(*loc[::-1]):
                intersect = False
                for tooth in teeth[::]:
                    if intersectionOverUnion([pt[0], pt[1],w,h,result[pt[1]][pt[0]]], tooth) > iouThreshold:
                        # if a location that intersects has a better matching score, replace
                        if result[pt[1]][pt[0]] > tooth[4]:
                            teeth.remove(tooth)
                        else: 
                            intersect = True
                
                # if no intersection, add to list of teeth
                if not intersect:
                    newTooth = [pt[0], pt[1], w, h, result[pt[1]][pt[0]], template]
                    teeth.append(newTooth)
        

    data = {'x':[],'y':[], 'w':[],'h':[], 'score':[], 'match':[]}
    if mode == Match.ONE_D:
        img = cv2.imread(os.path.join("processed", "projection", fileName), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(os.path.join("img", fileName), cv2.IMREAD_GRAYSCALE)

    for pt in teeth:
        cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + pt[2], pt[1] + pt[3]), (255,255,0), 2)
        data['x'].append(pt[0])
        data['y'].append(pt[1])
        data['w'].append(pt[2])
        data['h'].append(pt[3])
        data['score'].append(pt[4])
        data['match'].append(pt[5])
    df = pd.DataFrame(data=data)
    df.sort_values(by=['x'], inplace=True)


    # saves to coordinates saves marked image in appropriate folders
    currDir = os.getcwd()
    processedDir = os.path.join(currDir,"processed")
    if mode == Match.ONE_D:
        os.chdir(os.path.join(processedDir,"match data 1D"))
    else:
        os.chdir(os.path.join(processedDir,"match data"))
    df.to_csv(f"{imgName}.csv")
    if mode == Match.ONE_D:
        os.chdir(os.path.join(processedDir,"match visualization 1D"))
    else:
        os.chdir(os.path.join(processedDir,"match visualization"))
    cv2.imwrite(fileName, img)
    os.chdir(currDir)



def intersectionOverUnion(p1, p2):
    """
    outputs the intersection area size over the union area size of two boxes. p1 and p2 are the top left
    coordinates of the boxes. 
    """
    #calculation of overlap
    xA = max(p1[0], p2[0])
    yA = max(p1[1], p2[1])
    xB = min(p1[0]+p1[2], p2[0]+p2[2])
    yB = min(p1[1]+p1[3], p2[1]+p2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = p1[2] * p1[3]
    box2Area = p2[2] * p2[3]

    # score of overlap
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou 