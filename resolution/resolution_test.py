import numpy as np
import cv2
import streamlit as st
from termcolor import colored
import os
import math
import matplotlib.pyplot as plt
from util.legacy_writer import legacyWriter


debug=True

def get_data(dcm_slice):
    """Function to get data and rescale to standard TC values"""
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    st.write(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing



def select_roi(img, c_x, c_y, diameter, pix_dim, save=False, outname="roi"):

    """This function take as input img, x,y,d and generate a circular ROI"""
    print(f"[INFO] selecting the ROI with center {c_x, c_y} with {diameter} mm of diameter")
    st.write(f"[INFO] selecting the ROI with center {c_x, c_y} with {diameter} mm of diameter")
    d = int(diameter // pix_dim[0])

    # get the center
    # c_x,c_y=img.shape[0]//2,img.shape[1]//2
    # create a rectungular roi

    # generate rect mask with value -3000

    background_mask = np.zeros(img.shape)

    mask = cv2.circle(background_mask, (c_x, c_y), d // 2, color=(255, 255, 255), thickness=-1)
    mask = mask.astype("uint8")

    res = cv2.bitwise_and(img, img, mask=mask)

    # generate a lookup table to keep trace of values inside the selected roi
    back_circle = cv2.circle(background_mask, (c_x, c_y), d // 2, color=(255, 255, 255), thickness=-1)
    back_circle = back_circle.astype("uint8")
    lookup = cv2.bitwise_and(background_mask, background_mask, mask=back_circle)

    lookup = lookup != 255

    if save:
        # just for debug
        normalized = np.zeros(img.shape)
        cv2.normalize(img, normalized, 0, 255, cv2.NORM_MINMAX)
        # normalized=np.expand_dims(normalized,-1)
        normalized = normalized.astype("uint8")
        colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        mask = cv2.circle(colored, (c_x, c_y), d // 2, color=(0, 255, 0), thickness=1)
        cv2.imwrite(os.path.join("cq_images", outname + ".png"), mask)

        print(f"[INFO] Area of this ROI: {math.pi * (diameter / 2) ** 2} mm^2")
        st.write(f"[INFO] Area of this ROI: {math.pi * (diameter / 2) ** 2} mm^2")
    return res, lookup


def mask_values(img,lookup):
    """This function takes the img and a mask and return a masked array"""
    #mask values to ensure that just values in the roi will be used for calculations
    values=np.reshape(img,(img.shape[0]*img.shape[1],1))
    mask=np.reshape(lookup,(img.shape[0]*img.shape[1],1))
    return np.ma.masked_array(values,mask=mask)



def clean_freq_rois(minRect, max_ratio=1.5, max_width=250, align=True):
    # this function take as input the minRect list, squarify them if their width/height ratio is lower than max_ratio
    print(f"[INFO] squaring all rois, eventually dropping the biggest one that rapresents the phantom")
    st.write(f"[INFO] squaring all rois, eventually dropping the biggest one that rapresents the phantom")
    good_roi = []
    for rect in minRect:
        (x, y), (w, h), o = rect
        if max(w, h) / min(w, h) < max_ratio and w < max_width:
            l = min(w, h)
            good_roi.append(((x, y), (l, l), o))

    ordered = sorted(good_roi, key=lambda x: x[1][0] * x[1][0], reverse=True)

    (x, y), (w, h), o = ordered[0]
    circle = ((int(x), int(y)), int(w // 2))  # center and radius of circle
    squared = ordered[1:]

    if align:
        print(f"[INFO] roi alignment..")
        st.write(f"[INFO] roi alignment..")
        angles = [el[2] for el in squared]
        mean_angle = np.mean(angles)
        squared = [(el[0], el[1], mean_angle) for el in squared]

    return squared, circle




def prepare_checkcontrast_rois(circle,img,pix_dim,diameter=20):
    print(f"[INFO] generating simmetric rois to check contrast with {diameter} mm of diamter...")
    st.write(f"[INFO] generating simmetric rois to check contrast with {diameter} mm of diamter...")
    water_circle=(int(circle[0][0]),int(circle[0][1]),int(diameter//pix_dim[0]))
    x,y=circle[0]
    simm_x=int(img.shape[0]//2-(x-img.shape[0]//2))
    simm_y=int(img.shape[1]//2-(y-img.shape[1]//2))
    simmetric_circle=(simm_x,simm_y,int(diameter//pix_dim[0]))
    return water_circle,simmetric_circle

def check_contrast(wc,sc,img,pix_dim,target_diff=118,relative_displacement=0.1):
    passed=False
    water,water_look=select_roi(img,wc[0],wc[1],20,pix_dim)
    water=mask_values(water,water_look)

    pmm,pmm_look=select_roi(img,sc[0],sc[1],20,pix_dim)
    pmm=mask_values(pmm,pmm_look)

    w_mean=np.mean(water)
    p_mean=np.mean(pmm)
    difference=p_mean-w_mean
    if difference>target_diff*(1-relative_displacement) and difference<target_diff*(1+relative_displacement):
        passed=True
        print(f"[INFO] contrast test: {colored('[SUCCESS]','green')} with difference between PMMA and water= {difference}")
        st.markdown(f"[INFO] contrast test: <font color='green'>[SUCCESS]</font> with difference between PMMA and water= {difference}",unsafe_allow_html=True)
    else:
        print(f"[WARNING] contrast test: {colored('[FAILED]','red')} with difference between PMMA and water= {difference}")
        st.markdown(f"[WARNING] contrast test: <font color='red'>[FAILED]</font> with difference between PMMA and water= {difference}",unsafe_allow_html=True)

    return passed,p_mean,w_mean, np.std(pmm),np.std(water)







def mask_square(img, squares):
    """This function mask the squares for resolution check"""

    boxes = []
    lookups = []
    print(f"[INFO] masking all the squares...")
    st.write(f"[INFO] masking all the squares...")
    for sq in squares:
        lookup = np.ones(img.shape)
        box = cv2.boxPoints(sq)
        box = np.intp(box)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                mask = cv2.pointPolygonTest(box, (j, i), False)
                if mask > 0:
                    lookup[i, j] = 0
        lookups.append(lookup)

    for b, l in zip(squares, lookups):
        boxes.append(np.ma.masked_array(img.copy(), l))
    return boxes


def check_resolution(boxes, diff, min_value=34.1, max_value=41.7):
    means = [np.mean(i) for i in boxes]
    stds = [np.std(i) for i in boxes]
    MTF = [222 * j / diff for j in stds]
    passed = False
    if stds[0] > min_value and stds[0] < max_value:
        print(f"[INFO] resolution test: {colored('[SUCCESS]', 'green')} with value {stds[0]}")
        st.markdown(f"[INFO] resolution test: <font color='green'>[SUCCESS]</font> with value {stds[0]}",unsafe_allow_html=True)
        passed = True
    else:
        print(f"[WARNING] resolution test: {colored('[FAILED]', 'red')} with value {stds[0]}")
        st.markdown(f"[WARNING] resolution test: <font color='red'>[FAILED]</font> with value {stds[0]}",unsafe_allow_html=True)

    return passed, means, stds, MTF










def test_contrast_resolution(dcm_img,legacy=None,sheet=None,filter="std",debug=True):
    #ris_img = pydicom.dcmread(path)  # load image for linear spacing
    img, pix_dim = get_data(dcm_img)  # rescale img

    print(f"[INFO] Normalizing image..")
    st.write(f"[INFO] Normalizing image..")
    gray = img.copy()

    # normalize img
    dst = np.zeros(gray.shape)

    # drop the background
    gray[gray < 0] = 0
    cv2.normalize(gray, dst, 0, 255, cv2.NORM_MINMAX)

    print(f"[INFO] Looking for contours..")
    st.write(f"[INFO] Looking for contours..")

    dst = dst.astype("uint8")
    display = dst.copy()  # to show at the end

    if filter=="bone":
        dst = cv2.GaussianBlur(dst, (5, 5), 1.2)
        ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY_INV)
        min_area=100

    elif filter=="std":
        dst = cv2.GaussianBlur(dst, (5, 5), 1.21)
        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY_INV)
        min_area = 80

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    big_conts = []
    for cnt, area in zip(contours, areas):
        if area > min_area:
            big_conts.append(cnt)

    # LAVORARE QUA SUI CONTORNI -> RENDERLI QUADRATI MAGARI RUOTATI, SCARTARE QUELLI ESTERNI comunque è buono

    print(f"[INFO] Generating miniminum enclosing rectangles..")
    st.write(f"[INFO] Generating miniminum enclosing rectangles..")
    minRect = [None] * len(big_conts)
    for i, c in enumerate(big_conts):
        minRect[i] = cv2.minAreaRect(c)

    squares, circle = clean_freq_rois(minRect, align=True)
    wc,sc=prepare_checkcontrast_rois(circle,img,pix_dim)


    #here do contrast test
    passed,p,w,std_p,std_w=check_contrast(wc, sc, img, pix_dim)


    ##HERE START THE RESOLUTION TEST
    #roi 6 is w, roi 7 is pmm

    print(f"[INFO] Proceeding with resolution test, found {len(squares)} boxes..")
    st.write(f"[INFO] Proceeding with resolution test, found {len(squares)} boxes..")
    if filter=="bone":
        sorted_squares = sorted(squares, key=lambda x: x[0][0], reverse=True)[:-1]
    elif filter=="std":
        center_squares=[]
        for i in range(len(squares)):
            if squares[i][0][0]>150 and squares[i][0][0]<350:
                #keep only the centered squares
                center_squares.append(squares[i])
        sorted_squares = sorted(center_squares, key=lambda x: x[0][0], reverse=True)


    if debug:
        print(f"[DEBUG] showing squares ..")

        drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        for i, c in enumerate(squares):
            # contour
            # cv2.drawContours(display, squares, i, (255,255,255))
            # ellipse
            # if c.shape[0] > 5:
            #    cv2.ellipse(display, minEllipse[i],(255,255,255), 2)
            # rotated rectangle
            box = cv2.boxPoints(squares[i])
            box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
            cv2.drawContours(display, [box], 0, (255, 255, 255))

        cv2.circle(display, (wc[0], wc[1]), wc[2], (255, 255, 255), 1)
        cv2.circle(display, (sc[0], sc[1]), sc[2], (255, 255, 255), 1)

        fig, ax = plt.subplots()
        ax.imshow(display, cmap="gray")
        st.write(fig)
        #cv2.imshow('Spatial ROIs', display)
        #cv2.waitKey(0)


    boxes=mask_square(img,sorted_squares)
    if filter=="std":
        passed,means,stds,MTFS=check_resolution(boxes, p - w, min_value=34.1,max_value=41.7)
    elif filter=="bone":
        passed,means,stds,MTFS=check_resolution(boxes, p - w, min_value=46.9, max_value=57.3)


    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        st.write(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        leg = legacyWriter(legacy, sheet)
        leg.write_resolution_report(means,w,p,stds,std_w,std_p,filter)
