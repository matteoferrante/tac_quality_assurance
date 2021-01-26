import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
from termcolor import colored
import os
import math
import glob
import pandas as pd

from util.legacy_writer import legacyWriter
import streamlit as st

def get_data(dcm_slice):
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    st.write(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing

standard_circles=[(256,256,13), (256,415,13),(415,256,13),(95,256,13),(256,95,13)]
tollerance=5


def find_circles(test):
    gray = cv2.GaussianBlur(test.astype("uint8"), (5, 5), 0)
    # normalize img
    dst = np.zeros(gray.shape)
    dst = cv2.normalize(gray, dst, 0, 255, cv2.NORM_MINMAX)

    dst = dst.astype("uint8")
    display = dst.copy()
    rows = gray.shape[0]
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY_INV)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 10, 100,
                               param1=100, param2=100,
                               minRadius=4, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            # cv2.circle(dst, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(display, center, radius, (0, 0, 0), 3)
    return circles[0]


def clean_circles(circles,standard_circles):
    #this function find the closest to standard values, fill the omitted values and drop the other, return a 5 element array
    cleaned={}
    center,nord,est,ovest,sud=standard_circles
    cleaned["center"]=find_nearest(circles,center)
    cleaned["nord"]=find_nearest(circles,nord)
    cleaned["est"]=find_nearest(circles,est)
    cleaned["ovest"]=find_nearest(circles,ovest)
    cleaned["sud"]=find_nearest(circles,sud)
    return cleaned


def find_nearest(a, a0, tollerance=10):
    "Element in nd array `a` closest to the  value `a0`"
    d = []
    for x in a:
        d.append(np.linalg.norm(x[:2] - a0[:2]))

    idx = np.argmin(d)
    if d[idx] > tollerance:
        print(f"[INFO] nearest not found using default..")
        st.write(f"[INFO] nearest not found using default..")
        return a0
    else:

        print(f"[INFO] nearest found..")
        st.write(f"[INFO] nearest found..")
        return a[idx]


def select_roi(img, c_x, c_y, d):
    print(f"[INFO] selecting the ROI with center {c_x, c_y} with {d} as diameter")
    st.write(f"[INFO] selecting the ROI with center {c_x, c_y} with {d} as diameter")
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

    return res, lookup


def mask_values(values,lookup):
    #mask values to ensure that just values in the roi will be used for calculations
    values=np.reshape(values,(values.shape[0]*values.shape[1],1))
    mask=np.reshape(lookup,(values.shape[0]*values.shape[1],1))
    return np.ma.masked_array(values,mask= mask)


def check_iodine(images, cir, baseline={"center": 60, "nord": 66, "sud": 63, "est": 66, "ovest": 66},
                 default_radius=11):
    print(f"[INFO] Iodine test..")
    st.write(f"[INFO] Iodine test..")
    results = []
    for (i, c) in enumerate(cir):
        res = {}
        print(f"[INFO] testing {i + 1}/{len(cir)} slice..")
        st.write(f"[INFO] testing {i + 1}/{len(cir)} slice..")
        img = images[i]
        for (k, v) in c.items():
            roi, lookup = select_roi(img, c[k][0], c[k][1], 2 * default_radius)
            res[k] = np.mean(mask_values(roi, lookup))
        results.append(res)
    df = pd.DataFrame.from_dict(results)
    means = df.mean().to_dict()

    delta = {}
    passed = {}
    for (k, v) in baseline.items():
        delta[k] = abs(means[k] - baseline[k]) / baseline[k]
        if delta[k] < 0.10:
            passed[k] = True
            print(f"[INFO] testing {k} iodine means: {colored('[SUCCESS]', 'green')} with delta: {delta[k]}")
            st.markdown(f"[INFO] testing {k} iodine means: <font color='green'>[SUCCESS]</font> with delta: {delta[k]}",
                        unsafe_allow_html=True)
        else:
            passed[k] = False
            print(f"[WARNING] testing {k} iodine means: {colored('[FAILED]', 'red')} with delta: {delta[k]}")
            st.markdown(f"[WARNING] testing {k} iodine means:  <font color='red'>[FAILED]</font> with delta: {delta[k]}",
                        unsafe_allow_html=True)

    return passed, results, df, means



def test_iodine(im_list,legacy,sheet):

    print(f"[INFO] IODINE TEST")

    loaded_images = [get_data(image)[0] for image in im_list]
    pix_dim = im_list[0].PixelSpacing
    images = loaded_images[:]
    images=[img.astype("uint32") for img in images]
    standard_circles=[(256,256,13), (256,415,13),(415,256,13),(95,256,13),(256,95,13)]
    tollerance=5

    cir = []
    dis = []
    for img in images:
        found = find_circles(img)
        final_circles = clean_circles(found, standard_circles)
        cir.append(final_circles)

    passed, results, df, means = check_iodine(loaded_images, cir)
    st.write(df)

    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        leg=legacyWriter(legacy,sheet)
        leg.write_iodine_report(results)