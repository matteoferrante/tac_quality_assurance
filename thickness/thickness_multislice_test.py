from scipy.optimize import curve_fit

from util.legacy_writer import legacyWriter
from thickness.thickness_functions import *
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from termcolor import colored


def get_data(dcm_slice):
    """Function to get data and rescale to standard TC values"""
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    st.write(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing





def check_thickness(x, y, reference_value, max_distance=0.2,function="gaussian"):

    """This function take x,y and check fit the desidered function"""
    passed = False
    p0 = [200., -30., 1, 100]
    if function=="gaussian":
        popt, pcov = curve_fit(gauss, x[5:-5], y[5:-5], p0=p0, maxfev=1000)
    elif function=="hyper_gaussian":
        popt, pcov = curve_fit(hyper_gauss, x[5:-5], y[5:-5], p0=p0, maxfev=1000)

    A, mu, sigma, y_off = popt

    FWMH = 2.355 * sigma

    xx=np.linspace(min(x),max(x),len(x))

    y_fit = gauss(xx, A, mu, sigma, y_off)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(xx, y_fit, color="black")
    ax.set_title("Slice Thickness")
    ax.set_ylabel("HU units")
    ax.text(mu-3.5*sigma, 300, f'Fit output:\nmu={round(mu, 2)}\nsigma={round(sigma, 2)}\nfwhm:{round(FWMH, 3)}', style='italic', fontsize=10,
            bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 10})



    print(f"[INFO] FUll Width at Medium Height: {FWMH}")
    st.write(f"[INFO] FUll Width at Medium Height: {FWMH}")
    if FWMH > reference_value * (1 - max_distance) and FWMH < reference_value * (1 + max_distance):
        passed = True
        print(f"[INFO] Slice thickness test: {colored('[SUCCESS]', 'green')}")
        st.markdown(f"[INFO] Slice thickness test: <font color='green'>[SUCCESS]</font>",unsafe_allow_html=True)
    else:
        print(f"[WARNING] Slice thickness test: {colored('[FAILED]', 'red')}")
        st.markdown(f"[WARNING] Slice thickness test: <font color='red'>[FAILED]</font>",unsafe_allow_html=True)

    return passed, FWMH, fig






def test_thickness_multislice(im_list, legacy, sheet, function):
    print(f"[INFO] THICKNESS TEST..")
    st.write(f"[INFO] THICKNESS TEST..")
    #images_list = glob.glob(os.path.join(path, "*.DCM"))

    #dcm_images = [pydicom.dcmread(image) for image in images_list]
    images = [get_data(image)[0] for image in im_list]
    start_x, start_y, w, h = (100, 100, 300, 300)
    cropped = [crop_img(img, start_x, start_y, w, h) for img in images]
    print(f"[INFO] preprocessing images to find the dot..")
    st.write(f"[INFO] preprocessing images to find the dot..")

    normalized = normalize(cropped)
    print(f"[INFO] threesholding images to find the dot..")
    st.write(f"[INFO] threesholding images to find the dot..")

    thresholded=[]
    for i in normalized:
        ret,thresh = cv2.threshold(i,120,255,cv2.THRESH_BINARY)
        thresholded.append(thresh)

    # check out the images that cointains the dot
    dots_img = []
    for i in thresholded:
        if np.max(i) == 255:
            dots_img.append(i)

    print(f"[INFO] find the dot in all images, and use the biggest minRect as ROI for all images...")
    st.write(f"[INFO] find the dot in all images, and use the biggest minRect as ROI for all images...")
    conts = []
    for thresh in dots_img:
        thresh = thresh.astype("uint8")
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conts.append(contours)

    print(f"[INFO] Generating miniminum enclosing rectangles..")
    st.write(f"[INFO] Generating miniminum enclosing rectangles..")
    minRect = [None] * len(conts)
    for i, c in enumerate(conts):
        minRect[i] = cv2.boundingRect(c[0])

    areas = [rect[2] * rect[3] for rect in minRect]
    print(f"[INFO] computing max area rect..")
    st.write(f"[INFO] computing max area rect..")
    rect_roi = minRect[np.argmax(areas)]

    areas = [rect[2] * rect[3] for rect in minRect]
    print(f"[INFO] computing max area rect..")
    st.write(f"[INFO] computing max area rect..")
    rect_roi = minRect[np.argmax(areas)]

    print(f"[INFO] transfer the roi...")
    st.write(f"[INFO] transfer the roi...")
    l = max(rect_roi[2], rect_roi[3])
    if (l>10):
        l=10
    or_x = rect_roi[0]
    or_y = rect_roi[1]

    rect_roi = (or_x, or_y, l, l)
    x = []
    y = []
    x_rect, y_rect, w_rect, h_rect = rect_roi
    cc = []
    for i in range(len(im_list)):
        x.append(im_list[i].SliceLocation)
        cc.append(crop_img(cropped[i], y_rect, x_rect, w_rect, h_rect))
        y.append(np.mean(cc[i]))


    passed, fwhm, fig = check_thickness(x, y, reference_value=2.5, max_distance=0.2,function=function)
    st.write(fig)

    #finally complete with legacy
    if legacy is not None:
        if legacy is not None:
            print(f"[INFO] legacy mode: work on {legacy} in {sheet} \nPlease be careful: this should be the semester dose file..." )
            st.write(f"[INFO] legacy mode: work on {legacy} in {sheet} \nPlease be careful: this should be the semester dose file...")
            leg = legacyWriter(legacy, sheet)
            leg.write_dot_thickness_report(fwhm)
