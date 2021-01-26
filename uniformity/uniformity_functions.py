import os
import cv2
import numpy as np
import math
from termcolor import colored
import streamlit as st

PHANTOM_DIAMETER=215
save_images=False

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




def check_noise(roi, max_std_value=5.1):

    """This function check the NOISE"""
    # check the noise, the std in the roi at the center of water must be below 5.1 HU
    passed = False
    std = np.std(roi)
    if std < max_std_value:
        passed = True
        print(f"[INFO] Noise test:\t\t {colored('[SUCCESS]', 'green')} with std {std}")
        st.markdown(f"[INFO] Noise test:\t\t <font color=‘green’>[SUCCESS]</font> with std {std}",unsafe_allow_html=True)
    else:
        print(f"[WARNING] Noise test:\t\t {colored('[FAILED]', 'red')} not passed with std {std}")
        st.markdown(f"[WARNING] Noise test:\t\t <font color='orange'>[FAILED]</font> with std {std}",unsafe_allow_html=True)

    return passed,std

def check_waterct(roi,max_value=4):

    """CHECK WATER CT"""
    passed=False
    mean=np.mean(roi)
    if mean>-max_value and mean<max_value:
        passed=True
        print(f"[INFO] Water CT number test:\t {colored('[SUCCESS]','green')} with CT number {mean}")

        st.markdown(f"[INFO] Water CT number test:<font color=‘green’>[SUCCESS]</font> with CT number {mean}",unsafe_allow_html=True)
    else:
        print(f"[WARNING] Water CT number test:\t {colored('[FAILED]','red')} not passed CT number {mean}")
        st.markdown(f"[WARNING] Water CT number test:<font color='red'>[FAILED]</font> with CT number {mean}",unsafe_allow_html=True)

    return passed,mean



def make_border_rois(img,pix_dim, diameter=30, distance=20):
    images = []
    rois = []

    pix_distance = int(distance // pix_dim[0])  # distance in pixel
    pix_phantom_radius = int(PHANTOM_DIAMETER // pix_dim[0]) // 2
    d = int(diameter // pix_dim[0])

    # left roi
    c_x = -pix_phantom_radius + pix_distance + d // 2 + img.shape[0] // 2
    c_y = img.shape[1] // 2

    print(f"[INFO] generating left roi with center {c_x, c_y}")
    st.write(f"[INFO] generating left roi with center {c_x, c_y}")
    res, lookup = select_roi(img, c_x, c_y, diameter,pix_dim, save=save_images, outname="left")
    rois.append(mask_values(res, lookup))
    images.append(res)

    # upper roi
    c_x = img.shape[0] // 2
    c_y = pix_phantom_radius - pix_distance - d // 2 + img.shape[1] // 2

    print(f"[INFO] generating upper roi with center {c_x, c_y}")
    st.write(f"[INFO] generating upper roi with center {c_x, c_y}")
    res, lookup = select_roi(img, c_x, c_y, diameter,pix_dim, save=save_images, outname="upper")
    rois.append(mask_values(res, lookup))
    images.append(res)

    # right roi
    c_x = pix_phantom_radius - pix_distance - d // 2 + img.shape[0] // 2
    c_y = img.shape[1] // 2

    print(f"[INFO] generating right roi with center {c_x, c_y}")
    st.write(f"[INFO] generating right roi with center {c_x, c_y}")
    res, lookup = select_roi(img, c_x, c_y, diameter,pix_dim, save=save_images, outname="right")
    rois.append(mask_values(res, lookup))
    images.append(res)

    # bottom roi
    c_x = img.shape[0] // 2
    c_y = pix_phantom_radius - pix_distance - d // 2

    print(f"[INFO] generating bottom roi with center {c_x, c_y}")
    st.write(f"[INFO] generating bottom roi with center {c_x, c_y}")
    res, lookup = select_roi(img, c_x, c_y, diameter,pix_dim, save=save_images, outname="bottom")
    rois.append(mask_values(res, lookup))
    images.append(res)

    return rois, images




def check_uniformity(roi, border_rois, max_difference=2.):
    """This function check for uniformity"""
    passed = False
    pass_array = []
    central_mean = np.mean(roi)

    means = [np.mean(edge_roi) for edge_roi in border_rois]
    for edge in means:
        if abs(edge - central_mean) < max_difference:
            pass_array.append(True)
            print(
                f"[INFO] Edge Uniformity check:\t {colored('[SUCCESS]', 'green')} with difference {abs(edge - central_mean)}")
            st.markdown(f"[INFO] Edge Uniformity check:\t <font color=‘green’>[SUCCESS]</font> with difference {abs(edge - central_mean)}",unsafe_allow_html=True)

        else:
            pass_array.append(False)
            print(
                f"[WARNING] Edge Uniformity check:\t {colored('[FAILED]', 'red')} not passed with difference {abs(edge - central_mean)}")
            st.markdown(f"[WARNING] Edge Uniformity check:\t <font color='red'>[FAILED]</font> with difference {abs(edge - central_mean)}",unsafe_allow_html=True)

    passed = np.prod(np.array(pass_array))

    if passed:
        print(f"[INFO] Global Uniformity check:\t {colored('[SUCCESS]', 'green')}")
        st.markdown(f"[INFO] Global Uniformity check:\t <font color=‘green’>[SUCCESS]</font>",unsafe_allow_html=True)
    else:
        print(f"[WARNING] Global Uniformity check:\t {colored('[FAILED]', 'red')} not passed")
        st.markdown(f"[WARNING] Global Uniformity check:\t <font color='red'>[FAILED]</font>",unsafe_allow_html=True)


    return passed,means



