import numpy as np
import streamlit as st




def gauss(x, *p):
    """Gaussian function"""
    A, mu, sigma, offset = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset


def hyper_gauss(x, *p):
    """Gaussian function"""
    A, mu, sigma, offset = p
    return A * np.exp(-((x - mu) ** 2)**3 / (2. * sigma ** 2))**3 + offset


def crop_img(img,x,y,w,h):
    """This function crop the image from x to x+w and from y to y+h"""
    print(f"[INFO] cropping image with (w,h): {(w,h)} from (x,y): {(x,y)}")
    st.write(f"[INFO] cropping image with (w,h): {(w,h)} from (x,y): {(x,y)}")
    return img[x:x+w,y:y+h]



def normalize(images, force_min=None):
    """This function globally normalize a list of images"""

    print(f"[INFO] normalizing images ..")
    maxs = []
    mins = []
    for i in images:
        maxs.append(np.max(i))
        mins.append(np.min(i))
    global_max = np.max(maxs)
    global_min = np.min(mins)
    if force_min is not None:
        global_min = force_min

    print(f"[INFO] global min: {global_min}, global_max: {global_max}")
    st.write(f"[INFO] global min: {global_min}, global_max: {global_max}")

    normalized = []
    for i in images:
        norm = np.zeros(i.shape)
        for x in range(i.shape[0]):
            for y in range(i.shape[1]):
                norm[x, y] = 255 * (i[x, y] - global_min) / (global_max - global_min)
        normalized.append(norm)
    return normalized
