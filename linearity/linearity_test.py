import pandas as pd
import numpy as np
import cv2
import math
import os
from util.boundingBox import BoundingBoxWidget
from util.legacy_writer import legacyWriter
from termcolor import colored
import streamlit as st
import matplotlib.pyplot as plt


base={'nord': (256, 376, 10), 'sud': (256, 136, 10), 'est': (376, 256, 10), 'ovest': (136, 256, 10), 'sudovest': (198, 152, 10), 'sudest': (314, 152, 10), 'nordest': (314, 360, 10), 'nordovest': (198, 360, 10)}

def get_data(dcm_slice):
    """Function to get data and rescale to standard TC values"""
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing


def find_native_circles(img):

    """This function takes the img as input and return a list of circles found with hough gradient"""
    # this function returns a list of circle (cx,cy,r) and a grayscale img

    gray = img.copy()
    print(f"[INFO] using HOUGH_GRADIENT algorithm to find circles...")
    st.write(f"[INFO] using HOUGH_GRADIENT algorithm to find circles...")
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # normalize img
    dst = np.zeros(gray.shape)
    cv2.normalize(gray, dst, 0, 255, cv2.NORM_MINMAX)

    dst = dst.astype("uint8")
    display = dst.copy()
    rows = gray.shape[0]
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 3, 50,
                               param1=45, param2=25,
                               minRadius=4, maxRadius=15)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            # cv2.circle(dst, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(dst, center, radius, (255, 0, 255), 3)

    return circles, display



def check_simmetry(x_expected,y_expected,cir,tollerance=3): #check if the circle was already found or find it by simmetry
    """This function take as input a point and a list of circle and check if the desidered one is already found or find it by simmetry"""

    for i in cir:
        if abs(x_expected-i[0])<tollerance and abs(y_expected-i[1])<tollerance:
            print(f"[INFO] circle already found! return original x and y")
            st.write(f"[INFO] circle already found! return original x and y")
            return i[0],i[1]

        else:
            print(f"[INFO] circle was found by simmetry")
            st.write(f"[INFO] circle was found by simmetry")
            return x_expected,y_expected


def find_circles_simmetry(circles,img,pix_dim, roi_diameter=10, tollerance=3):
    cir = circles[0]
    cir = np.array(cir)

    print(f"[INFO] seek for NORD roi and use it to orientate the other ROIs")
    st.write(f"[INFO] seek for NORD roi and use it to orientate the other ROIs")
    # found the highest
    x, y, r = cir[np.argmax(cir[:, 1])]
    mean_r = int((roi_diameter // 2) / pix_dim[0])  # take the final radius

    # tollerance=3
    points = {}
    if abs(img.shape[0] // 2 - x) < tollerance:
        print(f"[INFO] {colored('NORD found!', 'green')} Check SUD existence by Hough algorithm or find it by simmetry...")
        st.markdown(f"[INFO] <font color=‘green’>NORD FOUND</font>, Check SUD existence by Hough algorithm or find it by simmetry...",unsafe_allow_html=True)
        points["nord"] = (x, y, mean_r)
        # find the lowest
        x_expected = x
        y_expected = img.shape[1] // 2 - (y - img.shape[1] // 2)
        x_low, y_low = check_simmetry(x_expected, y_expected, cir)
        points["sud"] = (x_low, y_low, mean_r)
    else:
        print(f"[INFO] {colored('NORD not found', 'red')} manual roi placement needed..")
        st.markdown(f"[INFO] <font color=‘red’>NORD not found</font> manual roi placement needed..",unsafe_allow_html=True)
    # find rightest

    print(f"\n[INFO] looking for EST roi and use it to orientate the other ROIs")
    st.write(f"\n[INFO] looking for EST roi and use it to orientate the other ROIs")
    x, y, r = cir[np.argmax(cir[:, 0])]

    # check the leftest or find it by simmetry
    if abs(img.shape[0] // 2 - y) < tollerance:
        print(
            f"[INFO] {colored('EST found!', 'green')} Check OVEST existence by Hough algorithm or find it by simmetry...")
        st.markdown(f"[INFO] <font color=‘green’>EST FOUND</font> Check OVEST existence by Hough algorithm or find it by simmetry...",unsafe_allow_html=True)
        points["est"] = (x, y, mean_r)
        # find the lowest
        x_expected = img.shape[0] // 2 - (x - img.shape[0] // 2)
        y_expected = y
        x_left, y_left = check_simmetry(x_expected, y_expected, cir)
        points["ovest"] = (x_left, y_left, mean_r)
    else:
        print(f"[INFO] {colored('EST not found', 'red')} manual roi placement needed..")
        st.markdown(f"[INFO] <font color=‘red’>EST not found</font> manual roi placement needed..",unsafe_allow_html=True)
    # get the remaining circles
    print(f"\n\n[INFO] selecting remaining points")
    st.write(f"\n\n[INFO] selecting remaining points")
    borders = []
    for i in cir:
        new = True
        for j in points.values():
            if (np.linalg.norm(i - j) < 10):
                new = False
        if new:
            borders.append(i)

    for border in borders:

        (x, y, z) = border
        if border[0] >= img.shape[0] // 2 and border[1] >= img.shape[1] // 2:
            x, y, r = border
            points["nordest"] = (x, y, mean_r)

        elif border[0] >= img.shape[0] // 2 and border[1] < img.shape[1] // 2:

            x, y, r = border
            points["sudest"] = (x, y, mean_r)

        elif border[0] < img.shape[0] // 2 and border[1] >= img.shape[1] // 2:

            x, y, r = border
            points["nordovest"] = (x, y, mean_r)

        elif border[0] < img.shape[0] // 2 and border[1] < img.shape[1] // 2:

            x, y, r = border
            points["sudovest"] = (x, y, mean_r)

        # work on sudest or sudovest if present
    if "sudest" in points.keys():
        points = simmetry_from_borders(points["sudest"],img,pix_dim, "sudest", points, circles)
    elif "sudovest" in points.key():
        points = simmetry_from_borders(points["sudovest"],img,pix_dim, "sudovest", points, circles)
    else:
        print(f"[{colored('WARNING', 'red')}] wasn't possible to find obliques inserts. Manual ROI placement required!")
        st.markdown(f"[<font color=‘orange’>WARNING</font>] wasn't possible to find obliques inserts. Manual ROI placement required!",unsafe_allow_html=True)
    print(f"[INFO] points found: \t{len(points)}")
    st.write(f"[INFO] points found: \t{len(points)}")

    return points


def simmetry_from_borders(border,img,pix_dim, side, points, circles, roi_diameter=10):
    # simmetry function to find out the insterts
    cir = circles[0]
    cir = np.array(cir)
    mean_r = int((roi_diameter // 2) / pix_dim[0])

    if side == "sudest":
        x, y, r = border
        print("[INFO] seek for sudovest")
        st.write("[INFO] seek for sudovest")
        expected_x = img.shape[0] // 2 - (x - img.shape[0] // 2)
        expected_y = y

        s_x, s_y = check_simmetry(expected_x, expected_y, cir)
        points["sudovest"] = (s_x, s_y, mean_r)

        print("[INFO] seek for nordest")
        st.write("[INFO] seek for nordest")
        expected_x = x
        expected_y = img.shape[1] // 2 + (img.shape[1] // 2 - y)
        s_x, s_y = check_simmetry(expected_x, expected_y, cir)
        points["nordest"] = (s_x, s_y, mean_r)

        print("[INFO] seek for nordovest")
        st.write("[INFO] seek for nordovest")
        expected_x = img.shape[0] // 2 - (x - img.shape[0] // 2)
        expected_y = img.shape[1] // 2 + (img.shape[1] // 2 - y)
        s_x, s_y = check_simmetry(expected_x, expected_y, cir)
        points["nordovest"] = (s_x, s_y, mean_r)

    if side == "sudovest":
        x, y, r = border
        print("[INFO] seek for sudest")
        st.write("[INFO] seek for sudest")
        expected_x = img.shape[0] // 2 + (x - img.shape[0] // 2)
        expected_y = y
        s_x, s_y = check_simmetry(expected_x, expected_y, cir)
        points["sudest"] = (s_x, s_y, mean_r)
        simmetry_from_borders(points["sudest"], img,pix_dim,"sudest")
        # recursive tu sudest

    return points


def check_roi(display, mean_r, points):

    """Check if found rois are good"""
    bw = BoundingBoxWidget(display, mean_r, points)
    checkCircle = True
    while checkCircle:

        ##TEXT INFO

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (20, 30)
        org2 = (20, 50)
        org3 = (20, 70)
        # fontScale
        fontScale = 0.7

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(bw.show_image(), 'Left click - add ROI', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        # Using cv2.putText() method
        image = cv2.putText(bw.show_image(), 'Right click - restart', org2, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        # Using cv2.putText() method
        image = cv2.putText(bw.show_image(), 'Press q - Accept', org3, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        ### END TEXT INFO
        cv2.imshow('image', bw.show_image())
        key = cv2.waitKey(1)
        roi = bw.get_roi()
        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            checkCircle = False

    return roi, bw


def modify_roi(points,new_rois,bw):

    """Eventually modify the found rois"""
    for new in new_rois:
        distances={}
        for (k,v) in points.items():
            distances[k]=np.linalg.norm((np.array(new)-np.array(v)))
        dist=(sorted(distances.items(), key=lambda item: item[1]))
        print(f"[INFO] modifying {dist[0][0]} from {v} to {new}")
        st.write(f"[INFO] modifying {dist[0][0]} from {v} to {new}")
        points[dist[0][0]]=new
        bw.clean_roi()
    return points, len(new_rois)


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



def check_linearity(rois,densities,min_corr):
    passed=False
    corr=np.corrcoef(rois,densities)
    r=corr[0,1]

    ## GRAPH INFO
    fig, ax = plt.subplots()
    ax.scatter(densities,rois)
    m, b = np.polyfit(densities, rois, 1)
    ax.plot(densities, m * densities + b,'r')
    ax.set_title("Linearity check")
    ax.set_xlabel("Density (g/cm^3)")
    ax.set_ylabel("HU units")

    ax.text(0.2, 500, f'Fit output:\nm={round(m,2)}\nb={round(b,2)}\nr^2:{round(r,5)}', style='italic', fontsize=10,
            bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 10})

    st.write(fig)

    ### END GRAPH INFO
    if r>=min_corr:
        print(f"[INFO] Linearity test: {colored('[SUCCESS]','green')} with correlation value: {r}")
        st.markdown(f"[INFO] Linearity test: <font color=‘green’>[SUCCESS]</font> with correlation value: {r}",unsafe_allow_html=True)
        passed=True
    else:
        print(f"[WARNING] Linearity test: {colored('[FAILED]','red')} with correlation value: {r}")
        st.markdown(f"[WARNING] Linearity test: <font color=‘red’>FAILED</font> with correlation value: {r}",unsafe_allow_html=True)
    return passed,r


def get_hu(lin_roi):
    print(f"[INFO] computing mean value for each insert...")
    st.write(f"[INFO] computing mean value for each insert...")
    hu=[np.mean(lin) for lin in lin_roi]
    return sorted(hu)

def density_2_ct(density):
    return 1000*(density-1.)/1.

def test_linearity(lin_img,legacy=None,sheet=None,option="fbp"):

    """The idea is to preprocess the image to find some cirlce with hough gradients, find the others by simmetry and test the density vs ct number"""

    print(f"[INFO] LINEARITY TEST")
    st.markdown(f"[INFO] LINEARITY TEST")
    #lin_img = pydicom.dcmread(path)  # load image for linear spacing
    img, pix_dim = get_data(lin_img)  # rescale img
    roi_diameter = 10
    circles, display = find_native_circles(img)  # start finding circles
    points=find_circles_simmetry(circles,img,pix_dim)     #complete circles search



    if len(points)!=8:
        print(f"{colored('[WARNING]','red')}[WARNING] Could not find all the circles. Using default list instead")
        st.markdown(f"[<font color='orange'>WARNING</font>] Could not find all the circles. Using default list instead",unsafe_allow_html=True)
        points=base
    #eventually manual changes are possible

    mean_r=int((roi_diameter//2)/pix_dim[0])
    checking=False
    l=0
    while checking:
        new,bw=check_roi(display,mean_r,points)
        if len(new)>l:
            points,l=modify_roi(points,new,bw)
        else:
            checking=False

    print(f"[INFO] ROis accepted, going to linearity test..")
    st.write(f"[INFO] ROis accepted, going to linearity test..")
    materials = []
    materials.append({"name": "Air", "density": 0})
    materials.append({"name": "PMP", "density": 0.83})
    materials.append({"name": "LDPE", "density": 0.92})
    materials.append({"name": "Polystyrene", "density": 1.05})
    materials.append({"name": "Acrilyc", "density": 1.18})
    materials.append({"name": "Delrin", "density": 1.41})
    materials.append({"name": "Teflon", "density": 2.16})
    dens_df = pd.DataFrame(materials)
    dens_df['ct'] = dens_df['density'].apply(density_2_ct)

    # generate the rois and measure the mean number in each one
    linearity_rois = []
    for p in points.values():
        res, lookup = select_roi(img, p[0], p[1], 10, pix_dim)
        linearity_rois.append(mask_values(res, lookup))

    lin = get_hu(linearity_rois)[1:]

    check_linearity(lin, dens_df["density"].values, 0.99)
    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        st.write(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        leg=legacyWriter(legacy,sheet)
        leg.write_linearity_report(lin,option)
