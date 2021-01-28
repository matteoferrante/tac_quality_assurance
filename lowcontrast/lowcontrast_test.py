import streamlit as st
import numpy as np
import math
from util.legacy_writer import legacyWriter
from termcolor import colored
from scipy.stats import norm
import matplotlib.pyplot as plt


def get_data(dcm_slice):
    """Function to get data and rescale to standard TC values"""
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    st.write(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing

def crop_img(img,x,y,w,h):
    """This function crop the image from x to x+w and from y to y+h"""
    print(f"[INFO] cropping image with (w,h): {(w,h)} from (x,y): {(x,y)}")
    st.write(f"[INFO] cropping image with (w,h): {(w,h)} from (x,y): {(x,y)}")
    return img[x:x+w,y:y+h]

def compute_means(cropped,dimensions,l=216):

    """Compute mean for each tiny box in low contrast test"""
    print(f"[INFO] computing contrast check for images...")
    st.write(f"[INFO] computing contrast check for images...")
    dims=[]
    for dim in dimensions:
        print(f"[INFO] working on dim {dim}")
        st.write(f"[INFO] working on dim {dim}")
        along=[]
        for cr in cropped:
            #compute the dim
            submatrix=split_matrix(cr,dim)
            #submatrix=split(cr,dim,dim)
            along.append(np.array([np.mean(sub) for sub in submatrix]))

        box_per_slice=int((l/dim)**2)
        #along=np.reshape(np.array(along),(len(cropped)*box_per_slice,dim,dim))
        print(f"[INFO] dimensions: {dim} computed {box_per_slice} boxes per slice")
        st.write(f"[DEBUG] dimensions: {dim} computed {box_per_slice} boxes per slice")
        #means=np.array([(np.mean(i),np.std(i)) for i in along])      #LAVORARE QUA -> MEDIA DI OGNI BOX NELLE FETTE
        dims.append(along)
    return dims

def split_matrix(img,side):
    """Split a square image into multiple submatrices"""
    array=[]
    n_x=int(math.modf(img.shape[0]/side)[1])
    n_y=int(math.modf(img.shape[1]/side)[1])
    for i in range(0,img.shape[0],side):
        for j in range(0,img.shape[1],side):
            vals=np.array(img[i:i+side,j:j+side])
            array.append(vals)
    return array

def check_contrast(means, lcd_expected=[5.096, 3.647, 2.278, 1.451, 1.105, 0.812], distance=0.1,
                   dimensions_mm=[3, 4, 6, 9, 12, 18]):
    passed_list = []
    mu_final = []
    std_final = []
    lcd_final = []

    fig,ax=plt.subplots(2,3)

    for (j,m) in enumerate(means):
        mu_list = []
        std_list = []
        lcd = []

        for sl in m:  # m is a list of slices
            (mu, sigma) = norm.fit(sl)
            # the histogram of the data
            #n, bins, patches = plt.hist(sl, 30, normed=1, facecolor='cyan', alpha=0.75)
            mu_list.append(mu)
            std_list.append(sigma)
            lcd.append(3.29 * sigma)
        mu_final.append(np.mean(mu_list))
        std_final.append(np.mean(std_list))
        lcd_final.append(np.mean(lcd))

        # GRAPH PART
        if j < 3:
            mm = np.array(m)
            x = np.linspace(np.min(mm), np.max(mm), np.max(mm.shape))
            mi = mu_final[j]
            si = std_final[j]
            li = lcd_final[j]
            y = norm.pdf(x, mi, si)
            ax[0, j].hist(m, density=1)
            ax[0, j].plot(x, y, color='black')
            txt = f'Fit output:\nmu={round(mi, 2)}\nsigma={round(si, 2)}\nLCD:{round(li, 3)}'
            ax[0, j].text(-3.5 * si, max(y) * 0.75, txt,
                          style='italic', fontsize=6,
                          bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 5})
            ax[0, j].set_title(f"{dimensions_mm[j]} mm")

        if j > 2:
            mm = np.array(m)
            x = np.linspace(np.min(mm), np.max(mm), np.max(mm.shape))
            mi = mu_final[j]
            si = std_final[j]
            li = lcd_final[j]
            y = norm.pdf(x, mi, si)
            ax[1, j - 3].hist(m, density=1)
            ax[1, j - 3].plot(x, y, color='black')
            txt = f'Fit output:\nmu={round(mi, 2)}\nsigma={round(si, 2)}\nLCD:{round(li, 3)}'
            ax[1, j - 3].text(-3.5 * si, max(y) * 0.75, txt,
                              style='italic', fontsize=6,
                              bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 5})
            ax[1, j - 3].set_title(f"{dimensions_mm[j]} mm")

    for (i, lcd_value) in enumerate(lcd_final):
        if lcd_value > lcd_expected[i] * (1 - distance) and lcd_value < lcd_expected[i] * (1 + distance):
            passed_list.append(True)
            print(
                f"[INFO] test for {dimensions_mm[i]} mm contrast: {colored('[SUCCESS]', 'green')} with value: {lcd_value}")

            st.markdown(f"[INFO] test for {dimensions_mm[i]} mm contrast: <font color='green'>[SUCCESS]</font> with value: {lcd_value}",unsafe_allow_html=True)
        else:
            passed_list.append(False)
            print(
                f"[WARNING] test for {dimensions_mm[i]} mm contrast: {colored('[FAILED]', 'red')} with value: {lcd_value}")

            st.markdown(f"[WARNING] test for {dimensions_mm[i]} mm contrast: <font color='red'>[FAILED]</font> with value: {lcd_value}",unsafe_allow_html=True)
    passed = bool(np.prod(passed_list))
    fig.subplots_adjust(hspace=0.5,wspace=0.4)

    return passed, mu_final, std_final, lcd_final,fig





def test_lowcontrast(dcm_images, legacy=None, sheet=None):
    print(f"[INFO] LOW CONTRAST TEST..")
    st.write(f"[INFO] LOW CONTRAST TEST..")
    #load dicom image
    #images_list=glob.glob(os.path.join(path,"*.DCM"))

    #dcm_images=[pydicom.dcmread(image) for image in images_list]
    images=[get_data(image)[0] for image in dcm_images]
    pix_dim=dcm_images[0].PixelSpacing

    dimensions_mm = [3, 4, 6, 9, 12, 18]
    computed_dim = [int(round(0.886227 * dim_mm * 512 / 227, 0)) for dim_mm in dimensions_mm]
    start_x,start_y,w,h=(148,148,216,216)

    cropped=[crop_img(image,start_x,start_y,w,h) for image in images]
    means = compute_means(cropped, computed_dim)
    passed, mu_list, std_list, lcd,fig = check_contrast(means)

    st.write(fig)
    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet} ..." )
        st.write(f"[INFO] legacy mode: work on {legacy} in {sheet} ...")
        leg = legacyWriter(legacy, sheet)
        leg.write_lowcontrast_report(mu_list,std_list,lcd)