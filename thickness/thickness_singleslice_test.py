import math

from scipy.optimize import curve_fit

from util.legacy_writer import legacyWriter
from thickness.thickness_functions import *
import streamlit as st
import cv2
import matplotlib as plt
from termcolor import colored
import matplotlib.pyplot as plt



def get_data(dcm_slice):
    """Function to get data and rescale to standard TC values"""
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    st.write(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing


def band_filter(src,min_v=150,max_v=400):
    """Helper function for thickness single slice test"""
    out=np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i,j]>min_v and src[i,j]<max_v:
                out[i,j]=src[i,j]
    return out



def check_thickness_lines(keptRect, center, pix_dim, targetvalue=2.5, distance=0.625):
    passed = False

    thickness = []
    values=[]
    opt=[]
    nord, sud, est, ovest = keptRect

    draw=center.copy()

    # default values
    bb, hh, dx, dy = 40, 4, -10, -1
    print(f"[INFO] computing thickness of sud insert..")
    st.write(f"[INFO] computing thickness of sud insert..")

    # x=int(np.min(sud,axis=0)[0])-dx
    # y=int(np.min(sud,axis=0)[1])-dy

    # sudRect=(x, y, b,h)
    x, y, b, h = sud[0]
    sud_crop = center[y+dy:y + hh, x+dx:x + bb]

    coor=[(x+dx,y+dy),(x+bb,y+hh)]
    cv2.rectangle(draw,coor[0],coor[1],color=(255,255,255),thickness=1)



    vals = np.mean(sud_crop, axis=0)
    computed=computeThickness(sud_crop,pix_dim)
    thickness.append(computed[0])
    values.append(computed[1])
    opt.append(computed[2])

    print(f"[INFO] computing thickness of nord insert..")
    st.write(f"[INFO] computing thickness of nord insert..")
    # x=int(np.min(nord,axis=0)[0])-dx
    # y=int(np.min(nord,axis=0)[1])-dy

    # nordRect=(x, y, b,h)
    x, y, b, h = nord[0]
    coor=[(x+dx,y+dy),(x+bb,y+hh)]
    cv2.rectangle(draw,coor[0],coor[1],color=(255,255,255),thickness=1)

    nord_crop = center[y+dy:y + hh, x+dx:x + bb]
    vals = np.mean(nord_crop, axis=0)
    computed=computeThickness(nord_crop,pix_dim)
    thickness.append(computed[0])
    values.append(computed[1])
    opt.append(computed[2])

    print(f"[INFO] computing thickness of est insert..")
    st.write(f"[INFO] computing thickness of est insert..")
    # x=int(np.min(est,axis=0)[0])-dx
    # y=int(np.min(est,axis=0)[1])-dy

    # estRect=(x, y, b,h)
    x, y, b, h = est[0]

    coor=[(x+dy,y+dx),(x+hh,y+bb)]
    cv2.rectangle(draw,coor[0],coor[1],color=(255,255,255),thickness=1)
    est_crop = np.transpose(center[y+dx:y + bb, x+dy:x + hh])
    plt.imshow(est_crop)
    vals = np.mean(est_crop, axis=0)

    computed=computeThickness(est_crop,pix_dim)
    thickness.append(computed[0])
    values.append(computed[1])
    opt.append(computed[2])

    print(f"[INFO] computing thickness of ovest insert..")
    st.write(f"[INFO] computing thickness of ovest insert..")
    # dx=-2
    # x=int(np.min(ovest,axis=0)[0])-dx
    # y=int(np.min(ovest,axis=0)[1])-dy

    x, y, b, h = ovest[0]

    coor=[(x+dy,y+dx),(x+hh,y+bb)]
    cv2.rectangle(draw,coor[0],coor[1],color=(255,255,255),thickness=1)
    # ovestRect=(x, y, b,h)
    # hh=[np.max(ovest,axis=0)[0]-x,h]
    # h=int(np.mean(hh))

    ovest_crop = np.transpose(center[y+dx:y + bb, x+dy:x + hh])
    vals = np.mean(ovest_crop, axis=0)
    computed=computeThickness(ovest_crop,pix_dim)

    thickness.append(computed[0])
    values.append(computed[1])
    opt.append(computed[2])

    fig,ax=plt.subplots(1)
    ax.imshow(draw,cmap="gray")
    st.write(fig)


    thick_mean = np.mean(thickness)
    print(f"[INFO] means found: {thickness}")
    st.write(f"[INFO] means found: {thickness}")


    #GRAPH PART
    fig, ax = plt.subplots(2,2)

    st.write(f"[DEBUG] means found: {values[0]}")

    #nord
    x=np.linspace(0,len(values[0]),len(values[0]))
    A, mu, sigma, offset = opt[0]

    th=math.tan(23/180.*math.pi)*2.35*sigma*pix_dim[0]
    ax[0,0].plot(x,values[0],'b')
    ax[0,0].plot(x,gauss(x,A,mu,sigma,offset),'r')
    ax[0,0].set_xlabel("column")
    ax[0,0].set_ylabel("Pixel Value")
    ax[0,0].set_title("Nord ROI")
    ax[0,0].text(mu, 200, f'Fit output:\nmu={round(mu, 2)}\nsigma={round(sigma, 2)}\nthick:{round(th, 3)}', style='italic', fontsize=6,
            bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 5})


    #sud
    x = np.linspace(0, len(values[1]), len(values[1]))
    A, mu, sigma, offset = opt[1]

    th = math.tan(23 / 180. * math.pi) * 2.35 * sigma * pix_dim[0]
    ax[0, 1].plot(x, values[1], 'b')
    ax[0, 1].plot(x, gauss(x, A, mu, sigma, offset),'r')
    ax[0, 1].set_xlabel("column")
    ax[0, 1].set_ylabel("Pixel Value")
    ax[0, 1].set_title("Sud ROI")
    ax[0, 1].text(mu,200,
                  f'Fit output:\nmu={round(mu, 2)}\nsigma={round(sigma, 2)}\nthick:{round(th, 3)}', style='italic',
                  fontsize=6,
                  bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 5})


    #est
    x = np.linspace(0, len(values[2]), len(values[2]))
    A, mu, sigma, offset = opt[2]

    th = math.tan(23 / 180. * math.pi) * 2.35 * sigma * pix_dim[0]
    ax[1, 0].plot(x, values[2], 'b')
    ax[1, 0].plot(x, gauss(x, A, mu, sigma, offset),'r')
    ax[1, 0].set_xlabel("column")
    ax[1, 0].set_ylabel("Pixel Value")
    ax[1, 0].set_title("Est ROI")
    ax[1, 0].text(mu, 200,
                  f'Fit output:\nmu={round(mu, 2)}\nsigma={round(sigma, 2)}\nthick:{round(th, 3)}', style='italic',
                  fontsize=6,
                  bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 5})


    #ovest
    x = np.linspace(0, len(values[3]), len(values[3]))
    A, mu, sigma, offset = opt[3]

    th = math.tan(23 / 180. * math.pi) * 2.35 * sigma * pix_dim[0]
    ax[1, 1].plot(x, values[3], 'b')
    ax[1, 1].plot(x, gauss(x, A, mu, sigma, offset),'r')
    ax[1, 1].set_xlabel("column")
    ax[1, 1].set_ylabel("Pixel Value")
    ax[1, 1].set_title("Ovest ROI")
    ax[1, 1].text(mu, 200,
                  f'Fit output:\nmu={round(mu, 2)}\nsigma={round(sigma, 2)}\nthick:{round(th, 3)}', style='italic',
                  fontsize=6,
                  bbox={'facecolor': 'aquamarine', 'alpha': 0.5, 'pad': 5})

    fig.subplots_adjust(hspace=0.5,wspace=0.4)
    st.write(fig)



    if thick_mean > targetvalue - distance and thick_mean < targetvalue + distance:
        passed = True
        print(
            f"[INFO] Testing slice thickness with linear rois: {colored('[SUCCESS]', 'green')} with value: {thick_mean}")
        st.markdown(f"[INFO] Testing slice thickness with linear rois: <font color='green'>[SUCCESS]</font> with value: {thick_mean}",unsafe_allow_html=True)
    else:
        print(
            f"[WARNING] Testing slice thickness with linear rois: {colored('[FAILED]', 'red')} with value: {thick_mean}")
        st.markdown(f"[WARNING] Testing slice thickness with linear rois: <font color='red'>[FAILED]</font> with value: {thick_mean}",unsafe_allow_html=True)


    return passed, thick_mean


def computeThickness(roi,pix_dim):

    #HERE KEEP PROFILE
    vals=np.mean(roi,axis=0)
    p0=[100,20,3,100]
    popt, pcov = curve_fit(gauss, np.linspace(0,len(vals),len(vals)), vals,p0=p0, maxfev=1000)
    A, mu, sigma, offset = popt
    thick=math.tan(23/180.*math.pi)*2.35*sigma*pix_dim[0]   #questa potrebbe essere la cosa giusta
    return thick,vals,popt

def test_thickness_singleslice(path, legacy, sheet, function):

    print(f"[INFO] THICKNESS TEST - SINGLE SLICE MODE")
    st.write(f"[INFO] THICKNESS TEST - SINGLE SLICE MODE")
    #thk_img=pydicom.dcmread(path)
    img,pix_dim=get_data(path)
    center = crop_img(img, 170, 150, 180, 200)
    dst=np.zeros(img.shape)
    norma=cv2.normalize(center,dst,0,255,cv2.NORM_MINMAX)
    norma = norma.astype("uint8")
    out = band_filter(norma).astype("uint8")


    conts, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(conts)
    boundRect = [None]*len(conts)
    for i, c in enumerate(conts):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])


    nord = []
    sud = []
    est = []
    ovest = []
    area_min = 18
    for r in boundRect:
        if r[2] * r[3] > area_min:  # if greater than area min compute

            if (r[0] < 130 and r[0] > 60):
                if (r[1] < 100):
                    sud.append(r)
                else:
                    nord.append(r)
            elif (r[1] < 130 and r[1] > 60):
                if (r[0] < 100):
                    est.append(r)
                else:
                    ovest.append(r)
    keptRect = [nord, sud, est, ovest]

    #compute the test
    passed,thick_mean=check_thickness_lines(keptRect, center, pix_dim)

    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet} WARNING: This must be the semester dose file..." )
        st.write(f"[INFO] legacy mode: work on {legacy} in {sheet} WARNING: This must be the semester dose file..." )
        leg = legacyWriter(legacy, sheet)
        leg.write_singleslice_thickness_report(thick_mean)
