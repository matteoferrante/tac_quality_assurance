import random

import streamlit as st
import pydicom

import openpyxl

#test
from linearity.linearity_test import *
from uniformity.uniformity_multislice_test import *
from uniformity.uniformity_singleslice_test import *
from resolution.resolution_test import *
from thickness.thickness_multislice_test import *
from thickness.thickness_singleslice_test import *
from lowcontrast.lowcontrast_test import *
from cart.cart_test import *
from iodine.iodine_test import *

import matplotlib.pyplot as plt

from util import SessionState
from util.refresh_state import *
from util.refresh_state import _get_state

report=[]


#test_dict={"Linearity":["fbp","ar"],"Low Contrast":["227FOV"],"Resolution":["std","bone"],"Uniformity":["head","body","multislice","multislice_monoenergetic"],"Thickness":["single_slice","multi_slice"],"Cart":["Cart displacement"],"Iodine":["Iodine"]}

catphan_dict={"Linearity":["fbp","asir"],"Resolution":["std","bone"],"Thickness":["single_slice","multi_slice"],"Cart":["Cart displacement"]}
ge_dict={"Low Contrast":["227FOV"],"Resolution":["std","bone"],"Uniformity":["head","body","multislice","multislice_monoenergetic"],"Iodine":["Iodine"]}

st.header("TAC QUALITY ASSURANCE")

page_bg_img = '''
<style>
body {
background-image: url("https://edu.ieee.org/pa-upanama/wp-content/uploads/sites/374/2015/02/minimalistic-simple-background-white-206534-1920x12002.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

file_loader = st.empty()

state = _get_state()


if not state.widget_key:
    state.widget_key = str(random.randint(1000, 100000000))

file = file_loader.file_uploader("Upload DCM files", accept_multiple_files=True,key=state.widget_key)

if st.button('Clear uploaded files'):
    state.widget_key = str(random.randint(1000, 100000000))

state.sync()

if state.report is None:
    state["report"]=[]
###LAVORARE QUA

#if not state.widget_key:
#    state.widget_key = str(random.randint(1000, 100000000))
#file = file_loader.file_uploader("Upload a DCM image", accept_multiple_files=True,key=state.widget_key)

#if st.button('clear uploaded_file'):
#    state.widget_key = str(random.randint(1000, 100000000))
#    st.write("Refreshing uploaded files")

excel = st.sidebar.text_input('Enter the excel file path:')
sheet=None

if len(excel)>0:
    if excel[0]=='"':
        excel=excel[1:-1]  #remove quotes if present
    try:
        wb=openpyxl.load_workbook(excel)
        sheet=st.sidebar.selectbox(
            f"Which sheet you want to compile?",
            wb.sheetnames
        )
    except:
        st.sidebar.write("Please insert a valid excel path to report the results")



def get_data(dcm_slice):
    print(f"[INFO] Linearly rescaling image using Slope and Intercept written in the DICOM header")
    img=dcm_slice.pixel_array
    m=dcm_slice.RescaleSlope
    q=dcm_slice.RescaleIntercept
    return m*img+q,dcm_slice.PixelSpacing


machine = st.sidebar.selectbox(
    "Which phantom are you going to test?",
    ("GE", "Catphan")
)


if machine=="GE":
    test_dict=ge_dict
elif machine=="Catphan":
    test_dict = catphan_dict

test = st.sidebar.selectbox(
    "Which test are you going to do?",
   list(test_dict.keys())
)

option=st.sidebar.selectbox(
    f"Which kind of {test} are you asking for?",
    test_dict[test]
)

run_btn=st.sidebar.button("RUN","run_btn")

#switch per controllare i filtri possibili per i vari test

if test=="Thickness" and option=="multi_slice":
    function = st.sidebar.selectbox(
        f"Which function do you want to use for fitting? [Gaussian recommended]",
        ("gaussian","hyper_gaussian")
    )


im_list=[]

if test=="Cart":
    col1, col2 = st.beta_columns(2)
    default_displacement_forward = col1.text_input("Selected displacement forward", "",key="1")
    measured_displacement_forward= col2.text_input("Measured displacement forward", "",key="2")

    default_displacement_backward = col1.text_input("Selected displacement backward", "",key="3")
    measured_displacement_backward= col2.text_input("Measured displacement backward", "",key="4")


if file is not None:

    if len(file)>0:
        im_list = [pydicom.dcmread(i) for i in file]
        if len(im_list)>1:
            slider=st.slider("Image:",min_value=1,max_value=len(im_list))

            # test_img=pydicom.dcmread(file[0])
            images=[get_data(x)[0] for x in im_list]

            fig, ax = plt.subplots()
            ax.imshow(images[slider-1], cmap="gray")
            st.write(fig)

        else:
            #test_img=pydicom.dcmread(file[0])
            img,px_dim=get_data(im_list[0])
           # dst=np.zeros(img.shape)
           # cv_img=cv2.normalize(img,dst,0,255,cv2.NORM_MINMAX)
           # st.image(cv_img,clamp=True)
            fig, ax = plt.subplots()
            ax.imshow(img,cmap="gray")
            st.write(fig)

    st.write("Machine:\t", machine)
    st.write("Test:\t", test)
    st.write("Option:\t",option)

    #ss = SessionState.get(report=[])
    report=state.report
    ##SWITCH PER AVVIO
    if run_btn:

        #ss.report.append(test+" - "+option)
        state.report.append(test + " - " + option)
        if len(excel)<1:
            excel=None

        if test=="Linearity":
            test_linearity(im_list[0],excel,sheet,option)

        elif test=="Uniformity":
            #disambiguation for multislice or single slice
            if option=="multislice":
                unifomity_multislicetest(im_list,excel,sheet,option)   #pass a list of dcmimagesù
            elif option=="multislice_monoenergetic":
                test_uniformity(im_list[int(len(im_list)/2)], excel, sheet, option)
                unifomity_multislicetest(im_list, excel, sheet,option)
            else:
                test_uniformity(im_list[0],excel,sheet,option)

        elif test=="Resolution":
            if machine=="Catphan":
                #test_contrast_resolution(im_list[0],excel,sheet,option)
                test_contrast_resolution_catphan(im_list[0],excel,sheet,option)

            elif machine=="GE":

                test_contrast_resolution(im_list[0],excel,sheet,option)

        elif test=="Thickness":
            #disamiguation
            if option == "multi_slice":
                test_thickness_multislice(im_list, excel, sheet, function)
            elif option == "single_slice":
                test_thickness_singleslice(im_list[0], excel, sheet, "gaussian")

        elif test=="Low Contrast":
            test_lowcontrast(im_list,excel,sheet)
        elif test=="Cart":
            data=[default_displacement_forward,measured_displacement_forward,default_displacement_backward,measured_displacement_backward]
            data=[int(i) for i in data]
            test_cart(data,excel,sheet)

        elif test=="Iodine":
            test_iodine(im_list,excel,sheet)


st.sidebar.header("Conducted Tests:")
if state is not None:
    if state.report is not None:
        for r in state.report:
            st.sidebar.text(r)
