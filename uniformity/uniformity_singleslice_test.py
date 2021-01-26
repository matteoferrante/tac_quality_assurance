from uniformity.uniformity_functions import *
import streamlit as st
import pandas as pd
from util.legacy_writer import legacyWriter

PHANTOM_DIAMETER=215
save_images=False



def test_uniformity(dcm_img,legacy=None,sheet=None,tipology="body"):

    #set limits
    if tipology=="body":
        max_difference=10
    elif tipology=="head":
        max_difference = 3


    print(f"[INFO] UNIFORMITY, NOISE AND CT NUMBER CHECK - {tipology.upper()} MODE")
    st.write(f"[INFO] UNIFORMITY, NOISE AND CT NUMBER CHECK - {tipology.upper()} MODE")

    """This function perform all the processing and the test needed for water ct, uniformity and noise test"""
    if save_images:
        os.makedirs("cq_images", exist_ok=True)

    #dcm_slice = pydicom.dcmread(path)
    img, pix_dim = get_data(dcm_img)
    #select the center roi
    res,lookup=select_roi(img,img.shape[0]//2,img.shape[0]//2,20,pix_dim,save=save_images,outname="center")
    roi=mask_values(res,lookup)
    border_rois, images = make_border_rois(img,pix_dim, 30, 20)
    print(f"[INFO] Starting default test for quality control for image {tipology}...")
    st.write(f"[INFO] Starting default test for quality control for image {tipology}...")
    results=[]
    # 2.Noise
    noise,std=check_noise(roi)
    # 3. Water CT
    water,mean=check_waterct(roi)
    # 4. Uniformity
    uniformity,means=check_uniformity(roi, border_rois,max_difference=max_difference)

    border_std=[check_noise(br)[1] for br in border_rois]

    res_dic = {'noise': std, 'water_ct': mean, 'uniformity_left': means[0], 'std_left':border_std[0], 'uniformity_up': means[1],'std_up':border_std[1], 'uniformity_right': means[2], 'std_right':border_std[2],
             'uniformity_bottom': means[3], 'std_bottom':border_std[3], 'noise_test': noise, 'water_ct_test': water, 'uniformity_test': uniformity}
    results.append(res_dic)
    print("\n\n")

    df=pd.DataFrame(results)
    print(f"[INFO] printing out results for noise, water and uniformity test: \n {df}")
    st.write(f"[INFO] printing out results for noise, water and uniformity test:")
    st.write(df)

    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        leg=legacyWriter(legacy,sheet)
        if tipology=="body":
            leg.write_uniformity_body_report(df)
        elif tipology=="head":
            leg.write_uniformity_head_report(df)
        elif tipology=="multislice_monoergetic":
            leg.write_uniformity_monoenergetic_report(df)