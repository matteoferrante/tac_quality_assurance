from uniformity.uniformity_functions import *
import streamlit as st
import pandas as pd
from util.legacy_writer import legacyWriter

PHANTOM_DIAMETER=215
save_images=False



def unifomity_multislicetest(im_list,legacy=None,sheet=None,option="multislice"):


    print(f"[INFO] UNIFORMITY, NOISE AND CT NUMBER CHECK - MULTISLICE MODE")
    st.write(f"[INFO] UNIFORMITY, NOISE AND CT NUMBER CHECK - MULTISLICE MODE")

    """Handle mutlislice test"""
    if save_images:
        os.makedirs("cq_images", exist_ok=True)

    #im_list=glob.glob(os.path.join(path,"*.DCM"))



    results=[]
    for (i,image) in enumerate(im_list):
        #dcm_slice = pydicom.dcmread(image)
        img, pix_dim = get_data(image)
        #select the center roi
        res,lookup=select_roi(img,img.shape[0]//2,img.shape[0]//2,20,pix_dim,save=save_images,outname="center")
        roi=mask_values(res,lookup)
        border_rois, images = make_border_rois(img,pix_dim, 30, 20)
        print(f"[INFO] Starting default test for quality control for image {i+1}/{len(im_list)}...")
        st.write(f"[INFO] Starting default test for quality control for image {i+1}/{len(im_list)}...")
        #LAVORARE QUA AGGIUNGERE UNA TABELLA O QUALCOSA PER TENERE CONTO DEI VALORI

        # 2.Noise
        noise,std=check_noise(roi)
        # 3. Water CT
        water,mean=check_waterct(roi)
        # 4. Uniformity
        uniformity,means=check_uniformity(roi, border_rois)
        border_std = [check_noise(br)[1] for br in border_rois]

        res_dic = {'noise': std, 'water_ct': mean, 'uniformity_left': means[0], 'std_left': border_std[0],
                   'uniformity_up': means[1], 'std_up': border_std[1], 'uniformity_right': means[2],
                   'std_right': border_std[2],
                   'uniformity_bottom': means[3], 'std_bottom': border_std[3], 'noise_test': noise,
                   'water_ct_test': water, 'uniformity_test': uniformity}
        results.append(res_dic)
        print("\n\n")


    df=pd.DataFrame(results)
    print(f"[INFO] printing out results for noise, water and uniformity test: \n {df}")
    st.write(f"[INFO] printing out results for noise, water and uniformity test: ")

    st.write(df)

    if legacy is not None:

        #eventually add here option for monoenergetic
        if option=="multislice":

            print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
            st.write(f"[INFO] legacy mode: work on {legacy} in {sheet}")
            leg=legacyWriter(legacy,sheet)
            leg.write_uniformity_multislice_report(df,option)
        elif option=="multislice_monoenergetic":

            print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
            st.write(f"[INFO] legacy mode: work on {legacy} in {sheet}")
            leg = legacyWriter(legacy, sheet)
            leg.write_uniformity_multislice_report(df,option)