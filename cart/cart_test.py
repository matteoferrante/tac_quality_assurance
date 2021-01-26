import streamlit as st
from termcolor import colored

from util.legacy_writer import legacyWriter


def test_cart(data,legacy,sheet):
    print(f"[INFO] Cart displacement test")
    st.write(f"[INFO] Cart displacement test")
    if abs(data[0]-data[1])<6:
        print(f"[INFO] Forward cart displacement test: {colored('[SUCCESS]','green')}")
        st.markdown(f"[INFO] Forward cart displacement test:  <font color='green'>[SUCCESS]</font>",unsafe_allow_html=True)

    else:
        print(f"[WARNING] Forward cart displacement test: {colored('[FAILED]', 'red')}")

        st.markdown(f"[WARNING] Forward cart displacement test:  <font color='red'>[FAILED]</font>",unsafe_allow_html=True)

    if abs(data[2] - data[3]) < 6:
        print(f"[INFO] Backward cart displacement test: {colored('[SUCCESS]', 'green')}")

        st.markdown(f"[INFO] Backward cart displacement test:  <font color='green'>[SUCCESS]</font>",unsafe_allow_html=True)
    else:
        print(f"[WARNING] Backward cart displacement test: {colored('[FAILED]', 'red')}")

        st.markdown(f"[WARNING] Forward cart displacement test:  <font color='red'>[FAILED]</font>",unsafe_allow_html=True)


    if legacy is not None:
        print(f"[INFO] legacy mode: work on {legacy} in {sheet}")
        leg=legacyWriter(legacy,sheet)
        leg.write_cart_report(data)