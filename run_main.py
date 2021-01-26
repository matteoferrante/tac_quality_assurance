import streamlit.cli
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
import matplotlib.pyplot as plt

from util import SessionState
from util.refresh_state import *
from util.refresh_state import _get_state

if __name__ == '__main__':
    streamlit.cli._main_run_clExplicit('tac_quality_assurance.py', 'streamlit run')