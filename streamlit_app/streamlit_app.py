import streamlit as st #type: ignore

import sys
import os

folder = os.path.dirname(__file__)
sys.path.append(folder+"/..")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Ce programme vous donne trois outils :   
        
    Un Upscaler qui utilise des models de super résolution pour améliorer la précision des images thermiques.  
        
    Un Fuse qui permet de fusionner les images thermqiues et visible.    
        
    Un Classifier qui permet de détecter la presénce de fissure sur l'image.
    """
)

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)