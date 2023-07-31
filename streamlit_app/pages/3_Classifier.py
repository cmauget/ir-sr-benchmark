import streamlit as st #type: ignore
from utils.utils import Model_utils, Image 
import tensorflow as tf #type: ignore
import cv2 #type: ignore

def main():

    st.title("Classifier")

    st.write("Uploadez l'image")
    
    model = Model_utils.create_model()
    model.load_weights("models/crack_classifier/cp.ckpt")

    

    uploaded_file = st.sidebar.file_uploader("Importer une image", type=["png", "jpg", "jpeg","tif"])
    
    if uploaded_file is not None:

        image = Image.load_streamlit(uploaded_file)
        
        if st.button("Détecter"):
            img_path = "IMG/VN_vis_New.png"
            image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            image = Model_utils.predict_and_draw_boxes(image, model)
            #tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
            st.image(image, use_column_width=True, caption="Image prédite avec les fissures détectées")
            Image.save("IMG/output.png", image)

    

if __name__ == "__main__":
    main()

    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)