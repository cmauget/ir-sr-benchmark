import streamlit as st #type: ignore
from utils.utils import Model_utils, Image 

def main():

    st.title("Classifier")

    st.write("Uploadez l'image")
    
    model = Model_utils.create_model()
    model.load_weights("models/crack_classifier/cp.ckpt")

    uploaded_file = st.sidebar.file_uploader("Importer une image", type=["png", "jpg", "jpeg","tif"])
    
    if uploaded_file is not None:

        image_path = "temp_image.jpg"
        image = Image.load_streamlit(uploaded_file)
        
        if st.button("Détecter"):
            image = Model_utils.predict_and_draw_boxes(image, model)
            st.image(image, use_column_width=True, caption="Image prédite avec les fissures détectées")
    

if __name__ == "__main__":
    main()

    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)