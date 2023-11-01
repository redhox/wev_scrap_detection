import cv2
import pickle
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import streamlit as st
import tensorflow as tf
from PIL import Image



model = load_model('./best_model.h5')
#cap = cv2.VideoCapture(0) #capture video de la webcam
# Ajuster la résolution de la webcam
#cap.set(3, 640)  # Largeur
#cap.set(4, 480)  # Hauteur
def predict(image):
    # Prétraitement de l'image
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    # Faire la prédiction
    predictions = model.predict(image)
    label = tf.argmax(predictions, axis=1)[0]
    confidence = tf.reduce_max(predictions, axis=1)[0]
    return label, confidence , predictions

def main():
    # Titre et description de l'application
    st.title("Prédiction d'image")
    st.write("Téléchargez une image")
    # Téléchargement de l'image
    uploaded_file = st.file_uploader("Sélectionnez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Affichage de l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)

        # Bouton pour faire la prédiction
        if st.button("Faire la prédiction"):
            label, confidence , predictions= predict(image)
            if confidence < 0.5:
                st.write("porte un casque")

            else:
                st.write("ne porte pas un casque")
                
if __name__ == '__main__':
    main()
