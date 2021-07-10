import gradio as gr
import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.models import load_model
class_names=['Creta', 'Ford EcoSport', 'Hyundai Santro', 'Innova', 'MG Hector', 'Mahindra Bolero',
 'Maruti Baleno', 'Maruti Eeco', 'Maruti Vitara Brezza', 'Nano', 'Range Rover', 'Renault Duster',
 'Scorpio', 'hyundai i20', 'hyundai venue', 'mahindra thar', 'maruti swift', 'mini cooper']
model_load=load_model('best_model.h5')
def prediction(image):
    image = image.reshape((1, 224, 224, 3))
    image=tf.keras.applications.mobilenet.preprocess_input(image)
    prediction = model_load.predict(image).flatten()
    return {class_names[i]: float(prediction[i]) for i in range(18)}
# Use predefined input and output objects from gradio
image1 = gr.inputs.Image(shape=(224,224))
label1 = gr.outputs.Label(num_top_classes=18)

# Gradio interface to input an image and see its prediction with percentage confidence
gr.Interface(fn=prediction, inputs=image1, outputs=label1,
             #theme="huggingface",
             title="CLASSIFICATION OF CAR MODELS USING CONVOLUTIONAL NEURAL NETWORK",
             description =" Select an image and hit submit to see its classification",
             allow_flagging=False,
             layout="vertical",
             live=True,
             capture_session=True,
             interpretation='default').launch(debug='True',share=True)


