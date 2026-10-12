import gradio as gr
from PIL import Image
import requests
import io
import numpy as np

def recognize_digit(image):
    # the sketchpad returns a dict with a 'composite' key corresponding to the image
    image = image['composite']
    # By default the image is a 4 channels image, we need to convert it to a 1 channel image since the API expects a 1 channel image
    image = image[:, :, 0]
    # invert the image
    image = 255 - image
    # convert numpy to uint8
    image = image.astype(np.uint8)
    # Convert the image to a PIL Image
    image = Image.fromarray(image)
    # Convert the image to a binary file
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    # Send request to the API
    response = requests.post("http://api:5075/predict", data=img_binary.getvalue())
    return response.json()["prediction"]

if __name__=='__main__':

    interface = gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=False,
                description="Draw a number on the sketchpad to see the model's prediction.",
                )
    print("Starting Gradio app...")
    interface.launch(server_name="0.0.0.0", server_port=7860) # the server will be accessible externally under this address   
