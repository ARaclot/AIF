import gradio as gr
import argparse
import torch
import torchvision.transforms as transforms
from model import MNISTNet

from PIL import Image
import requests
import io
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default = '/home/raclot/AIF/AIF/weights/mnist_net.pth', help='path to model')
args = parser.parse_args()

model_path = args.model_path

model = MNISTNet().to(device)
# Load the model
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

    img_pil = image

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)
    
    return (int(predicted[0]))

if __name__=='__main__':

    interface = gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=False,
                description="Draw a number on the sketchpad to see the model's prediction.",
                )
    print("Starting Gradio app...")
    interface.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True) 