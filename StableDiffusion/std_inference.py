# Importing libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Downloading pretrained model
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, revision='fp16', dtype=torch.float16)

# Send the model to GPU
device = torch.device("cuda:0")
pipeline_runway = pipeline.to(device)

'''Prompts to send to Stable Diffusion model. 
  Tip: A more sophisticated prompt leads to the generation of higher quality images.'''
  
prompts = [
  "A cat is outside",
  "A cat standing on the floor",
  "A running dog",
  "Dog is at home",
  "Hen is eating",
  "Flying kite",
  "Peacock is walking",
  "Folding chair", 
  "Computer desk",
  "Dining table",
  "A man is riding a horse on the moon, photorealistic",
  "A man with a hat is riding a horse on the surface of the moon, photorealistic"
  ]

images = pipeline_runway(prompts, num_images_per_prompt=1, output_type="numpy").images

# Display the NumPy array as an image using Matplotlib
for i in range(images.shape[0]):
    plt.imshow(images[i])
    plt.axis('off')  # Turn off axes
    plt.show()