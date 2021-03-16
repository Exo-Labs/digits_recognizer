from model_relu_deploy import transform_image, get_prediction
from PIL import Image, ImageOps
import numpy as np
import io

img = Image.open(r"C:\Users\User\Desktop\111.png")
img = ImageOps.invert(img)

tensor = transform_image(img)
prediction = get_prediction(tensor)
print(prediction)