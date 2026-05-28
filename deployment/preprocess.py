import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image, image_size):
    image = image.convert("RGB")
    image = ImageOps.pad(
        image,
        (image_size, image_size),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image
