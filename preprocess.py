import numpy as np
from PIL import Image

def load_data(path, crop=True, size=None, mode="label", xp=np):
    img = Image.open(path)
    if crop:
        w,h = img.size
        if w < h:
            if w < size:
                img = img.resize((size, size*h//w))
                w, h = img.size
        else:
            if h < size:
                img = img.resize((size*w//h, size))
                w, h = img.size
        img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))

    if mode=="label":
        y = xp.asarray(img, dtype=xp.int32)
        mask = (y == 255)
        mask = mask.astype(xp.int32)
        y[mask] = -1
        return y

    elif mode=="data":
        x = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        #x -= 120
        return x

    elif mode=="predict":
        return img
