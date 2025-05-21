from base64 import b64decode
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

classes = ['அ', 'ஆ', 'ஓ', 'ஙூ', 'சூ', 'ஞூ', 'டூ', 'ணூ', 'தூ', 'நூ', 'பூ', 'மூ', 'யூ', 'ஃ', 'ரூ', 'லூ', 'வூ', 'ழூ', 'ளூ', 'றூ', 'னூ', 'ா', 'ெ', 'ே', 'க', 'ை', 'ஸ்ரீ', 'ஸு', 'ஷு', 'ஜு', 'ஹு', 'க்ஷு', 'ஸூ', 'ஷூ', 'ஜூ', 'ங', 'ஹூ', 'க்ஷூ', 'க்', 'ங்', 'ச்', 'ஞ்', 'ட்', 'ண்', 'த்', 'ந்', 'ச', 'ப்', 'ம்', 'ய்', 'ர்', 'ல்', 'வ்', 'ழ்', 'ள்', 'ற்', 'ன்', 'ஞ', 'ஸ்', 'ஷ்', 'ஜ்', 'ஹ்', 'க்ஷ்', 'ஔ', 'ட', 'ண', 'த', 'ந', 'இ', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன', 'ஈ', 'ஸ', 'ஷ', 'ஜ', 'ஹ', 'க்ஷ', 'கி', 'ஙி', 'சி', 'ஞி', 'டி', 'உ', 'ணி', 'தி', 'நி', 'பி', 'மி', 'யி', 'ரி', 'லி', 'வி', 'ழி', 'ஊ', 'ளி', 'றி', 'னி', 'ஸி', 'ஷி', 'ஜி', 'ஹி', 'க்ஷி', 'கீ', 'ஙீ', 'எ', 'சீ', 'ஞீ', 'டீ', 'ணீ', 'தீ', 'நீ', 'பீ', 'மீ', 'யீ', 'ரீ', 'ஏ', 'லீ', 'வீ', 'ழீ', 'ளீ', 'றீ', 'னீ', 'ஸீ', 'ஷீ', 'ஜீ', 'ஹீ', 'ஐ', 'க்ஷீ', 'கு', 'ஙு', 'சு', 'ஞு', 'டு', 'ணு', 'து', 'நு', 'பு', 'ஒ', 'மு', 'யு', 'ரு', 'லு', 'வு', 'ழு', 'ளு', 'று', 'னு', 'கூ']

def url_to_img(dataURL):
    string = str(dataURL)
    comma = string.find(",")
    code = string[comma + 1:]
    decoded = b64decode(code)
    buf = BytesIO(decoded)
    img = Image.open(buf)

    converted = img.convert("LA")
    la = np.array(converted)
    la[la[..., -1] == 0] = [255, 255]
    whiteBG = Image.fromarray(la)

    img_np = np.array(whiteBG.convert('L'))
    inverted = cv2.bitwise_not(img_np)
    kernel = np.ones((2, 2), np.uint8)
    # eroded = cv2.erode(inverted, kernel, iterations=2)
    dilated = cv2.dilate(inverted, kernel, iterations=2)
    opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (7, 7), 0)
    # _, smoothed = cv2.threshold(blurred, 80, 200, cv2.THRESH_BINARY)

    resized = cv2.resize(blurred, (64, 64))
    return Image.fromarray(resized)


def transformImg(img):
    my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    return my_transforms(img).unsqueeze(0)

def get_prediction(url, net):
    img = url_to_img(url)
    transformed = transformImg(img)
    output = net(transformed)
    output = nn.Softmax()(output)

    prob, predicted = torch.max(output.data, 1)
    confidence = int(round(prob.item() * 100))
    print(classes[predicted] + " " + str(confidence))
    return classes[predicted] + " " + str(confidence)