# ==============================================================================
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import numpy as np
# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

#colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


VOC_LABELS_LIST = {
    0:'none',
    1:'Bicycle',
    2:'Electromobile',
    3:'SchoolBus',
    4:'Bus',
    5:'DoubleDeck',
    6:'Motorhome',
    7:'FireTruck',
    8:'RacingCar', # need to remove in the future
    9:'SUV',
    10:'Truck',
    11:'Motorcycle',
    12:'Microbus',
    13:'Minivan',
    14:'Sedan',
    15:'PoliceCar',
    16:'Tricycle',
    17:'Placeholder_one',
    18:'Placeholder_two',
    19:'Placeholder_three' ,
    20:'Placeholder_four'
}
# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes_1(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
    #                           size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
   # label = '{} {:.2f}'.format(classes, scores)

   # label_size = draw.textsize(label, font)
    height = img.size[1]
    width = img.size[0]
    print(height, width)
    colors = {0:(255, 255, 255), 1:(31, 119, 180), 2:(174, 199, 232), 3:(255, 127, 14), 4:(255, 187, 120),
                  5: (44, 160, 44), 6:(152, 223, 138), 7:(214, 39, 40), 8:(255, 152, 150),
                  9:(148, 103, 189), 10:(197, 176, 213), 11:(140, 86, 75), 12:(196, 156, 148),
                  13:(227, 119, 194), 14:(247, 182, 210), 15:(127, 127, 127), 16:(199, 199, 199),
                  17:(188, 189, 34), 18:(219, 219, 141), 19:(23, 190, 207), 20:(158, 218, 229)}

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
    
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        class_name = VOC_LABELS_LIST[cls_id]
        score = scores[i]
        label = '{} {:.2f}'.format(class_name, score)
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)
        thickness = (height + width) // 300


        if cls_id >= 0:

            if cls_id not in colors:
                colors[cls_id] = (random.randint(0,256),random.randint(0,256), random.randint(0,256))

            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)

            # print(ymin, xmin, ymax, xmax, colors[cls_id])

            if ymin - label_size[1] >= 0:
                text_origin = np.array([xmin, ymin - label_size[1]])
            else:
                text_origin = np.array([xmin, ymin + 1])

            # draw.rectangle([xmin,ymin,xmax,ymax], outline=colors[cls_id],width=int(thickness))
            for i in range(thickness):
                draw.rectangle([xmin +i, ymin+i, xmax-i, ymax-i], outline=colors[cls_id])

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[cls_id])

            draw.text(text_origin,text=label,fill=(0,0,0),font=font)

            #del draw

    return img