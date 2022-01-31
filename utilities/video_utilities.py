# import the necessary packages
from PIL import ImageFont, ImageDraw, Image
from . import config
from matplotlib import pyplot as plt
from urllib.request import urlopen
import cv2
import datetime
import math
import numpy as np


def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def crop_square_center(image, cropX, cropY):
    h, w = image.shape[:2]
    startx = w//2-(cropX//2)
    starty = h//2-(cropY//2)
    return image[starty:starty+cropY, startx:startx+cropX]


def trim_and_save(image, imageName, bbox, jpegQuality=80):
    # timestamp (up to millisecond)
    timenow = datetime.datetime.now().strftime('%H%M%S.%f')[:-2]
    fileName = f'{imageName}-{timenow}.jpg'

    # trim ROI
    outputImage = trim(image, bbox) if bbox else image
    
    # save as JPEG quality
    outputPath = f'{config.UPLOAD_FOLDER}/{fileName}'
    cv2.imwrite(outputPath, outputImage, [int(cv2.IMWRITE_JPEG_QUALITY), jpegQuality])
    return outputImage, outputPath
    

def trim(image, bbox):
    if bbox:
        # crop image using bbox and margin
        (startX, startY, endX, endY) = safe_bbox(image, bbox)
        return image[startY:endY, startX:endX]
    else:
        return image


def get_portrait_bbox(face_bbox, face_to_portrait_ratio, portarit_aspect):
    (startX, startY, endX, endY) = face_bbox
    faceH = endY-startY
    faceW = endX-startX
    centroidX = int(startX+faceW/2)
    centroidY = int(startY+faceH/2)*.99
    
    portraitW = np.sqrt(faceH*faceW / (portarit_aspect*face_to_portrait_ratio))
    portraitH = portraitW * portarit_aspect
    portraitStartX = int(centroidX - portraitW/2)
    portraitStartY = int(centroidY - portraitH/2)
    portraitEndX = int(centroidX + portraitW/2)
    portraitEndY = int(centroidY + portraitH/2)

    return (portraitStartX, portraitStartY, portraitEndX, portraitEndY)


def expand_bbox(image, bbox, expand_ratio=0.1):
    h, w = image.shape[:2]
    offsetX = w * expand_ratio
    offsetY = h * expand_ratio
    (startX, startY, endX, endY) = bbox
    startX = startX - offsetX
    startY = startY - offsetY
    endX = endX + offsetX
    endY = endY + offsetY
    return safe_bbox(image, (startX, startY, endX, endY))


def safe_bbox(image, bbox):
    # set bbox coords to range from 0 ~ max height/width of the image
    h, w = image.shape[:2]
    startX = int(min(max(bbox[0], 0), w))
    startY = int(min(max(bbox[1], 0), h))
    endX = int(min(max(bbox[2], 0), w))
    endY = int(min(max(bbox[3], 0), h))
    return (startX, startY, endX, endY)


def draw_bbox_with_label(image, label, bbox, color=(0,200,0)):
    (startX, startY, endX, endY) = safe_bbox(image, bbox)
    
    # draw rectangle using bbox
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 3) 

    # text label inside top-left of bbox
    if label and len(label) > 1:
        labelHeight = 12
        
        # draw multi-line text label
        for enum, textline in enumerate(label.split('\n')):
            labelWidth = len(textline)*7

            # determine whether to show text label above or inside the bbox
            if startY < labelHeight:
                labelPositionY = startY+labelHeight*enum
                textPositionY = startY+labelHeight*(enum+1)-2
            else:
                labelPositionY = startY-labelHeight*(enum+1)
                textPositionY = startY-labelHeight*enum-2

            cv2.rectangle(image, (startX-2, labelPositionY), (startX+labelWidth, labelPositionY+labelHeight), color, -1)
            cv2.putText(image, textline, (startX, textPositionY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    return image


def translate_bbox(image, prediction_bbox):
        if len(prediction_bbox) == 4:
            h, w = image.shape[:2]
            startX = min(max(int(w * prediction_bbox[0]), 0), w)
            startY = min(max(int(h * prediction_bbox[1]), 0), h)
            endX = min(max(int(w * prediction_bbox[2]), 0), w)
            endY = min(max(int(h * prediction_bbox[3]), 0), h)
            return (startX, startY, endX, endY)
        else:
            return (None, None, None, None)


def draw_transparent_bbox(image, label, bbox, color=(0, 250, 0), alpha=0.5):
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = image.copy()
    
    (startX, startY, endX, endY) = safe_bbox(image, bbox)

    # draw a transparent rectangle using bbox
    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
    
    # apply the overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # text label inside top-left of bbox
    # if label and len(label) > 1:
    #     textLabelWidth = len(label)*9
    #     textLabelHeight = 16
        
    #     # determine whether to show text label above or inside the bbox
    #     if startY < textLabelHeight+5:
    #         labelPositionY = startY+textLabelHeight
    #         textPositionY = startY+textLabelHeight-1
    #     else:
    #         labelPositionY = startY-textLabelHeight
    #         textPositionY = startY-2
        
    #     # draw text label
    #     cv2.putText(image, label, (startX, textPositionY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return image

# Method to download image from URL and read it into CV2 format
def download_image(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, -1)
    return image


# Method to blur the supplied image
def blur_image(image, intensity=5):
    return cv2.blur(image, (intensity, intensity))


# Method to overlay a transparent image ontop of another image 
def overlay_transparent(background, overlay, x=0, y=0):
    bg_height, bg_width = background.shape[:2]

    if x >= bg_width or y >= bg_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > bg_width:
        print(f'Oh, x+w ({x+w}) exceeded bg_width ({bg_width})')
        w = bg_width - x
        overlay = overlay[:, :w]

    if y + h > bg_height:
        print(f'Oh, y+h ({y+h}) exceeded bg_height ({bg_height})')
        h = bg_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [overlay,
            np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image
    return background


# Display multiple images in Matplotlib subplots
def show_images(images, titles=None, height=16, width=16, axis='on'):
    
    # define plot matrix layout
    n = len(images)
    if n == 0:
        return None
    elif n >= 4:
        nCols = 4
        nRows = math.ceil(n/4)
    else:
        nCols = n
        nRows = math.ceil(n/3)
        
    # create figure object
    fig = plt.figure(figsize=(height, width))
    
    # for each image
    for i, image in enumerate(images):
        # convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # add subplots to figure
        fig.add_subplot(nRows, nCols, i+1)
        plt.axis(axis)
        
        # add subtitles
        if titles is not None:
            plt.gca().set_title(titles[i]) 

        # add images to two subplots
        plt.imshow(image_rgb)
    
    # show entire plot
    plt.show()
    return None


# Display single image in a popup window
def display_image(image, window_name='output', resizeRatio=1):
    (h, w) = image.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Get Centroids
def get_centroid(bbox):
    (startX, startY, endX, endY) = bbox
    centroidX = startX + int((endX-startX)//2)
    centroidY = startY + int((endY-startY)//2)
    return (centroidX, centroidY)