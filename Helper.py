import numpy as np
import cv2
from scipy.stats import itemfreq
import json
import math
from PIL import Image
from pathlib import Path

class HelperOBJ:
    def __init__(self, dataPath):
        self.data = json.loads(open(dataPath).read())

    def findNearestNeighbor(self, color, dominant=False):
        r, g, b = color[0], color[1], color[2]
        closest = None

        for index, imgName in enumerate(self.data):
            # dom_color = self.data[imgName]['dominant_color']
            avg_color = self.data[imgName]['average_color']
            if dominant:
                r_candit, g_candit, b_candit = dom_color[0], dom_color[1], dom_color[2]
                closeness = math.sqrt((r-r_candit)**2 + (g-g_candit)**2 + (b-b_candit)**2)

                if closest is None or closest[0] >= closeness:
                    closest = [closeness, imgName]
            else:
                r_candit, g_candit, b_candit = avg_color[0], avg_color[1], avg_color[2]
                closeness = math.sqrt((r-r_candit)**2 + (g-g_candit)**2 + (b-b_candit)**2)

                if closest is None or closest[0] >= closeness:
                    closest = [closeness, imgName]

        return closest[1]

def getAverageColor(img):
    return [img[:, :, i].mean() for i in range(img.shape[-1])]

def getDominantColor(img, n_colors=1):
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)
    return palette[np.argmax(itemfreq(labels)[:, -1])].tolist()

def loadFileJSON(file):
    dataFileStr = open(file).read()

    return json.loads(dataFileStr)

def splitRow(input_picture, splitByHorizontal, splitByVertical):
    width, boxH = input_picture.size
    boxW = width/splitByHorizontal
    width_steps = np.arange(0, width, boxW)
    for j in width_steps:
        box = (j, 0, j + boxW, boxH)
        yield input_picture.crop(box)

def splitImg(input_picture, splitByHorizontal, splitByVertical):
    width, height = input_picture.size
    boxH, boxW = height/splitByVertical, width/splitByHorizontal
    height_steps = np.arange(0, height, boxH)
    width_steps = np.arange(0, width, boxW)
    # height_steps = [boxH * x for x in range(0, splitByVertical)]
    # width_steps = [boxW * x for x in range(0, splitByHorizontal)]
    for i in height_steps: # range(0, height, int(boxH)):
        for j in width_steps: # range(0, width, int(boxW)):
            box = (j, i, j + boxW, i + boxH)
            yield input_picture.crop(box)
