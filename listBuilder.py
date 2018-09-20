import cv2
import os
import json
import Helper
from pathlib import Path

# Get the input filename
rootFolder = Path('./')
tilesFolder = rootFolder / "tiles/"
dataFilePath = rootFolder / 'out/data.json'
data = {}

if not os.path.isfile(dataFilePath):

    for imgName in os.listdir(folder):
        # Loads a grayscale image from a file passed as argument.
        img = cv2.imread(folder + imgName, cv2.IMREAD_COLOR)
        print("Processing Image:", imgName)
        data[imgName] = {}
        data[imgName]['average_color'] = Helper.getAverageColor(img)
        data[imgName]['dominant_color'] = Helper.getDominantColor(img)

    with open(dataFilePath, 'w') as outfile:
        print("Saving to:", dataFilePath)
        json.dump(data, outfile)
