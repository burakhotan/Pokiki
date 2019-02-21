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

if os.path.isfile(dataFilePath):
    print("File already exists, overwrite existing file? y/n")
    ans = input()
    if ans == 'y':
        for imgName in os.listdir(tilesFolder):
            img_path = tilesFolder / imgName
            if os.path.isfile(img_path):
                # Loads an image from a file passed as argument.
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print("Can't open", img_path)
                    quit()
                print("Processing Image:", imgName)
                data[imgName] = {}
                data[imgName]['average_color'] = Helper.getAverageColor(img)
                # data[imgName]['dominant_color'] = Helper.getDominantColor(img)
            else:
                print("Not a file:", img_path)
        with open(dataFilePath, 'w') as outfile:
            print("Saving to:", dataFilePath)
            json.dump(data, outfile)
            print("Done")
    else:
        print("Quitting.")