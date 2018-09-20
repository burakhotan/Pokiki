import Helper
import cv2
import numpy as np
from pathlib import Path

rootFolder = Path('./')
picPath = rootFolder / 'in/anen.jpeg'
tilesFolder = rootFolder / "tiles/"

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

result = None
columns = None
splitByHorizontal = 102
splitByVertical = 76
columnCount = 0
for count, img in enumerate(Helper.splitImg(picPath, splitByHorizontal, splitByVertical)):
    open_cv_IMG = np.array(img, dtype='uint8')[:, :, ::-1]
    open_cv_IMG = increase_brightness(open_cv_IMG)

    tile_color = Helper.getDominantColor(open_cv_IMG) 
    tile_pic_path = tilesFolder / Helper.findNearestNeighbor(tile_color)
    
    windowW, windowH = img.size
    tile_pic = cv2.imread( str(tile_pic_path), cv2.IMREAD_COLOR)  
    tile_pic = cv2.resize(tile_pic, (windowW*5, windowH*5))
    # tile_pic = open_cv_IMG
    if columns is None:
        columns = tile_pic
        columnCount += 1
    elif columnCount < (splitByHorizontal - 1):
        columns = np.hstack((columns, tile_pic))
        columnCount += 1
    else:
        if result is None:
            result = columns
        else:
            result = np.vstack((result, columns))
        columns = None
        columnCount = 0

cv2.imwrite('hmm.png', result)