import Helper
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
import time
from functools import partial

rootFolder = Path('./')
picPath = rootFolder / 'in/ecem.jpg'
tilesFolder = rootFolder / "tiles/"
input_picture = Image.open(picPath)

helperOBJ = Helper.HelperOBJ(Path('./out/data.json'))

def buildRows(splitByHorizontal, splitByVertical, picture_section):
    columns = None
    columnCount = 0
    for count, img in enumerate(Helper.splitRow(picture_section, splitByHorizontal, splitByVertical)):
        open_cv_IMG = np.array(img, dtype='uint8')
        open_cv_IMG = cv2.cvtColor(open_cv_IMG, cv2.COLOR_RGB2BGR)
        open_cv_IMG = increase_brightness(open_cv_IMG, value=20)

        tile_color = Helper.getDominantColor(open_cv_IMG) 
        tile_pic_path = tilesFolder / helperOBJ.findNearestNeighbor(tile_color)
        
        windowW, windowH = img.size
        tile_pic = cv2.imread( str(tile_pic_path), cv2.IMREAD_COLOR)  
        tile_pic = cv2.resize(tile_pic, (windowW*8, windowH*8))
        # tile_pic = open_cv_IMG
        if columns is None:
            columns = tile_pic
            columnCount += 1
        elif columnCount < (splitByHorizontal - 1):
            columns = np.hstack((columns, tile_pic))
            columnCount += 1
        else:
            return columns

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

if __name__=='__main__':
    start = time.time()
    time.clock()    

    splitByHorizontal = 256
    splitByVertical = 144
    pictureW, pictureH = input_picture.size

    rowH = pictureH / splitByVertical
    rowW = pictureW / splitByHorizontal
    rows = []
    for section_index in range(0, splitByVertical):
        section = (0, rowH * section_index, pictureW, rowH * (section_index + 1))
        row = input_picture.crop(section)
        rows.append(row)

    with Pool(processes=4) as pool:
        func = partial(buildRows, splitByHorizontal, splitByVertical)
        result_rows = pool.map(func, rows)

    resultIMG = None
    for row in result_rows:
        if resultIMG is None:
            resultIMG = row
        else:
            resultIMG = np.vstack((resultIMG, row))

    elapsed = time.time() - start
    print('Starmap elapsed time:', elapsed)
    cv2.imwrite('hmm.png', resultIMG)