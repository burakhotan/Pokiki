#!/usr/bin/python3

import Helper
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
import time
from functools import partial
import sys, getopt

rootFolder = Path('./')
tilesFolder = rootFolder / "tiles/"

helperOBJ = Helper.HelperOBJ(Path('./out/data.json'))

def buildRows(splitByHorizontal, splitByVertical, quality, picture_section):
    columns = None
    columnCount = 0
    for count, img in enumerate(Helper.splitRow(picture_section, splitByHorizontal, splitByVertical)):
        open_cv_IMG = np.array(img, dtype='uint8')
        open_cv_IMG = cv2.cvtColor(open_cv_IMG, cv2.COLOR_RGBA2BGRA)
        # open_cv_IMG = increase_brightness(open_cv_IMG, value=20)

        # tile_color = Helper.getDominantColor(open_cv_IMG) 
        tile_color = Helper.getAverageColor(open_cv_IMG) 
        tile_pic_path = tilesFolder / helperOBJ.findNearestNeighbor(tile_color)
        
        windowW, windowH = img.size
        tile_pic = cv2.imread( str(tile_pic_path), cv2.IMREAD_COLOR)  
        tile_pic = cv2.resize(tile_pic, (windowW*quality, windowH*quality))
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

def main(argv):
    inputfile = ''
    outputfile = ''
    
    quality = 4
    splitByHorizontal = 100
    splitByVertical = 100
    try:
        opts, args = getopt.getopt(argv,"hx:y:i:o:q:",["help", "horizontal=","vertical=", "ifile=","ofile=", "quality="])
    except getopt.GetoptError:
        print('Unexpected Error, example usage:')
        print ('test.py -i <inputfile> -o <outputfile> -x <horizontalDivide> -y <verticalDivide> -q <quality>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print ('test.py -i <inputfile> -o <outputfile> -x <horizontalDivide> -y <verticalDivide> -q <quality>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-x", "--horizontal"):
            splitByHorizontal = int(arg)
        elif opt in ("-y", "--vertical"):
            splitByVertical = int(arg)
        elif opt in ("-q", "--quality"):
            quality = int(arg)
        else:
            print('Command not understood, example usage:')
            print ('test.py -i <inputfile> -o <outputfile> -x <horizontalDivide> -y <verticalDivide> -q <quality>')
            sys.exit()
    
    if inputfile == '' or outputfile == '':
        print('Please specify input and output files, example usage:')
        print ('test.py -i <inputfile> -o <outputfile> -x <horizontalDivide> -y <verticalDivide> -q <quality>')
        sys.exit()

    input_picture = Image.open(inputfile)

    startTime = time.time()
    time.clock()    

    picDeconstructTime = time.time()
    pictureW, pictureH = input_picture.size

    rowH = pictureH / splitByVertical
    rowW = pictureW / splitByHorizontal
    rows = []
    for section_index in range(0, splitByVertical):
        section = (0, rowH * section_index, pictureW, rowH * (section_index + 1))
        row = input_picture.crop(section)
        rows.append(row)
    
    elapsed = time.time() - picDeconstructTime
    print('Image deconstruction:', elapsed)

    threadingTime = time.time()
    with Pool(processes=4) as pool:
        func = partial(buildRows, splitByHorizontal, splitByVertical, quality)
        result_rows = pool.map(func, rows)

    elapsed = time.time() - threadingTime
    print('Threading:', elapsed)

    assemblyTime = time.time()
    resultIMG = None
    for row in result_rows:
        if resultIMG is None:
            resultIMG = row
        else:
            resultIMG = np.vstack((resultIMG, row))

    elapsed = time.time() - assemblyTime
    print('Image construction:', elapsed)

    elapsed = time.time() - startTime
    print('Total elapsed time:', elapsed)
    cv2.imwrite(outputfile, resultIMG)

if __name__=='__main__':
    main(sys.argv[1:])
