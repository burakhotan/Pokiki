import json
import numpy as np
import Helper
from pathlib import Path
from PIL import Image

rootFolder = Path('./')

tilesFolder = rootFolder / "tiles/"
dataFilePath = rootFolder  / 'out/data.json'

data = Helper.loadFileJSON(dataFilePath)
vstackedIMG = np.array([])

for count, imgName in enumerate(data):
    image=Image.open( str(tilesFolder / imgName) )
    image.load()

    imageSize = image.size
    imageBox = image.getbbox()

    imageComponents = image.split()

    rgbImage = Image.new("RGB", imageSize, (0,0,0))
    rgbImage.paste(image, mask=imageComponents[3])
    croppedBox = rgbImage.getbbox()

    if imageBox != croppedBox:
        cropped=image.crop(croppedBox)
        cropped.save(str(tilesFolder / "cropped/" / imgName))
    else:
        print('same!!')