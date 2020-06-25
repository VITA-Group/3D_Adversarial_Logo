import os
import cv2 as cv
for file in os.listdir('output'):
    path = os.path.join('output', file)
    img = cv.imread(path)
    print(img.shape)
    img = cv.resize(img, (416, 416))
    cv.imwrite(path, img)