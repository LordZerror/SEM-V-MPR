import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 100  # only to increase the size by tiny bit
imgSize = 500

folder = "Data/Testing"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # bounding box

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("a"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_A{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("b"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_B{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("c"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_C{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("d"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_D{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("e"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_E{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("f"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_F{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("g"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_G{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("h"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_H{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("i"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_I{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("j"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_J{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("k"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_K{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("l"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_L{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("m"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_M{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("n"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_N{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("o"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_O{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("p"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_P{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("q"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_Q{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("r"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_R{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_S{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("t"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_T{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("u"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_U{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("v"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_V{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("w"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_W{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("x"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_X{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("y"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_Y{time.time()}.jpg', imgWhite)
        print(counter)
    if key == ord("z"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_Z{time.time()}.jpg', imgWhite)
        print(counter)
    '''
    Need to have points for better classification by the classifier
    '''
