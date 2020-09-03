# Please make sure to: pip install image
# pip install opencv-contrib-python
# python -m pip install matplotlib
import cv2
import numpy as np


# Read In images
one = cv2.imread("./1.jpg")
two = cv2.imread("./2.jpg")


height = one.shape[0]  # Capture img size height
width = one.shape[1]  # Capture img size width

# stting Row sizes and col sizes for the first cut
initRow, initCol, halfRow = int(0), int(0), int(height * .5)
endRow, endCol, halfCol = int(height), int(width), int(width * .5)

# crop first half of image and prep for rotation
crop = one[initRow:endRow, initCol:halfCol]
cropTwo = two[initRow:endRow, initCol:halfCol]

# getSize img
(cropH, cropW) = crop.shape[:2]

# getCenter img
cropCenter = (cropW / 2, cropH / 2)
initCropRow, initCropCol, halfCropRow = int(0), int(0), int(cropH * .5)
endCropRow, endCropCol, halfCropCol = int(cropH), int(cropW), int(cropW * .5)


# get Matrix of imgs and rotate images
MOne = cv2.getRotationMatrix2D(cropCenter, 180, 1.0)
MTwo = cv2.getRotationMatrix2D(cropCenter, 180, 1.0)
rotatedOne = cv2.warpAffine(crop, MOne, (cropW, cropH))
rotatedTwo = cv2.warpAffine(cropTwo, MTwo, (cropW, cropH))

# crop/cut image one and two halves
cropOneAlt = one[initRow: endRow, halfCol: endCol]
cropTwoAlt = two[initRow: endRow, halfCol: endCol]


# Cut the Left side on img one and two, which got rotated
rotatedOneLeft = rotatedOne[initCropRow:endCropRow, initCropCol:halfCropCol]
rotatedTwoRight = rotatedTwo[initCropRow:endCropRow, halfCropCol:endCropCol]

# Cut the Right side on imgs one and two in two Halves (Left, Right)
cropTwiceOneRight = cropOneAlt[initCropRow:endCropRow, halfCropCol:endCropCol]
cropTwicetwoleft = cropTwoAlt[initCropRow:endCropRow, initCropCol:halfCropCol]

# setting desired size to reSize() ----> you can set to any size and Also this ensures the uniform size for stitching
cutSize = (1024, 768)

# image resize for faster computation
ResizedOneRight = cv2.resize(
    cropTwiceOneRight, cutSize, interpolation=cv2.INTER_AREA)
ResizedTwoLeft = cv2.resize(
    cropTwicetwoleft, cutSize, interpolation=cv2.INTER_AREA)
resizedRotatedOneLeft = cv2.resize(
    rotatedOneLeft, cutSize, interpolation=cv2.INTER_AREA)
resizedRotatedTwoRight = cv2.resize(
    rotatedTwoRight, cutSize, interpolation=cv2.INTER_AREA)

# concatenate imgs
threeRight = np.concatenate(
    (ResizedTwoLeft, ResizedOneRight), axis=1)
threeLeft = np.concatenate(
    (resizedRotatedOneLeft, resizedRotatedTwoRight), axis=1)
finalBonus = np.concatenate(
    (threeLeft, threeRight), axis=1)


# Show image
cv2.imshow('threeleft', threeLeft)
cv2.imshow("threeRight", threeRight)
cv2.imshow('final', finalBonus)

# Save img
cv2.imwrite('3-left.jpg', threeLeft)
cv2.imwrite(" 3-right.jpg", threeRight)
cv2.imwrite("finalBonus.jpg", finalBonus)

cv2.waitKey(0)
cv2.destroyAllWindows()
