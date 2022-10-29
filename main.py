# Britney Clark
# Portfolio Project Milestone 3
# CSC515: Foundations of Computer Vision
# Dr. Joseph Issa
# February 6, 2022

import cv2
import numpy as np

# Function for Gamma Correction
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

# Load Cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Import and read images
img1 = cv2.imread(r'C:\Users\Stardust\Downloads\PXL_20220221_020344443.MP.jpg')

# Scale Images
scale_percent = 20
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)

res1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Image 1.1 Scaled', res1)
cv2.waitKey(0)

# Blur Facial Features
blurred = cv2.medianBlur(res1, 5)
cv2.imshow("Image1 Blurred", blurred)
cv2.waitKey(0)

# Gamma Correction
gamma = 2.5
adjusted1 = adjust_gamma(blurred, gamma=gamma)
cv2.imshow("Image1 Lighting Corrected", adjusted1)
cv2.waitKey(0)

# Detect Faces and Eyes
face1 = face_cascade.detectMultiScale(adjusted1, 1.1, 2)

# Draw Bounding Boxes
i = 0
for (x, y, w, h) in face1:
    face_center = (x + w // 2, y + h // 2)
    radius = int(round((w + h) * 0.25))
    cv2.circle(adjusted1, face_center, radius, (0, 255, 0), 2)
    i = i + 1
    cv2.imwrite('New Image1-' + str(i) + '.jpg', adjusted1[y:y+h, x:x+w])
    print([x, y, w, h])

# Display images with Bounding Boxes
cv2.imshow('Image 1 Detected', adjusted1)
cv2.waitKey(0)

# Upload Crop Images
img11 = cv2.imread(r'C:\Users\Stardust\PycharmProjects\515PortProj\New Image1-1.jpg')
cv2.imshow('Image 1.1 Cropped', img11)
cv2.waitKey(0)

# Detect Eyes
eyes11 = eye_cascade.detectMultiScale(img11)
for (ex, ey, ew, eh) in eyes11:
    cv2.rectangle(img11, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

cv2.imshow('Image 1.1 Eyes Detected', img11)
cv2.waitKey(0)

image = cv2.putText(img11, 'This is me', (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('Image 1.1 Eyes Detected', img11)
cv2.waitKey(0)

