import cv2
import numpy as np
from PIL import Image

#== Parameters
BLUR = 21
CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 55
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format

#-- face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')

#-- Read image
img = cv2.imread('./emoji/face.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area
contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
for c in contour_info:
    cv2.fillConvexPoly(mask, c[0], (255))

#-- Smooth mask, then blur it
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background
mask_stack  = mask_stack.astype('float32') / 255.0
img         = img.astype('float32') / 255.0
masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
masked = (masked * 255).astype('uint8')

cv2.imwrite('./emoji/masked.png', masked)

cv2.imshow('img', masked)                                   # Display
cv2.waitKey()
cv2.imwrite("WTF.jpg",masked)

img = Image.open('./emoji/masked.png').convert('RGB')
img = img.convert("RGBA")
datas = img.getdata()

newData = []

for item in datas:
    if item[0] >= 150 and item[1] < 50 and item[2] < 50:
        newData.append((item[0], item[1], item[2], 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save('./emoji/masked-final.png', "PNG")

img2 = cv2.imread('./emoji/masked-final.png', cv2.IMREAD_UNCHANGED)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray2, scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
for (x,y,w,h) in faces:
    cropped = img2[y:y + int(h/1.05),x:x + w]
    cv2.imwrite("./emoji/cropped.png", cropped)