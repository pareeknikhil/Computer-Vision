import numpy as np
import cv2
import sys

if len(sys.argv) != 3 :
  print(sys.argv[0], "takes 2 arguments. Not ", len(sys.argv)-1)
  sys.exit()

imagePath = sys.argv[1]
c = float(sys.argv[2])

image = cv2.imread(imagePath)

height, width, _ = image.shape

f = 1
u0 = width/2
v0 = height/2
a = c/1000
b = 0

transformationMatrix1= np.float32([
	[c, 0 , -c * u0],
	[0, c, -c * v0],
	[-a, -b, f + a * u0 + b * v0]
])

a = 0
b = -c/1000

transformationMatrix2= np.float32([
	[c, 0 , -c * u0],
	[0, c, -c * v0],
	[-a, -b, f + a * u0 + b * v0]
])

outputImage1 = cv2.warpPerspective(image, transformationMatrix1, (width*2, height*2), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT )
outputImage2 = cv2.warpPerspective(image, transformationMatrix2, (width*2, height*2), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP , borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Output Image 1", outputImage1)
cv2.imshow("Output Image 2", outputImage2)

cv2.waitKey(0)