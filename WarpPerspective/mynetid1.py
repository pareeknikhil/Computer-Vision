import numpy as np
import cv2
import sys

if len(sys.argv) != 8 :
  print(sys.argv[0], "takes 7 arguments. Not ", len(sys.argv)-1)
  sys.exit()

imagePath = sys.argv[1]
f = float(sys.argv[2])
u0 = float(sys.argv[3])
v0 = float(sys.argv[4])
a = float(sys.argv[5])
b = float(sys.argv[6])
c = float(sys.argv[7])

image = cv2.imread(imagePath)

height, width, _ = image.shape


transformationMatrix= np.float32([
	[c, 0 , -c * u0],
	[0, c, -c * v0],
	[-a, -b, f + a * u0 + b * v0]
])


outputImage = cv2.warpPerspective(image, transformationMatrix, (width, height), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP ,borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Output Image Inverse", outputImage)


cv2.waitKey(0)