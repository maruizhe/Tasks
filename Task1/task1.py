import cv2
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)

img = cv2.imread('./before.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,43,46])
upper_red = np.array([16,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)

res = cv2.bitwise_and(img, img, mask = mask)

# cv_show ('res/hsv/res', np.hstack((img, hsv, res)))

res1 = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
# cv_show('bgr', gray)

erode = cv2.erode(gray, None, iterations = 1)
dilate = cv2.dilate(erode, None, iterations = 1)
# cv_show('dilate', dilate)

ret, thresh = cv2.threshold(dilate, 20, 255, cv2.THRESH_BINARY)
# cv_show('thresh', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

x, y, w, h = cv2.boundingRect(cnt)
img1 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
# cv_show('after', img1)

text = str(x) + ' ' + str(y)
cv2.putText(img1, text, (x-35, y-15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
cv_show('after', img1)

cv2.imwrite('./after.png', img1)
