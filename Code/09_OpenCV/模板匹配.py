import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("imgs/target.jpg", 0)
img2 = img.copy()
template = cv.imread("imgs/face.png", 0)
w, h = template.shape[::-1]
methods = [
    'cv.TM_CCOEFF', 
    'cv.TM_CCOEFF_NORMED', 
    'cv.TM_CCORR', 
    'cv.TM_CCORR_NORMED', 
    'cv.TM_SQDIFF', 
    'cv.TM_SQDIFF_NORMED'
    ]

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

# img_rgb = cv.imread("imgs/target1.jpeg")
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread("imgs/jinbi.png", 0)

# w, h = template.shape[::-1]
# res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
# threshold1 = 0.4
# loc = np.where(threshold1 <= res)

# for pt in zip(*loc[::-1]):
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# cv.imshow('res', img_rgb)
# cv.waitKey(0)
# cv.destroyAllWindows()
