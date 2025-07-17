import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("imgs/target.jpg", 0)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
# plt.show()

rows, cols = img.shape
crow, ccol = rows//2, cols//2

mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
img_back_normalized = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
img_back_uint8 = np.uint8(img_back_normalized)

cv.imshow('t', img_back_uint8)

cv.waitKey(0)
cv.destroyAllWindows()
