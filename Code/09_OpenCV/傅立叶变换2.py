import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 执行傅立叶变换
    dft = cv.dft(np.float32(gray_frame), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray_frame.shape
    crow, ccol = rows // 2, cols // 2

    # 创建低通滤波器掩码
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # 应用掩码并执行逆傅立叶变换
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化并转换为uint8
    img_back_normalized = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
    img_back_uint8 = np.uint8(img_back_normalized)

    # 显示原始帧和处理后的帧
    cv.imshow('Original Frame', gray_frame)
    cv.imshow('Filtered Frame', img_back_uint8)

    # 按 'q' 键退出循环
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
