import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 使用ORB描述符进行Brute-Force匹配
# img1 = cv.imread("imgs/target.jpg", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("imgs/face.png", cv.IMREAD_GRAYSCALE)
#
# orb = cv.ORB.create()
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
#
# img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()

# 带有SIFT描述符和比例测试的Brute-Force匹配
# img1 = cv.imread("imgs/test1.jpg", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("imgs/test2.png", cv.IMREAD_GRAYSCALE)
# sift = cv.SIFT.create()
#
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
#
# good = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
#                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()

# 基于匹配器的FLANN
# img1 = cv.imread("imgs/test1.jpg", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("imgs/test2.png", cv.IMREAD_GRAYSCALE)
# sift = cv.SIFT.create()
#
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
# matchesMask = [[0, 0] for i in range(len(matches))]
#
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]
# draw_params = dict(matchColor=(0, 255, 0),
#                    singlePointColor=(255, 0, 0),
#                    matchesMask=matchesMask,
#                    flags=cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# plt.imshow(img3)
# plt.show()

# 特征匹配 + 单应性查找对象
MIN_MATCH_COUNT = 10
img1 = cv.imread("imgs/target.jpg", 0)
img2 = cv.imread("imgs/face.jpg", 0)
sift = cv.SIFT.create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # print(img1.shape)
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3)
plt.show()
