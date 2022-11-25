from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy import linalg as la
import random
# from scipy.spatial import distance
import scipy

def load_input():
    I1 = cv.imread('house1.jpg')
    I2 = cv.imread('house2.jpg')
    # I1 = cv.imread('E:/UIUC_course/FALL_SEMESTER/Computer_Vision/MP4/MP4_part2_data/MP4_part2_data/gaudi1.jpg')
    # I2 = cv.imread(r"E:\UIUC_course\FALL_SEMESTER\Computer_Vision\MP4\MP4_part2_data\MP4_part2_data\gaudi2.jpg")
    I3 = np.zeros((I1.shape[0], I1.shape[1] * 2, 3))
    I3[:, :I1.shape[1], :] = I1
    I3[:, I1.shape[1]:, :] = I2
    return I1, I2, I3

def SIFT_detect(img1, img2, threshold):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    sift1 = cv.SIFT_create()
    kp1, des1 = sift1.detectAndCompute(gray1, None)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray2, None)

    distance = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')

    pairs = []
    for row in range(des1.shape[0]):  # des1.shape[0]
        if (np.min(distance, axis=1)[row] < threshold):
            pairs.append(np.where(distance == np.min(distance, axis=1)[row]))
    print("pairs number: ", len(pairs))
    all_inliers = np.zeros((len(pairs), 4))
    for i in range(len(pairs)):
        all_inliers[i][0] = kp1[pairs[i][0][0]].pt[0]  # image1 x
        all_inliers[i][1] = kp1[pairs[i][0][0]].pt[1]  # image1 y
        all_inliers[i][2] = kp2[pairs[i][1][0]].pt[0]  # image2 x
        all_inliers[i][3] = kp2[pairs[i][1][0]].pt[1]  # image2 x
    return all_inliers

def plot_matches(I1, I3, matches):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3 / 255).astype(float))
    ax.plot(matches[:, 0], matches[:, 1], '+r')
    ax.plot(matches[:, 2] + I1.shape[1], matches[:, 3], '+r')
    ax.plot([matches[:, 0], matches[:, 2] + I1.shape[1]], [matches[:, 1], matches[:, 3]], 'r')
    plt.show()

def visualize_epipolar(F,matches,I2):
    N = len(matches)
    M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image
    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 100  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 100

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2 / 255).astype(float))
    ax.plot(matches[:, 2], matches[:, 3], '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()

def fundamental_matrix(matches):
    N = len(matches)
    sample_index = random.sample(range(0, N), 8)
    A = np.zeros(shape=(8,9))
    for i in range(8):
        A[i][0] = matches[sample_index[i]][0] * matches[sample_index[i]][2]
        A[i][1] = matches[sample_index[i]][1] * matches[sample_index[i]][2]
        A[i][2] = matches[sample_index[i]][2]
        A[i][3] = matches[sample_index[i]][0] * matches[sample_index[i]][3]
        A[i][4] = matches[sample_index[i]][1] * matches[sample_index[i]][3]
        A[i][5] = matches[sample_index[i]][3]
        A[i][6] = matches[sample_index[i]][0]
        A[i][7] = matches[sample_index[i]][1]
        A[i][8] = 1
    U, S, V = la.svd(A)
    F_unnoramlized = V[len(V) - 1].reshape(3, 3)
    _U, _sigma, _V = la.svd(F_unnoramlized)
    _sigma[2] = 0
    F_normalized = _U @ (np.diag(_sigma)) @ _V
    return F_unnoramlized,F_normalized

def f_residual(threshold, matches, F):
    total_inlier = 0
    residual = 0
    inlier = np.zeros(shape=(len(matches),4))
    for i in range(len(matches)):
        X_2 = [matches[i][2], matches[i][3],1]
        X_1 = [[matches[i][0]],[matches[i][1]],[1]]
        each_residual = X_2 @ F @ X_1
        if abs(each_residual) < threshold:
            inlier[total_inlier] = matches[i]
            total_inlier += 1
            residual += abs(each_residual)
    return total_inlier, residual, inlier

def normalize(matches):
    N = len(matches)
    normalize_match = matches
    max = np.max(matches, axis=0)
    min = np.min(matches, axis=0)
    for col in range(4):
        for row in range(N):
            normalize_match[row][col] = (matches[row][col] - min[col])/(max[col] - min[col])
    return normalize_match

def fit_fundamental(iteration, threshold, matches):
    N = len(matches)
    pre_inlier = 0
    F_copy = []
    for _iter in range(iteration):
        F_unnoramlized, F_normalized = fundamental_matrix(matches)
        total_inlier, residual, inlier_pairs = f_residual(threshold, matches, F_normalized)
        if pre_inlier < total_inlier:
            pre_inlier = total_inlier
            pre_residual = float(residual / pre_inlier)
            F_copy = F_normalized
            pre_inlier_pairs = inlier_pairs[:pre_inlier, :]
    return F_copy, pre_inlier, pre_residual, pre_inlier_pairs

if __name__ == '__main__':
    I1, I2, I3 = load_input()
    matches = SIFT_detect(I1, I2, threshold=20000)
    plot_matches(I1, I3, matches)

    # normalize_match = normalize(matches)
    F, inliers, residual, inlier_pairs = fit_fundamental(iteration=10000, matches = matches, threshold = 0.001)
    print("residual: ", residual)
    print("inlier numbers: ", inliers)
    # visualize_epipolar(F,matches,I2)
    visualize_epipolar(F, inlier_pairs, I2)
    # print("inlier_pairs[:inliers,:]: ", inlier_pairs)
