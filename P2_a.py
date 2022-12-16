# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy import linalg as la
import random

def load_input():
    I1 = cv.imread('library1.jpg')
    I2 = cv.imread('library2.jpg')
    lmatches = np.loadtxt('library_matches.txt')

    I3 = np.zeros((I1.shape[0], I1.shape[1] * 2, 3))
    I3[:, :I1.shape[1], :] = I1
    I3[:, I1.shape[1]:, :] = I2
    return I1, I2, I3, matches

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

def f_residual(matches,F):
    residual = 0
    for i in range(len(matches)):
        X_2 = [matches[i][2], matches[i][3],1]
        X_1 = [[matches[i][0]],[matches[i][1]],[1]]
        residual += X_2 @ F @ X_1
    return residual

def fit_fundamental(iteration, matches):
    N = len(matches)
    pre_residual = 1000
    F_copy = []
    for _iter in range(iteration):
        F_unnoramlized, F_normalized = fundamental_matrix(matches)
        residual = f_residual(matches, F_normalized)
        if pre_residual > abs(residual):
            pre_residual = abs(residual)
            F_copy = F_normalized
        print(pre_residual)
    return F_copy

if __name__ == '__main__':
    I1, I2, I3, matches = load_input()
    # plot_matches(I1, I3, matches)

    F = fit_fundamental(iteration=3000, matches = matches)
    visualize_epipolar(F,matches,I2)
