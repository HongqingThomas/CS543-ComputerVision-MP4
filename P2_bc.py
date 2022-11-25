from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy import linalg as la
import random

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def load_data():
    lab_2d = np.loadtxt('lab_matches.txt')
    lab_3d = np.loadtxt('lab_3d.txt')
    lab1_2d = lab_2d[:,0:2]
    lab2_2d = lab_2d[:,2:4]
    return lab_3d,lab1_2d,lab2_2d

def Projection_matrix(data_2d,data_3d):
    sample_index = random.sample(range(0, 20), 6)
    a = np.zeros(shape=(12,12))
    for i in range(6):
        a[2 * i][0] = 0
        a[2 * i][1] = 0
        a[2 * i][2] = 0
        a[2 * i][3] = 0
        a[2 * i][4] = -1 * data_3d[sample_index[i]][0]
        a[2 * i][5] = -1 * data_3d[sample_index[i]][1]
        a[2 * i][6] = -1 * data_3d[sample_index[i]][2]
        a[2 * i][7] = -1
        a[2 * i][8] = data_2d[sample_index[i]][1] * data_3d[sample_index[i]][0]
        a[2 * i][9] = data_2d[sample_index[i]][1] * data_3d[sample_index[i]][1]
        a[2 * i][10] = data_2d[sample_index[i]][1] * data_3d[sample_index[i]][2]
        a[2 * i][11] = data_2d[sample_index[i]][1]
        a[2 * i + 1][0] = data_3d[sample_index[i]][0]
        a[2 * i + 1][1] = data_3d[sample_index[i]][1]
        a[2 * i + 1][2] = data_3d[sample_index[i]][2]
        a[2 * i + 1][3] = 1
        a[2 * i + 1][4] = 0
        a[2 * i + 1][5] = 0
        a[2 * i + 1][6] = 0
        a[2 * i + 1][7] = 0
        a[2 * i + 1][8] = -1 *data_2d[sample_index[i]][0] * data_3d[sample_index[i]][0]
        a[2 * i + 1][9] = -1 *data_2d[sample_index[i]][0] * data_3d[sample_index[i]][1]
        a[2 * i + 1][10] = -1 *data_2d[sample_index[i]][0] * data_3d[sample_index[i]][2]
        a[2 * i + 1][11] = -1 *data_2d[sample_index[i]][0]
    U, S, V = np.linalg.svd(a)
    projection_matrix = V[len(V) - 1].reshape(3,4)
    projection_matrix = projection_matrix/projection_matrix[2][3]
    return projection_matrix

def check_residual(projection_matrix, points_3d, points_2d):
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(projection_matrix, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    return residual

def find_camera_projection_matrix(data_2d, data_3d,iterations):
    pre_residual = 10000
    actual_projection_matrix = []
    for iter in range(iterations):
        projection_matrix = Projection_matrix(data_2d,data_3d)
        residual = check_residual(projection_matrix, data_3d, data_2d)
        if residual < pre_residual:
            pre_residual = residual
            actual_projection_matrix = projection_matrix
    return actual_projection_matrix

def visualize_3dprojects(lab_2d,lab_2d_proj):
    Img = cv.imread('lab1.jpg')
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(Img / 255).astype(float))
    ax.plot(lab_2d[:, 0], lab_2d[:, 1], '+r')
    ax.plot(lab_2d_proj[:, 0], lab_2d_proj[:, 1], '+g')
    plt.show()

def camera_matrix(projection_matrix):
    # matrix A's null space means v that make AV=0
    camera_center = np.zeros(shape = (3,1))
    U, S, V = np.linalg.svd(projection_matrix)
    homography_camera_center = V[len(V) - 1].reshape(1,4)
    for i in range(3):
        camera_center[i] = homography_camera_center[0][i] / homography_camera_center[0][3]
    return camera_center.transpose()

if __name__ == '__main__':
    lab_3d,lab1_2d,lab2_2d = load_data()
    lab_target = lab1_2d
    actual_projection_matrix = find_camera_projection_matrix(lab_target, lab_3d,iterations = 100)
    points_3d_proj, residual = evaluate_points(actual_projection_matrix, lab_target, lab_3d)
    visualize_3dprojects(lab_target,points_3d_proj)
    print("residual:",residual)
    print("projection_matrix:",actual_projection_matrix)

    # show camera center
    # for lab(estimated matrix)
    camera_center = camera_matrix(actual_projection_matrix)
    print("camera_center:",camera_center)

    # # for library(provided matrix)
    # lab_projection_matrix = np.loadtxt('library2_camera.txt')
    # camera_center = camera_matrix(lab_projection_matrix)
    # print("camera_center:",camera_center)