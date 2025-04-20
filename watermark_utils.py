import numpy as np
import cv2
import pywt

def rgb2yiq(img):
    transformation = np.array([[0.299, 0.587, 0.114],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
    return img @ transformation.T

def yiq2rgb(img):
    transformation = np.array([[1.0, 0.956, 0.621],
                               [1.0, -0.272, -0.647],
                               [1.0, -1.106, 1.703]])
    rgb_img = img @ transformation.T
    return np.clip(rgb_img, 0, 255).astype(np.uint8)

def get_edge_mask(img, threshold=100):
    edges = cv2.Canny(img.astype(np.uint8), threshold, threshold*2)
    return (edges > 0).astype(np.float32)

def dwt2(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    return coeffs2[0], coeffs2[1]

def idwt2(LL, bands):
    return pywt.idwt2((LL, bands), 'haar')

# ✅ Fixed: Return S as 2D diagonal matrix
def svd2(band):
    U, S, V = np.linalg.svd(band, full_matrices=False)
    S_mat = np.diag(S)
    return U, S_mat, V

# ✅ Fixed: Use already diagonal S matrix (no np.diag needed)
def isvd2(U, S, V):
    return U @ S @ V

def arnold_scramble(img, times=10):
    N = img.shape[0]
    res = img.copy()
    for _ in range(times):
        new_img = np.zeros_like(res)
        for x in range(N):
            for y in range(N):
                new_x = (x + y) % N
                new_y = (x + 2*y) % N
                new_img[new_x, new_y] = res[x, y]
        res = new_img
    return res

def arnold_unscramble(img, times=10):
    N = img.shape[0]
    res = img.copy()
    for _ in range(times):
        new_img = np.zeros_like(res)
        for x in range(N):
            for y in range(N):
                new_x = (2*x - y) % N
                new_y = (-x + y) % N
                new_img[new_x, new_y] = res[x, y]
        res = new_img
    return res

def resize_image(img, shape):
    return cv2.resize(img, (shape[1], shape[0]))
