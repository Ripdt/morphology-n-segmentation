import cv2
import numpy as np

from threshold import basic_threshold, otsu_threshold
from morph import morph

from skin_color_thresholding import skin_color_thresholding
from kmeans import kmeans_face_detection

from os import mkdir
from os.path import exists
from shutil import rmtree

# Criação do diretório de saída
directory = './out'
if exists(directory):
    rmtree(directory)

mkdir(directory)

def show_and_save_img(img: np.ndarray, filename: str) -> None:
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(directory + '/' + filename + '.png', img)

# ========== project 1 ==========

img = cv2.imread('res/sfinge-fingerprint-1.jpg', cv2.IMREAD_GRAYSCALE)
show_and_save_img(img, 'fingerprint_original')

noise = np.random.normal(0, 50, img.shape)
img_noise = img + noise
show_and_save_img(img_noise, 'fingerprint_noised')

img_basic_threshold = basic_threshold(img=img_noise, threshold_value=100)
img_otsu_threshold = otsu_threshold(img=img_noise) 

img_basic_morphed = morph(img_basic_threshold)
img_otsu_morphed = morph(img_otsu_threshold)

show_and_save_img(img_basic_morphed, 'fingerprint_basic_threshold')
show_and_save_img(img_otsu_morphed, 'fingerprint_otsu_threshold')

# ========== project 2 ==========

img_face = cv2.imread('res/unexistent-person-1.jpg')
show_and_save_img(img_face, 'unexistent-person-1-original')

img_skin = skin_color_thresholding(img_face)
show_and_save_img(img_skin, 'face_skin_color_thresholding')

img_kmeans = kmeans_face_detection(img=img_face, k=2)
show_and_save_img(img_kmeans, 'face_kmeans_detection')

