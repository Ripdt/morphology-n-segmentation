import cv2
import numpy as np

from threshold import basic_threshold, otsu_threshold
from morph import morph

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

noise = np.random.normal(0, 50, img.shape)
img_noise = img + noise
show_and_save_img(img_noise, 'fingerprint_noised')

img_basic_threshold = basic_threshold(img=img_noise, threshold_value=100)
img_otsu_threshold = otsu_threshold(img=img_noise) # TODO fix crash

img_basic_morphed = morph(img_basic_threshold)
img_otsu_morphed = morph(img_otsu_threshold)

show_and_save_img(img_basic_morphed, 'fingerprint_basic_threshold')
show_and_save_img(img_otsu_morphed, 'fingerprint_otsu_threshold')