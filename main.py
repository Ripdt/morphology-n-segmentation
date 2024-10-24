import cv2
import numpy as np

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

img = cv2.imread('res/unexistent-person-1.jpg', cv2.IMREAD_GRAYSCALE)
show_and_save_img(img, 'unexistent-person-1')