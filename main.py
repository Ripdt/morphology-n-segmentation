import cv2
import numpy as np

from threshold import basic_threshold, otsu_threshold
from morph import morph

from skin_color_thresholding import skin_color_thresholding
from kmeans import KMeans3D

from os import mkdir
from os.path import exists
from shutil import rmtree

from argparse import ArgumentParser

directory = './out'

def show_and_save_img(img: np.ndarray, filename: str) -> None:
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(directory + '/' + filename + '.png', img)

def project_1() -> None:
    img = cv2.imread('res/sfinge-fingerprint-1.jpg', cv2.IMREAD_GRAYSCALE)
    show_and_save_img(img, 'fingerprint_original')

    img_basic_threshold = basic_threshold(img=img, threshold_value=100)
    img_otsu_threshold = otsu_threshold(img=img)

    show_and_save_img(img_basic_threshold, 'fingerprint_basic_threshold')
    show_and_save_img(img_otsu_threshold, 'fingerprint_otsu_threshold')

    noise = np.random.normal(0, 1, img.shape)  # Gaussian noise
    img_noise = img + noise
    show_and_save_img(img_noise, 'fingerprint_noised')

    img_noised_basic_threshold = basic_threshold(img=img_noise, threshold_value=100)
    img_noised_otsu_threshold = otsu_threshold(img=img_noise)

    img_basic_morphed = morph(img_noised_basic_threshold)
    img_otsu_morphed = morph(img_noised_otsu_threshold)

    show_and_save_img(img_basic_morphed, 'fingerprint_noised_basic_threshold')
    show_and_save_img(img_otsu_morphed, 'fingerprint_noised_otsu_threshold')

def project_2() -> None:
    img_face = cv2.imread('res/unexistent-person-1.jpg')
    show_and_save_img(img_face, 'unexistent-person-1-original')

    img_skin_color, img_skin_binary = skin_color_thresholding(img_face)   
    show_and_save_img(img_skin_binary, 'face_skin_binary')  # Imagem preto e branco
    show_and_save_img(img_skin_color, 'face_skin_color_thresholding')  # Imagem colorida com a pele destacada
    
    img_kmeans = KMeans3D(img_face, k=2, max_iterations=10, imgNameOut="face_kmeans_detection.png")
    show_and_save_img(img_kmeans, 'face_kmeans_detection')

def main() -> None:
    parser = ArgumentParser(description='selection the desired project (\'1\', \'2\' or both)')
    parser.add_argument('argumento', type=str, nargs='?', help='project to be executed')

    args = parser.parse_args()

    if exists(directory):
        rmtree(directory)

    mkdir(directory)

    if args.argumento == '1':
        project_1()
    elif args.argumento == '2':
        project_2()
    else:
        project_1()
        project_2()

if __name__ == "__main__":
    main()