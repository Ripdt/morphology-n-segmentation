import cv2
import numpy as np

from threshold import basic_threshold, otsu_threshold
from morph import morph

from metrics import calculate_psnr

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
    show_and_save_img(img, 'original')

    noise = np.random.normal(0, 1, img.shape)  # Gaussian noise
    img_noise = cv2.add(img, noise.astype(np.uint8))
    show_and_save_img(img_noise, 'noised')

    _, img_basic_threshold = cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    _, img_otsu_threshold = cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show_and_save_img(img_basic_threshold, 'basic_threshold')
    show_and_save_img(img_otsu_threshold, 'otsu_threshold')

    _, img_noised_basic_threshold = cv2.threshold(img_noise, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    _, img_noised_otsu_threshold = cv2.threshold(img_noise, thresh=100, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show_and_save_img(img_noised_basic_threshold, 'noised_basic_threshold')
    show_and_save_img(img_noised_otsu_threshold, 'noised_otsu_threshold')

    img_basic_morphed_dilate = morph(img_noised_basic_threshold, kernel_width=2, kernel_height=2, dilate=True)
    img_otsu_morphed_dilate = morph(img_noised_otsu_threshold, kernel_width=2, kernel_height=2, dilate=True)

    show_and_save_img(img_basic_morphed_dilate, 'morphed_dilate_basic_threshold')
    show_and_save_img(img_otsu_morphed_dilate, 'morphed_dilate_otsu_threshold')

    img_basic_morphed_eroded = morph(img_noised_basic_threshold, kernel_width=2, kernel_height=2, dilate=False)
    img_otsu_morphed_eroded  = morph(img_noised_otsu_threshold, kernel_width=2, kernel_height=2, dilate=False)

    show_and_save_img(img_basic_morphed_eroded, 'morphed_erode_basic_threshold')
    show_and_save_img(img_otsu_morphed_eroded, 'morphed_erode_otsu_threshold')

    img_basic_morphed_dilated_eroded = morph(img_basic_morphed_dilate, kernel_width=2, kernel_height=2, dilate=False)
    img_otsu_morphed_dilated_eroded  = morph(img_otsu_morphed_dilate, kernel_width=2, kernel_height=2, dilate=False)

    show_and_save_img(img_basic_morphed_dilated_eroded, 'morphed_dilated_erode_basic_threshold')
    show_and_save_img(img_otsu_morphed_dilated_eroded, 'morphed_dilated_erode_otsu_threshold')

    psnr_basic_threshold_dilated = calculate_psnr(img_basic_threshold, img_basic_morphed_dilate)
    psnr_basic_threshold_eroded = calculate_psnr(img_basic_threshold, img_basic_morphed_eroded)
    psnr_basic_threshold_dilated_eroded = calculate_psnr(img_basic_threshold, img_basic_morphed_dilated_eroded)

    print('Projeto 1\n\nPSNR:')
    print(F'\n\t- Limiarização básica dilatada: {psnr_basic_threshold_dilated}')
    print(F'\n\t- Limiarização básica erodida: {psnr_basic_threshold_eroded}')
    print(F'\n\t- Limiarização básica dilatada+erodida: {psnr_basic_threshold_dilated_eroded}')

def project_2() -> None:
    img_face = cv2.imread('res/unexistent-person-1.jpg')
    show_and_save_img(img_face, 'unexistent-person-1-original')

    img_skin = skin_color_thresholding(img_face)
    show_and_save_img(img_skin, 'face_skin_color_thresholding')

    img_kmeans = KMeans3D(img=img_face, k=2)
    show_and_save_img(img_kmeans, 'face_kmeans_detection')

def main() -> None:
    parser = ArgumentParser(description='selection the desired project (\'1\', \'2\' or both)')
    parser.add_argument('project', type=str, nargs='?', help='project to be executed')

    args = parser.parse_args()

    if exists(directory):
        rmtree(directory)

    mkdir(directory)

    if args.project == '1':
        project_1()
    elif args.project == '2':
        project_2()
    else:
        project_1()
        project_2()

if __name__ == "__main__":
    main()