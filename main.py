import cv2
import numpy as np

from threshold import basic_threshold, otsu_threshold
from morph import morph

from metrics import calculate_psnr
from metrics_segmentation import calculate_iou, calculate_precision, calculate_recall, calculate_f1_score

from skin_color_thresholding import skin_color_thresholding
from kmeans import KMeans3D
from seeds import seeds

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

    img_basic_threshold = basic_threshold(img=img, threshold_value=100)
    img_otsu_threshold = otsu_threshold(img=img)

    show_and_save_img(img_basic_threshold, 'basic_threshold')
    show_and_save_img(img_otsu_threshold, 'otsu_threshold')

    img_noised_basic_threshold = basic_threshold(img=img_noise, threshold_value=100)
    img_noised_otsu_threshold = otsu_threshold(img=img_noise)

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

    print('\nProjeto 1\n\nPSNR:')

    psnr_basic_x_otsu = calculate_psnr(img_basic_threshold, img_otsu_threshold)
    psnr_otsu_x_basic = calculate_psnr(img_otsu_threshold, img_basic_threshold)

    print(F'\n\t- Limiarização básica x Otsu: {psnr_basic_x_otsu}')
    print(F'\n\t- Limiarização Otsu x básica: {psnr_otsu_x_basic}')

    psnr_basic_threshold_dilated = calculate_psnr(img_basic_threshold, img_basic_morphed_dilate)
    psnr_basic_threshold_eroded = calculate_psnr(img_basic_threshold, img_basic_morphed_eroded)
    psnr_basic_threshold_dilated_eroded = calculate_psnr(img_basic_threshold, img_basic_morphed_dilated_eroded)

    print(F'\n\t- Limiarização básica dilatada: {psnr_basic_threshold_dilated}')
    print(F'\n\t- Limiarização básica erodida: {psnr_basic_threshold_eroded}')
    print(F'\n\t- Limiarização básica dilatada+erodida: {psnr_basic_threshold_dilated_eroded}')

    psnr_otsu_threshold_dilated = calculate_psnr(img_otsu_threshold, img_otsu_morphed_dilate)
    psnr_otsu_threshold_eroded = calculate_psnr(img_otsu_threshold, img_otsu_morphed_eroded)
    psnr_otsu_threshold_dilated_eroded = calculate_psnr(img_otsu_threshold, img_otsu_morphed_dilated_eroded)

    print(F'\n\t- Limiarização Otsu dilatada: {psnr_otsu_threshold_dilated}')
    print(F'\n\t- Limiarização Otsu erodida: {psnr_otsu_threshold_eroded}')
    print(F'\n\t- Limiarização Otsu dilatada+erodida: {psnr_otsu_threshold_dilated_eroded}')
    print('\n')

def project_2() -> None:
    img_face = cv2.imread('res/unexistent-person-1.jpg')
    show_and_save_img(img_face, 'unexistent-person-1-original')  # Original Image

    ground_truth_mask = cv2.imread('res/unexistent-person-1-binary-mask.png')
    ground_truth_mask_gray = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, ground_truth_mask_binary = cv2.threshold(ground_truth_mask_gray, 127, 255, cv2.THRESH_BINARY)  # Create binary mask

    img_skin, img_skin_binary = skin_color_thresholding(img_face)
    show_and_save_img(img_skin_binary, 'face_skin_binary')  # Binary image of the skin
    show_and_save_img(img_skin, 'face_skin_color_thresholding')  # Image with skin highlighted
    
    # Calculate metrics for Skin Color Thresholding
    skin_binary_mask = img_skin_binary  # The generated binary mask
    iou_skin = calculate_iou(skin_binary_mask, ground_truth_mask_binary)  # Use binary ground truth mask
    precision_skin = calculate_precision(skin_binary_mask, ground_truth_mask_binary)
    recall_skin = calculate_recall(skin_binary_mask, ground_truth_mask_binary)
    f1_skin = calculate_f1_score(precision_skin, recall_skin)

    print(f"Skin Color Thresholding Metrics: IoU={iou_skin}, Precision={precision_skin}, Recall={recall_skin}, F1 Score={f1_skin}")

    # Calculate metrics for SEEDS
    labels, img_seeds = seeds(img_face)
    show_and_save_img(img_seeds, 'image_seeds')  # Image after SEEDS

    # Create a binary mask from the SEEDS output
    seeds_binary_mask = (labels > 0).astype(np.uint8) * 255  # Convert labels to binary mask

    iou_seeds = calculate_iou(seeds_binary_mask, ground_truth_mask_binary)
    precision_seeds = calculate_precision(seeds_binary_mask, ground_truth_mask_binary)
    recall_seeds = calculate_recall(seeds_binary_mask, ground_truth_mask_binary)
    f1_seeds = calculate_f1_score(precision_seeds, recall_seeds)

    print(f"SEEDS Metrics: IoU={iou_seeds}, Precision={precision_seeds}, Recall={recall_seeds}, F1 Score={f1_seeds}")

    # K-means - Rodar no próprio método
    #img_kmeans = KMeans3D(img=img_face, k=2)
    #show_and_save_img(img_kmeans, 'face_kmeans_detection')  # Image after K-means
    #kmeans_binary_mask = (img_kmeans > 0).astype(np.uint8) * 255  # Adjust to get the binary mask

    #iou_kmeans = calculate_iou(kmeans_binary_mask, ground_truth_mask_binary)
    #precision_kmeans = calculate_precision(kmeans_binary_mask, ground_truth_mask_binary)
    #recall_kmeans = calculate_recall(kmeans_binary_mask, ground_truth_mask_binary)
    #f1_kmeans = calculate_f1_score(precision_kmeans, recall_kmeans)

    #print(f"K-means Metrics: IoU={iou_kmeans}, Precision={precision_kmeans}, Recall={recall_kmeans}, F1 Score={f1_kmeans}")

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