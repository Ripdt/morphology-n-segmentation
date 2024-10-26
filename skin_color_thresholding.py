import cv2
import numpy as np

def skin_color_thresholding(img: np.ndarray) -> np.ndarray:
    # Converte para YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Defina o intervalo para os pixels da pele
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([235, 173, 127], dtype=np.uint8)

    # Mascara para filtrar apenas pixels de pele
    mask_skin = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    skin = cv2.bitwise_and(img, img, mask=mask_skin)

    return skin

if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE

    img = imread('res/unexistent-person-1', IMREAD_GRAYSCALE)

    img_skin_color_thresholding = skin_color_thresholding(img=img)

    imshow('basic', img_skin_color_thresholding)
    waitKey(0)
    destroyAllWindows()