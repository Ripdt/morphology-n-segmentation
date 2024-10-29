import cv2
import numpy as np

def skin_color_thresholding(img: np.ndarray) -> (np.ndarray, np.ndarray):
    imagem_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Ajusta o intervalo de cores para a pele em HSV
    lower_skin = np.array([0, 30, 100], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Criar uma máscara para o intervalo de pele
    mascara_pele = cv2.inRange(imagem_hsv, lower_skin, upper_skin)

    # Aplicar a máscara à imagem original para isolar a pele
    resultado_colorido = cv2.bitwise_and(img, img, mask=mascara_pele)

    # Aplica morfologia para suavizar a máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_pele = cv2.morphologyEx(mascara_pele, cv2.MORPH_CLOSE, kernel)

    resultado_colorido = cv2.bitwise_and(img, img, mask=mascara_pele)

    resultado_binario = cv2.cvtColor(resultado_colorido, cv2.COLOR_BGR2GRAY)
    _, binario = cv2.threshold(resultado_binario, 1, 255, cv2.THRESH_BINARY)

    return resultado_colorido, binario


if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows

    img = imread('res/unexistent-person-1.jpg') 

    img_skin_color, img_skin_binary = skin_color_thresholding(img=img)

    imshow('Imagem com Pele Destacada', img_skin_color)
    imshow('Imagem Binária', img_skin_binary)
    waitKey(0)
    destroyAllWindows()
