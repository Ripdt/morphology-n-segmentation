import cv2
import numpy as np

from threshold import basic_threshold

def skin_color_thresholding(img: np.ndarray) -> (np.ndarray, np.ndarray):
    imagem_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Ajusta o intervalo de cores para a pele em HSV
    lower_skin = np.array([0, 30, 100], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Cria a máscara de pele 
    altura, largura, _ = imagem_hsv.shape
    mascara_pele = np.zeros((altura, largura), dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            h, s, v = imagem_hsv[i, j]
            if (lower_skin[0] <= h <= upper_skin[0] and
                lower_skin[1] <= s <= upper_skin[1] and
                lower_skin[2] <= v <= upper_skin[2]):
                mascara_pele[i, j] = 255  # Define pixel como parte da pele

    # Aplica morfologia para suavizar a máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_pele = cv2.morphologyEx(mascara_pele, cv2.MORPH_CLOSE, kernel)

    # Aplica a máscara à imagem original para isolar a pele
    resultado_colorido = cv2.bitwise_and(img, img, mask=mascara_pele)

    # Criar a imagem em preto e branco
    resultado_binario = cv2.cvtColor(resultado_colorido, cv2.COLOR_BGR2GRAY)
    img_binario = basic_threshold(resultado_binario, 1)

    return resultado_colorido, img_binario


if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows

    img = imread('res/unexistent-person-1.jpg') 

    img_skin_color, img_skin_binary = skin_color_thresholding(img=img)

    imshow('Imagem com Pele Destacada', img_skin_color)
    imshow('Imagem Preto e Branco', img_skin_binary)
    waitKey(0)
    destroyAllWindows()