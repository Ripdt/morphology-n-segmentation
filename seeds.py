import numpy as np
import cv2
import sys

def apply_seeds(img, num_superpixels=30, num_iterations=30):
    seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], num_superpixels, num_iterations)
    
    # Converte a imagem para o formato correto (BGR para LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # Inicia a segmentação
    seeds.iterate(img_lab)

    # Obtém os rótulos e a imagem de superpixels
    labels = seeds.getLabels()
    mask_slic = seeds.getLabelContourMask(False)

    # Cria uma imagem colorida onde os segmentos são mostrados com contornos
    output = img.copy()
    output[mask_slic == 255] = [0, 0, 255]  # Contorno

    return labels, output

if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows

    img = imread('res/unexistent-person-1.jpg') 
    labels, img_seeds = apply_seeds(img=img)

    imshow('seeds', img_seeds)
    
    waitKey(0)
    destroyAllWindows()
