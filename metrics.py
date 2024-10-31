import numpy as np

def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    err = np.sum((original.astype(float) - compressed.astype(float)) ** 2)
    err /= float(original.shape[0] * original.shape[1])
    return err

def calculate_rmse(original: np.ndarray, compressed: np.ndarray) -> float:
    err = calculate_mse(original, compressed)
    return np.sqrt(err)

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    mse = calculate_mse(original, compressed)
    if mse == 0: # imagens iguais
        return float('inf') # PSNR infinito
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)