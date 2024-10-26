import numpy as np

def basic_threshold(img: np.ndarray, threshold_value: int) -> np.ndarray:
    img_out = np.zeros_like(img)
    height, width = img.shape

    for i in range(height):
        for j in range(width):
            if img[i, j] >= threshold_value:
                img_out[i, j] = 255
            else:
                img_out[i, j] = 0

    return img_out

def _compute_otsu_criteria(im, th):
    # Create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # Compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # If one of the classes is empty, that threshold is not considered
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # Find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # Compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

def otsu_threshold(img: np.ndarray) -> np.ndarray:
    # Convert max value to integer for range
    threshold_range = range(int(np.max(img)) + 1)
    criterias = np.array([_compute_otsu_criteria(img, th) for th in threshold_range])

    # Best threshold minimizes the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]

    binary = img.copy()
    binary[binary > best_threshold] = 255
    binary[binary <= best_threshold] = 0

    return binary

if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE

    img = imread('res/sfinge-fingerprint-1.jpg', IMREAD_GRAYSCALE)

    img_basic_threshold = basic_threshold(img=img, threshold_value=100)
    img_otsu_threshold = otsu_threshold(img=img)

    imshow('basic', img_basic_threshold)
    waitKey(0)
    destroyAllWindows()
    
    imshow('otsu', img_otsu_threshold)
    waitKey(0)
    destroyAllWindows()
