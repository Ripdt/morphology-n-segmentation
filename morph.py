import numpy as np

def create_element(size_x, size_y):
   element = np.zeros((size_x, size_y), dtype = np.uint8)
   for i in range(size_x):
      for j in range(size_y):
         element[i,j] = 255
   return element

def morf_dilate_erode(img, kernel, padding=True, dilate=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape
    
    # Get dimensions of the image
    img_height, img_width = img.shape
    
    # Calculate padding if necessary
    pad_height = k_height // 2
    pad_width = k_width // 2
    
    # Create a padded version of the image to handle edges
    if padding:
        padded_img = add_padding(img, pad_height, pad_width)
    else:
        padded_img = img

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=np.uint8)

    # Perform dilation or erosion
    for i_img in range(img_height):
        for j_img in range(img_width):
            match_element = 0 if dilate else 1

            # Apply the kernel
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    img_i = i_img + i_kernel - pad_height
                    img_j = j_img + j_kernel - pad_width

                    # Check if within bounds for images without padding
                    if not padding:
                        if img_i < 0 or img_i >= img_height or img_j < 0 or img_j >= img_width:
                            continue
                    
                    # Determine match for dilation and erosion
                    if kernel[i_kernel, j_kernel] == 255:
                        if dilate:
                            if padded_img[img_i, img_j] == 255:
                                match_element = 1
                                break  # Exit early for dilation
                        else:  # Erosion
                            if padded_img[img_i, img_j] != 255:
                                match_element = 0
                                break  # Exit early for erosion
                if (dilate and match_element == 1) or (not dilate and match_element == 0):
                    break
            
            # Set the output pixel
            output[i_img, j_img] = 255 if match_element == 1 else 0

    return output

def add_padding(img, pad_height, pad_width):
    return np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)


def morph(img : np.ndarray, padding : bool = False, dilate : bool = True, kernel_width : int = 3, kernel_height : int = 3):
    kernel = create_element(kernel_width, kernel_height)
    return morf_dilate_erode(img=img, kernel=kernel, padding=padding, dilate=dilate)