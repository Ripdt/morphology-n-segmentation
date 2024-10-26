import numpy as np

def add_padding(img, padding_height, padding_width):
    n, m = img.shape
    
    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height : n + padding_height, padding_width : m + padding_width] = img
    
    return padded_img

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
    
    # Calculate padding required
    pad_height = k_height // 2
    pad_width = k_width // 2

    
    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)
    else:
        padded_img = img

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  

    if dilate == True:
      for i_img in range(img_height):
         for j_img in range(img_width):
               match_element = 0
               for i_kernel in range(k_height):
                  for j_kernel in range(k_width):
                           if (kernel[i_kernel, j_kernel] == 255):
                              if(padded_img[i_img+i_kernel, j_img+j_kernel] == kernel[i_kernel, j_kernel]):  
                                 match_element = 1
               if(match_element == 1):
                  output[i_img, j_img] = 255
               else:
                  output[i_img, j_img] = 0
    else:
      for i_img in range(img_height):
         for j_img in range(img_width):
               match_element = 1
               for i_kernel in range(k_height):
                  for j_kernel in range(k_width):
                           if (kernel[i_kernel, j_kernel] == 255):
                              if(padded_img[i_img+i_kernel, j_img+j_kernel] != kernel[i_kernel, j_kernel]):  # Analisa se não há algum hit entre elemento e janela
                                 match_element = 0
               if(match_element == 0):
                  output[i_img, j_img] = 0
               else:
                  output[i_img, j_img] = 255
    return output

def morph(img, padding = True, dilate = True):
    kernel = create_element(3, 3)
    return morf_dilate_erode(img=img, kernel=kernel, padding=padding, dilate=dilate)