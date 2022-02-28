from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
import numpy as np
import cv2


def blurFilter(img, filter_sz=None, blur_weights=None):
    if blur_weights is None:
        blur_weights = np.ones(filter_sz)/filter_sz
    
    img = convolve1d(img, weights=blur_weights, axis=0)
    img = convolve1d(img, weights=blur_weights, axis=1)
    return img


BLUR3 = np.ones(3) / 3
BLUR15 = np.ones(15) / 15
BLUR30 = np.ones(30) / 30

SOBEL_KERNELX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_KERNELY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
LAPLACIAN_KERNEL = np.array([[0,1,0],[1,-4,1],[0,1,0]])

SOBEL_KERNEL_0 = np.array([1, 2, 1])
SOBEL_KERNEL_1 = np.array([-1, 0, 1])



def sobelFilter(img, normalize = True):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    border_X = convolve2d(img, SOBEL_KERNELX,
                          mode='same', boundary='fill')
    border_Y = convolve2d(img, SOBEL_KERNELY,
                          mode='same', boundary='fill')

    convolved_img = border_X + border_Y

    if normalize:
        convolved_img = ((convolved_img - np.mean(convolved_img)) / (np.var(convolved_img)))
        return (255*convolved_img).astype('uint8')

    else:
        return convolved_img.astype('uint8')


def laplaceFilter(img, normalize = True):

    #convierte la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #aplica el filtro del kernel de laplace
    convolved_img = convolve2d(gray, LAPLACIAN_KERNEL,
                               mode='same', boundary='fill')

    if normalize:
        convolved_img = (convolved_img -np.mean(convolved_img))/(np.var(convolved_img))
        return (255*convolved_img).astype('uint8')

    else:
        return convolved_img.astype('uint8')



def equalize_hsv_channel(img, channel=2):

    n, m, _ = np.shape(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    sat = img[:, :, channel]
    unique, counts = np.unique(sat, return_counts=True)
    mapping = np.zeros(256)

    mapping[unique] = counts
    mapping = 255/(m*n)*np.cumsum(mapping)
    img[:, :, channel] = mapping[sat]

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def sharpness_sobel(img):
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    borders = sobelFilter(img, normalize=False)
    
    img_hsv[:,:,2] = img_hsv[:,:,2] - borders
    print(np.max(img_hsv[:,:,2]))    
    print(np.min(img_hsv[:,:,2]))
    
    output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return output.astype('uint8')


def sharpness_laplace(img):
    
    #obtiene los bordes de la imagen usando 
    #el kernel de laplace
    borders = laplaceFilter(img, False)
    
    #convierte la imagen a HSV para quedarse con 
    #'value' de la matriz y aumentarle el sharpness
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:,:,2] = img_hsv[:,:,2] - borders
    
    #convierte de regreso a RGB
    output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return output.astype('uint8')




