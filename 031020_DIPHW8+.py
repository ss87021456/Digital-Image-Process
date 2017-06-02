from cmath import exp, pi
from math import log, ceil
import numpy,cv2,sys,os

def fft(f):     # fft for list type
    N = len(f)
    if N <= 1: return f
    even = fft(f[0::2])
    odd =  fft(f[1::2])
    return [even[k] + exp(-2j*pi*k/N)*odd[k] for k in xrange(N/2)] + \
           [even[k] - exp(-2j*pi*k/N)*odd[k] for k in xrange(N/2)]

def pad2(f):    # 2D zero padding for list type
    m, n = len(f), len(f[0])
    M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
    F = [ [0]*N for _ in xrange(M) ]
    for i in range(0, m):
        for j in range(0, n):
            F[i][j] = f[i][j]
    return F, M, N

def FFT2D(image):
    width , height = image.shape
    image = image.tolist()
    image , M , N = pad2(image)     # zero padding for FFT
    image_temp = image              # initial a temp_2Darray
    
    ########## execute row FFT ###########
    for i in range(M):              
        image_temp[i] = fft(image[i])
    
    image_temp = numpy.asarray(image_temp,dtype = complex) # transfer list to numpy.arr
    image_temp = numpy.transpose(image_temp)               # transpose for further work
    image = image_temp.tolist()                            # transfer numpy.arr to list
    
    ########## execute column FFT ###########
    for i in range(N): 
        image_temp[i] = fft(image[i])
    
    image_temp = numpy.transpose(image_temp)    # transpose for correct position
    
    ########## transfer mag to log(1+mag) and normalize ###########
    magnitude = cv2.sqrt(image_temp.real**2.0 + image_temp.imag**2.0)
    log_spectrum = cv2.log(1.0 + magnitude)
    cv2.normalize(log_spectrum, log_spectrum, 0.0, 255.0, cv2.NORM_MINMAX)

    ########## resize the Frequency padding image back to original size ###########
    res = cv2.resize(log_spectrum,(height, width), interpolation = cv2.INTER_CUBIC)
    fshift = numpy.fft.fftshift(res)   # exchange position of Frequency image
    return fshift


def main():
    dirname = "output"                      
    if not os.path.isdir("./output"):       # check whether exist output dir
        os.mkdir(dirname)

    for i in range(1,5):
        path = "./input/Q" + str(i) + ".tif"            # load input image from input dir
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = FFT2D(image)
        file_name = "Q" + str(i) + " result"
        cv2.imwrite(os.path.join(dirname, file_name+".tif"), result) # save result image to output dir
        cv2.normalize(result, result, 0.0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow(file_name, result)
    cv2.waitKey(0)

if __name__ == "__main__" :
    main()