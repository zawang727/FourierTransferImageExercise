import scipy.fftpack as fft
import matplotlib.pyplot as plt
import numpy
import imageio
import math
from pylab import *
from numpy.fft import fft2, ifft2, fftshift, ifftshift

#Reference: https://lab4sys.com/en/fourier-transform-of-an-image/?cn-reloaded=1

#N =256
#t = np.arange(N)

#m=4
#nu = float(m)/N
#f= np.sin(2*np.pi*nu*t)
#ft=np.fft.fft(f)
#freq=np.fft.fftfreq(N)
#plt.plot(freq,ft.real**2+ft.imag**2)
#plt.show()

def matriceImage(matrice,gamma,rgb):
    s= matrice.shape
    a=1.0/gamma;
    norm=matrice.max()
    m=numpy.power(matrice/norm,a) #normalization, gamma correction
    im=numpy.zeros((s[0],s[1],3),dtype=float64) #declare image matrix
    im[:,:,0] = rgb[0]*m
    im[:,:,1] = rgb[1]*m
    im[:,:,2] = rgb[2]*m
    return im
    
def matriceImageLog(matrice,rgb):
    s= matrice.shape
    m = numpy.log10(1+matrice)
    min = m.min()
    max = m.max()
    m = (m-min)/(max-min)
    im=numpy.zeros((s[0],s[1],3),dtype=float64) #declare image matrix
    im[:,:,0] = rgb[0]*m
    im[:,:,1] = rgb[1]*m
    im[:,:,2] = rgb[2]*m
    return im
    
def plotSpectre(image,Lx,Ly):
    (Ny,Nx,p) = image.shape
    fxm = Nx*1.0/(2*Lx)
    fym = Ny*1.0/(2*Ly)
    imshow(image,extent=[-fxm,fxm,-fym,fym])
    xlabel("fx")
    ylabel("fy")
    
def low_pass_transfert(fx,fy,p): #low pass filtering
    fc=p[0]
    return 1.0/numpy.power(1+(fx*fx+fy*fy)/(fc*fc),2)

def high_pass_transfert(fx,fy,p): #high pass filtering
    fc=p[0]
    return 1.0-1.0/numpy.power(1+(fx*fx+fy*fy)/(fc*fc),2)

def Gaussian_notch_transfert(fx,fy,p): #Gaussian notch filter
    f0 = p[0]
    sigma = p[1]
    ux = numpy.absolute(fx)-f0
    return 1.0-numpy.exp(-(ux*ux)/(sigma*sigma))
    
def matriceFiltre(matrice,transfert,p):
    s = matrice.shape
    Nx = s[1]
    Ny = s[0]
    nx = Nx/2
    ny = Ny/2
    Mat = zeros((Ny,Nx),dtype=numpy.complex)
    for n in range(Nx):
        for l in range(Ny):
            fx = float(n-nx-1)/Nx
            fy = float(l-ny-1)/Ny
            Mat[l,n] = matrice[l,n]*transfert(fx,fy,p)
    return Mat
    

    
print('Hi')
imA = imageio.imread("imageA.png")
U=numpy.array(imA[:,:,0],dtype=numpy.float64) #remove red layer
V=fft2(U)
VC = fftshift(V)
P = numpy.power(numpy.absolute(VC),2)
img = matriceImage(P,2.0,[1.0,0.0,0.0])
#plotSpectre(img,200.0,100.0)
img = matriceImageLog(P,[1.0,0.0,0.0])
#plotSpectre(img,200.0,100.0)
H = numpy.ones(VC.shape,dtype=numpy.complex)
H = matriceFiltre(H,low_pass_transfert,[0.1])
VCF = VC*H
PF = numpy.power(numpy.absolute(VCF),2)
imPF = matriceImageLog(PF,[1.0,0,0])
plotSpectre(imPF,200,100)
VF = ifftshift(VCF)
UF=ifft2(VF)
imageF = matriceImage(UF,1.0,[1.0,1.0,1.0])
figure()
imshow(imageF)

