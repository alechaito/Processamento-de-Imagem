import fileinput
import time
import random
import numpy as np
import scipy.spatial.distance as distance


def main():
    # Load images
    #img = Image(w=481, h=321, path='star.ppm')
    img = Image(w=481, h=321, path='flower.ppm')
    #img_3 = Image(w=800, h=361, path='maca.ppm')
    # Perfoming
    img.mahalanobis()
    

class Pixel():
    def __init__(self, r=255, g=255, b=255):
        self.r = r
        self.g = g
        self.b = b
    

    
class Image():
    def __init__(self, w=None, h=None, path=None):
        self.w = w
        self.h = h
        self.path = path
        self.pixels = self.map_pixels()
    
    def load(self):
        with open(self.path) as file:
            array = file.readlines()
        array.remove(array[0])
        array.remove(array[0])
        array.remove(array[0])
        for i in range(0, len(array)):
            array[i] = array[i].replace('\n', '')
        return array

    def compose(self):
        file = open('new-'+self.path, 'w')
        #add header img
        file.write("P3\n")
        file.write(str(self.w)+" "+str(self.h)+"\n")
        file.write("255\n")
        #adddata
        for pixel in self.pixels:
            file.write(str(pixel.r)+"\n")
            file.write(str(pixel.g)+"\n")
            file.write(str(pixel.b)+"\n")
        file.close()

    def map_pixels(self):
        FLAG = 0
        IMG = self.load()
        pixels = []
        for i in range(0, self.h):
            for j in range(0, self.w):
                r = IMG[i*self.w*3 + j*3];
                g = IMG[i*self.w*3 + j*3+1];
                b = IMG[i*self.w*3 + j*3+2];
                _PIXEL = Pixel(int(r), int(g), int(b))
                pixels.append(_PIXEL)
        return pixels


    def mahalanobis(self):
        R = np.array([153, 77, 229, 189])
        G = np.array([23, 13, 151, 36])
        B = np.array([20, 1, 0, 28])
        # Centro da nuvem
        CENTER = [193, 118, 0]
        _COV = np.cov([R, G, B])
        #print(_COV)
        _INV = np.linalg.inv(_COV)
        for pixel in self.pixels:
            ds = distance.mahalanobis(CENTER, [pixel.r, pixel.g, pixel.b], _INV)
            if(ds > 19.2):
                print("maior:%s"% ds)
                pixel.r = 0
                pixel.g = 0
                pixel.b = 0
        self.compose()

    def sphere(self, center, threshold):
        img = self.pixels
        center = Pixel(center[0], center[1], center[2])
        for pixel in img:
            if(self.dist(center, pixel) < threshold):
                pixel.r = 0
                pixel.g = 0
                pixel.b = 0  
        self.compose()

    def cube(self):
        for pixel in self.pixels:
            STT1 = pixel.r > 60 and pixel.r < 200
            STT2 = pixel.g > 0 and pixel.g < 60
            STT3 = pixel.b > 0 and pixel.b < 60
            if(STT1 and STT2 and STT3):
                pixel.r = 0
                pixel.g = 0
                pixel.b = 0
        self.compose()

    
    def neighbors(self, threshold):
        img = self.pixels
        # Center of sphere 1
        C1 = Pixel(138, 5, 6)
        # Center of sphere 2
        C2 = Pixel(176, 20, 24)
        # Center of sphere 3
        C3 = Pixel(190, 46, 95)
        for pixel in img:
            dist_1 = self.dist(C1, pixel)
            dist_2 = self.dist(C2, pixel)
            dist_3 = self.dist(C3, pixel)
            # Calculate the min distance between the point and the spheres
            smallest = min([dist_1, dist_2, dist_3])
            if(smallest < threshold):
                pixel.r = 0
                pixel.g = 0
                pixel.b = 0
        self.compose()

    def dist(self, center, pixel):
        dist = np.sqrt(
            (center.r - pixel.r)**2 + 
            (center.g - pixel.g)**2 + 
            (center.b - pixel.b)**2
        )
        return dist

main()