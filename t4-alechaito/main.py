import fileinput
import time
import random
import numpy as np
import scipy.spatial.distance as distance
import sys
import copy
import random as rd
import cv2

k1 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
k2= [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
k3 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
k4 = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
k5 = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
k6 = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
k7 = [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
k8 = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]

def main():
    img = Image(path=sys.argv[1], pgm=False) 
    img.partition()
    img.compose()


class Pixel():
    def __init__(self, r=255, g=255, b=255, x=0, y=0):
        self.r = r
        self.g = g
        self.b = b
        self.x = x
        self.y = y
    
    
class Image():
    def __init__(self, w=None, h=None, path=None, pgm=False):
        self.w = w
        self.h = h
        if(pgm == True):
            self.path = "dataset/pgm/"+path
        else:
            self.path = "dataset/ppm/"+path
        self.pgm = pgm
        self.pixels = self.map_pixels()
        self.thresh = int(sys.argv[2])
        self.regions = 0
    
    def load(self):
        with open(self.path) as file:
            array = file.readlines()
        array.remove(array[0])
        size = array[0].split(' ')
        self.w = int(size[0])
        self.h = int(size[1])
        array.remove(array[0])
        array.remove(array[0])
        for i in range(0, len(array)):
            array[i] = array[i].replace('\n', '')
        return array

    def compose(self):
        name = np.random.randint(999)
        spr = self.path.split(".")
        self.path = spr[0]+str(name)+".ppm"
        print(self.path)
        file = open(self.path, 'w')
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
        IMG = self.load()
        pixels = []
        for i in range(0, self.h):
            for j in range(0, self.w):
                if(self.pgm == True):
                    r = IMG[i*self.w + j];
                    g = IMG[i*self.w + j];
                    b = IMG[i*self.w + j];
                else:
                    r = IMG[i*self.w*3 + j*3];
                    g = IMG[i*self.w*3 + j*3+1];
                    b = IMG[i*self.w*3 + j*3+2];
                _PIXEL = Pixel(int(r), int(g), int(b), j, i)
                pixels.append(_PIXEL)
        return pixels
    
    def partition(self):
        k = 1
        ref = [0]*(self.w*self.h)
        for i in range(0, self.h):
            for j in range(0, self.w):
                if(ref[i*self.w+j] == 0):
                    self.classify(j, i, k, ref)
                    k = k +1
        self.make(ref)

    def classify(self, x, y, index, ref):
        pixels = copy.copy(self.pixels)
        _stack = []
        # Defino um pixel inicial
        focus = pixels[y*self.w+x]
        ref[y*self.w+x] = index
        _stack.append(focus)
        self.regions += 1

        while(len(_stack) > 0):
            direc = self.walk(focus, ref, _stack)
            if(direc == 0):
                focus = _stack[-1]
                _stack.pop()
            else:
                ref[direc.y*self.w+direc.x] = index
                focus = direc
                _stack.append(focus)
        

    def walk(self, focus, ref, _stack):
        # Return true if can wal
        vec = {
            'n': False,
            'w': False,
            's': False,
            'e': False,
        }
        # N   
        idx_n = (focus.y-1)*self.w+focus.x
        idx_w = focus.y*self.w+(focus.x-1)
        idx_s = (focus.y+1)*self.w+focus.x
        idx_e = focus.y*self.w+(focus.x+1)

        '''if(len(_stack) > 5):
            sumr = 0 
            sumg = 0
            sumb = 0
            for i in range(1, 5):
                #print(i)
                sumr += _stack[len(_stack)-i].r
                sumg += _stack[len(_stack)-i].g
                sumb += _stack[len(_stack)-i].b
            focus = Pixel(sumr/5, sumg/5, sumr/5, focus.x, focus.y)'''

        if(focus.y > 0):
            N = copy.copy(self.pixels[idx_n])
            if (ref[N.y*self.w+N.x] == 0):
                if(self.dist(focus, N) < self.thresh):
                    vec['n'] = True
                    return N
        # W
        if(focus.x > 0):
            # Instanciando candidato
            W = copy.copy(self.pixels[idx_w])
            if (ref[W.y*self.w+W.x] == 0):
                if(self.dist(focus, W) < self.thresh):
                    vec['w'] = True
                    return W
        # S
        if(focus.y+1 <= self.h-1):
            # Instanciando candidato
            S = copy.copy(self.pixels[idx_s])
            if (ref[S.y*self.w+S.x] == 0):
                if(self.dist(focus, S) < self.thresh):
                    vec['S'] = True
                    return S
        # E
        if(focus.x+1 <= self.w-1):
            # Instanciando candidato
            E = copy.copy(self.pixels[idx_e])
            if (ref[E.y*self.w+E.x] == 0):
                if(self.dist(focus, E) < self.thresh):
                    vec['E'] = True
                    return E
        return 0

    def dist(self, p1, p2):
        #self.printer(p1)
        #self.printer(p2)
        x1 = [p1.r, p1.g, p1.b]
        x2 = [p2.r, p2.g, p2.b]
        x3 = np.array([x1, x2])
        x3 = np.var(x3, axis=0)
        return x3[0]+x3[1]+x3[2]

    def printer(self, pixel):
        print("[OUTPUT] Pixel|r:%s, g:%s, b:%s, x:%s, y:%s| \n"% (pixel.r, pixel.g, pixel.b, pixel.x, pixel.y))
                    

    def make(self, ref):
        print("Total de regioes:%s"% self.regions)
        r = np.random.randint(255, size=self.regions)
        g = np.random.randint(255, size=self.regions)
        b = np.random.randint(255, size=self.regions)
        colors = [r,g,b]
        for y in range(0, self.h):
            for x in range(0, self.w):
                if(ref[y*self.w+x] > 0):
                    self.pixels[y*self.w + x].r = colors[0][ref[y*self.w+x]-1]
                    self.pixels[y*self.w + x].g = colors[1][ref[y*self.w+x]-1]
                    self.pixels[y*self.w + x].b = colors[2][ref[y*self.w+x]-1]

main()