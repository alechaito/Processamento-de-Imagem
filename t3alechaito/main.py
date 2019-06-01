import fileinput
import time
import random
import numpy as np
import scipy.spatial.distance as distance
import sys

k1 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
k2= [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
k3 = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
k4 = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
k5 = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
k6 = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
k7 = [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
k8 = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]

def main():
    img = Image(path=sys.argv[1], pgm=True) 
    img.robinson(3)


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
        self.path = path
        self.pgm = pgm
        self.pixels = self.map_pixels()
    
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
    
    def load_points(self, HOT):
        p0 = self.pixels[(HOT.x-1) + (HOT.y-1)*self.w]
        p1 = self.pixels[(HOT.x-1) + (HOT.y)*self.w]
        p2 = self.pixels[(HOT.x-1) + (HOT.y+1)*self.w]
        p3 = self.pixels[HOT.x + (HOT.y-1)*self.w]
        p5 = self.pixels[HOT.x + (HOT.y+1)*self.w]
        p6 = self.pixels[(HOT.x+1) + (HOT.y-1)*self.w]
        p7 = self.pixels[(HOT.x+1) + (HOT.y)*self.w]
        p8 = self.pixels[(HOT.x+1) + (HOT.y+1)*self.w]
        return [p0, p1, p2, p3, HOT, p5, p6, p7, p8]


    def sobel(self, kernel, size):
        buffer = np.copy(np.array(self.pixels))
        for i in range(1, self.h-1):
            for j in range(1, self.w-1):
                HOT = self.pixels[i*self.w+j]
                points = self.load_points(HOT)
                k_total = 0
                R = 0
                G = 0
                B = 0
                RGB = []
                for x in range(0, size):
                    for y in range(0, size):
                        # Calculando o total da matriz de kernel
                        k_total += kernel[x][y]
                        # Somatoria RGB para cada pixel do kernel e foto
                        R += points[x+y].r*kernel[x][y]
                        G += points[x+y].g*kernel[x][y]
                        B += points[x+y].b*kernel[x][y]
                # Dividindo pelo total da matriz de kernel      
                R = int(abs(R)) 
                G = int(abs(G)) 
                B = int(abs(B)) 
                # Removendo outliers
                #print("[+] hot pixel/ r:%s, g:%s, b:%s, x:%s, y:%s \n"% (new.r, new.g, new.b, new.x, new.y))
                buffer[i*self.w+j] = Pixel(R, G, B, j, i)
        # Sobrescrevendo antigos pixels para os com o filtro aplicado
        self.pixels = buffer
    
    def roberts(self, size):
        x = self.sobel2(_K1, 2)
        y = self.sobel2(_K2, 2)
        result = []
        for i in range(0, len(x)):
            new = Pixel()
            new.r = np.sqrt(x[i].r**2 + y[i].r**2)
            new.g = np.sqrt(x[i].g**2 + y[i].g**2)
            new.b = np.sqrt(x[i].b**2 + y[i].b**2)
            result.append(new)
            #print("[+] R:%s,G:%s,B:%s \n", new.r, new.g, new.b)
        self.pixels = result
    
    def normalize(self, dx, dy):
        bufferx = []
        buffery = []

        for i in range(0, len(dx)-1):
            bufferx.append(dx[i].r)
            bufferx.append(dx[i].g)
            bufferx.append(dx[i].b)
        
        for i in range(0, len(dx)-1):
            buffery.append(dy[i].r)
            buffery.append(dy[i].g)
            buffery.append(dy[i].b)

        minnx = min(bufferx)
        maxxx = max(bufferx)
        minny = min(buffery)
        maxxy = max(buffery)
    
        for i in range(0, len(dx)-1):
            dx[i].r = (dx[i].r-minnx) / (maxxx-minnx) * 255;
            dx[i].g = (dx[i].g-minnx) / (maxxx-minnx) * 255;
            dx[i].b = (dx[i].b-minnx) / (maxxx-minnx) * 255;

            dy[i].r = (dy[i].r-minny) / (maxxy-minny) * 255;
            dy[i].g = (dy[i].g-minny) / (maxxy-minny) * 255;
            dy[i].b = (dy[i].b-minny) / (maxxy-minny) * 255;

        for i in range(0, len(dx)-1):
            pdx = dx[i]
            pdy = dy[i]
            self.pixels[i].r =  np.sqrt(pdx.r**2+pdy.r**2)
            self.pixels[i].g =  np.sqrt(pdx.g**2+pdy.g**2)
            self.pixels[i].b =  np.sqrt(pdx.b**2+pdy.b**2)

    def robinson(self, size):
        print("[+] Calculando convulucoes...")
        m1 = self.sobel2(k1, size)
        m2 = self.sobel2(k2, size)
        m3 = self.sobel2(k3, size)
        m4 = self.sobel2(k4, size)
        m5 = self.sobel2(k5, size)
        m6 = self.sobel2(k6, size)
        m7 = self.sobel2(k7, size)
        m8 = self.sobel2(k8, size)

        buffer = np.copy(m1)
        for i in range(0, len(m1)-1):
            buffer[i].r = int(np.sqrt(
                m1[i].r**2 +
                m2[i].r**2 +
                m3[i].r**2 +
                m4[i].r**2 +
                m5[i].r**2 +
                m6[i].r**2 +
                m7[i].r**2 +
                m8[i].r**2
            ))
            buffer[i].g = buffer[i].r
            buffer[i].b = buffer[i].r

        normalize = []
        for i in range(0, len(buffer)-1):
            normalize.append(buffer[i].r)
            normalize.append(buffer[i].g)
            normalize.append(buffer[i].b)

        minn = min(normalize)
        maxx = max(normalize)
    
        for i in range(0, len(buffer)-1):
            buffer[i].r = (buffer[i].r-minn) / (maxx-minn) * 255;
            buffer[i].g = (buffer[i].g-minn) / (maxx-minn) * 255;
            buffer[i].b = (buffer[i].b-minn) / (maxx-minn) * 255;

        self.pixels = buffer
        self.compose()
    
    def sobel2(self, kernel, size):
        buffer = np.copy(np.array(self.pixels))
        for i in range(1, self.h-1):
            for j in range(1, self.w-1):
                HOT = self.pixels[i*self.w+j]
                points = self.load_points(HOT)
                k_total = 0
                R = 0
                G = 0
                B = 0
                RGB = []
                for x in range(0, size):
                    for y in range(0, size):
                        # Calculando o total da matriz de kernel
                        k_total += kernel[x][y]
                        # Somatoria RGB para cada pixel do kernel e foto
                        R += points[x+y].r*kernel[x][y]
                        G += points[x+y].g*kernel[x][y]
                        B += points[x+y].b*kernel[x][y]
                # Dividindo pelo total da matriz de kernel      
                R = R
                G = G
                B = B  
                #print("[+] hot pixel/ r:%s, g:%s, b:%s, x:%s, y:%s \n"% (new.r, new.g, new.b, new.x, new.y))
                buffer[i*self.w+j] = Pixel(R, G, B, j, i)
        # Sobrescrevendo antigos pixels para os com o filtro aplicado
        return buffer

    def blur(self, kernel, size):
        buffer = np.copy(np.array(self.pixels))
        for i in range(1, self.h-1):
            for j in range(1, self.w-1):
                HOT = self.pixels[i*self.w+j]
                points = self.load_points(HOT)
                k_total = 0
                R = 0
                G = 0
                B = 0
                RGB = []
                for x in range(0, size):
                    for y in range(0, size):
                        # Calculando o total da matriz de kernel
                        k_total += kernel[x][y]
                        # Somatoria RGB para cada pixel do kernel e foto
                        R += points[x+y].r*kernel[x][y]
                        G += points[x+y].g*kernel[x][y]
                        B += points[x+y].b*kernel[x][y]
                # Dividindo pelo total da matriz de kernel      
                R = int(R/k_total) 
                G = int(G/k_total)
                B = int(B/k_total)
                # Removendo outliers
                if(R > 255):
                    R = 255
                if(G > 255):
                    G = 255
                if(B > 255):
                    B = 255   
                buffer[i*self.w+j] = Pixel(R, G, B, j, i)
        # Sobrescrevendo antigos pixels para os com o filtro aplicado
        self.pixels = buffer
    
    def classify(self):
        pixels = self.pixels
        ref = np.empty([self.w, self.h])
        _stack = []
        # Defino um pixel inicial
        focus = Pixel(r=0, g=0, b=0, x=386, y=48)
        focus = pixels[self.idxvec(focus)]

        for i in range(0, 5000):
            # Peco para andar e retornar o pixel vizinho caso True
            direc = self.walk(focus, ref)
            if(direc == 0):
                size = len(_stack)
                if(size >= 2):
                    print("Desempilhei...")
                    print("Focus antigo.. \n")
                    self.printer(focus)
                    _stack.pop()
                    focus = _stack[-1]
                    print("Focus novo.. \n")
                    self.printer(focus)
                else:
                    print("acabou ja nao tenho onde ir")
                    break
            else:
                # Atualizo no mapa aonde já foi percorrido
                ref[focus.x][focus.y] = 1
                # Coloco o ultimo focus na pilha
                _stack.append(focus)
                print("Empilhei...\n")
                # Atualizo o focus com o pixel vizinho
                print("Focus TRUE.. \n")
                self.printer(focus)
                focus = direc
                print("Focus TRUE.. \n")
                self.printer(focus)

        self.test(ref)
        
        ##VOU QUERER COMECAR ANDAR
        # 1 - Verifico se da pra ir pra N

    def walk(self, focus, ref):
        # Return true if can wal
        vec = {
            'n': False,
            'w': False,
            's': False,
            'e': False,
        }
        # N   
        if(focus.y > 0):
            # Instanciando candidato
            N = Pixel(focus.r, focus.g, focus.b, focus.x, focus.y)
            N.y = N.y-1
            N = self.pixels[self.idxvec(N)]
            if not(ref[focus.x][focus.y] == 1):
                if(self.dist(focus, N) < 25):
                    print("dist:%s"% self.dist(focus, N))
                    print("[N] Foco|x:%s,y:%s||Vizinho:x:%s,y:%s \n"% (focus.x, focus.y, N.x, N.y))
                    vec['n'] = True
                    return N
        # W
        if(focus.x > 0):
            # Instanciando candidato
            W = Pixel(focus.r, focus.g, focus.b, focus.x, focus.y)
            W.x = W.x-1
            W = self.pixels[self.idxvec(W)]
            if not(ref[focus.x][focus.y] == 1):
                if(self.dist(focus, W) < 25):
                    print("dist:%s"% self.dist(focus, W))
                    print("[W] Foco|x:%s,y:%s||Vizinho:x:%s,y:%s \n"% (focus.x, focus.y, W.x, W.y))
                    vec['w'] = True
                    return W
        # S
        if(focus.y+1 <= self.h-1):
            # Instanciando candidato
            S = Pixel(focus.r, focus.g, focus.b, focus.x, focus.y)
            S.y = S.y+1
            S = self.pixels[self.idxvec(S)]
            if not(ref[focus.x][focus.y] == 1):
                if(self.dist(focus, S) < 25):
                    print("dist:%s"% self.dist(focus, S))
                    print("[S] Foco|x:%s,y:%s||Vizinho:x:%s,y:%s \n"% (focus.x, focus.y, S.x, S.y))
                    vec['S'] = True
                    return S
        # E
        if(focus.x+1 <= self.w-1):
            # Instanciando candidato
            E = Pixel(focus.r, focus.g, focus.b, focus.x, focus.y)
            E.x = E.x+1
            print("1111")  
            self.printer(E)
            E = self.pixels[self.idxvec(E)]
            if not(ref[focus.x][focus.y] == 1):
                if(self.dist(focus, E) < 25):
                    print("dist:%s"% self.dist(focus, E))
                    print("222")
                    self.printer(E)
                    vec['E'] = True
                    return E
        return 0

    def dist(self, p1, p2):
        #self.printer(p2)
        #self.printer(p1)
        r = (p1.r-p2.r)**2
        g = (p1.g-p2.g)**2
        b = (p1.b-p2.b)**2
        return np.sqrt(r+g+b)

    def printer(self, pixel):
        print("[OUTPUT] Pixel|r:%s, g:%s, b:%s, x:%s, y:%s| \n"% (pixel.r, pixel.g, pixel.b, pixel.x, pixel.y))
                    


    def test(self, ref):
        for y in range(0, self.h):
            for x in range(0, self.w):
                if(ref[x][y] == 1):
                    self.pixels[y*self.w + x].r = 255
                    self.pixels[y*self.w + x].g = 255
                    self.pixels[y*self.w + x].b = 255

    def idxvec(self, pixel):
        return pixel.y*self.w + pixel.x

main()