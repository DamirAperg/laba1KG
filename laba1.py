import numpy as np
from PIL import Image  
import math
class ImageClass:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.img = np.zeros((H,W,3), dtype=np.uint8)
    def dot_line(self, x0, y0, x1, y1, count, color):
        step = 1.0/count
        for t in np.arange(0, 1, step):
            x = round((1 - t)*x0 + t*x1)
            y = round((1 - t)*y0 + t*y1)
            self.img[y, x] = color   

    def dot_line_fix(self, x0, y0, x1, y1, color):
        count = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        step = 1.0/count
        for t in np.arange(0, 1, step):
            x = round((1 - t)*x0 + t*x1)
            y = round((1 - t)*y0 + t*y1)
            self.img[y, x] = color 

    def dot_line_2new(self, x0, y0, x1, y1, color):
        for x in range (x0, x1):
            t = (x - x0)/(x1 - x0)
            y = round((1.0 - t)*y0 + t*y1)
            self.img[y, x] = color            

    def dot_line_2fix1(self, x0, y0, x1, y1, color):
        if (x0 > x1):
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in range (x0, x1):
            t = (x - x0)/(x1 - x0)
            y = round((1.0 - t)*y0 + t*y1)
            self.img[y, x] = color 

    def dot_line2(self, x0, y0, x1, y1, color):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        xChange = False
        if(abs(x0 - x1) < abs(y0 - y1)):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xChange = True
        if(x0 > x1):
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in range(x0, x1):
            t = (x - x0)/(x1 - x0)
            y = round((1 - t)*y0 + t*y1)
            if (xChange):
                self.img[x, y] = color
            else:
                self.img[y,x] = color

    def dot_line3(self, x0, y0, x1, y1, color):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        xChange = False
        if(abs(x0 - x1) < abs(y0 - y1)):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xChange = True
        if(x0 > x1):
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        y = y0 
        dy = 2.0*abs(y1 - y0)
        deeror = 0.0
        y_update = 1 if y1 > y0 else -1
        for x in range(x0, x1):
            if xChange:
                self.img[x, y] = color
            else:
                self.img[y,x] = color
            deeror += dy
            if deeror > (x1-x0):
                deeror -= 2.0*(x1 - x0)*1.0
                y += y_update
    def show_img(self):
        Image.fromarray(self.img).show()     


def Brezenherm(image,x0,y0,x1,y1,color):
   x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
   xChange = False
   if(abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xChange = True
   if(x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
   y = y0 
   dy = 2.0*abs(y1 - y0)
   deeror = 0.0
   y_update = 1 if y1 > y0 else -1
   for x in range(x0, x1):
        if xChange:
            image[x, y] = color
        else:
            image[y,x] = color
        deeror += dy
        if deeror > (x1-x0):
            deeror -= 2.0*(x1 - x0)*1.0
            y += y_update
# image = ImageClass(200, 200)
# for i in range(13):
#     a = 2*np.pi*i/13
#     x1 = int(100 + 95*np.cos(a))
#     y1 = int(100 + 95*np.sin(a))
#     x0 = 100
#     y0 = 100
    #image.dot_line_fix(100, 100, 100 + 95*np.cos(a), 100 + 95*np.sin(a), (255, 0, 0))
    #image.dot_line(100, 100, 100 + 95*np.cos(a), 100 + 95*np.sin(a), 10, (255, 0, 0))
    #image.dot_line2(100, 100, x1, y1, (255, 0, 0))
    #image.dot_line_2new(100, 100, x1, y1, (255, 0, 0))
    #image.dot_line_2fix1(x0, y0, x1, y1, (255, 0, 0))
    #image.dot_line3(x0, y0, x1, y1, (255, 0, 0))
#image.show_img() 
f = open("model_1.obj")

v = []
_f = []
for s in f:
    sp = s.split()
    if(sp[0] == 'v'):
        x = float(sp[1])
        y = float(sp[2])
        v.append([x, y])
    if(sp[0] == 'f'):
       x = int(sp[1].split('/')[0])
       y = int(sp[2].split('/')[0])
       z = int(sp[3].split('/')[0])
       _f.append([x, y, z])   
# Создаём пустое изображение
img2 = np.zeros((2000, 2000), dtype=np.uint8)
# Заполняем изображение точками
for i in v:
    X = i[0]
    Y = i[1]
    x_pixel = int(5000 * X + 1000)
    y_pixel = int(5000 * Y + 1000)
    img2[y_pixel, x_pixel] = 255
for l in _f:
    x0=5000*v[l[0] - 1][0]+1000
    y0=5000*v[l[0] - 1][1]+1000
    x1=5000*v[l[1] - 1][0]+1000
    y1=5000*v[l[1] - 1][1]+1000
    x2=5000*v[l[2] - 1][0]+1000
    y2=5000*v[l[2] - 1][1]+1000
    Brezenherm(img2,x0,y0,x1,y1, 255)
    Brezenherm(img2,x1,y1,x2,y2, 255)
    Brezenherm(img2, x0, y0, x2,y2, 255)

img12 = Image.fromarray(img2)
img12.show()
