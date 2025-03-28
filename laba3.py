import numpy as np
from PIL import Image  
import math
import random
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
        temp = 2*(x1 - x0)
        y = y0 
        dy = temp*abs(y1 - y0) / (x1 - x0)
        deeror = 0.0
        y_update = 1 if y1 > y0 else -1
        for x in range(x0, x1):
            if xChange:
                self.img[x, y] = color
            else:
                self.img[y,x] = color
            deeror += dy
            if deeror > temp*0.5:
                deeror -= temp*1.0
                y += y_update
    def show_img(self):
        Image.fromarray(self.img).show()     


def Brezenherm(image,x0,y0,x1,y1,color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y=y0
    dy = 2.0*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0,x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror >(x1-x0)):
            derror -= 2.0*(x1-x0)*1.0
            y += y_update
def baricentr0(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = float(((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    return lambda0
def baricentr1(x, y, x0, y0, x1, y1, x2, y2):
    lambda1 = float(((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    return lambda1
def baricentr2(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = float(((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda1 = float(((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)))
    lambda2 = float(1.0 - lambda0 - lambda1)
    return lambda2
def triangle(image, x0, y0, x1, y1, x2, y2, color):
    x0 = float(x0)
    x1 = float(x1)
    x2 = float(x2)
    y0 = float(y0)
    y1 = float(y1)
    y2 = float(y2)
    xmax = int(max(x0, x1, x2))
    xmin = int(min(x0, x1, x2))
    ymax = int(max(y0, y1, y2))
    ymin = int(min(y0, y1, y2))
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax < xmin:
        xmax = 0
    if ymax < ymin:
        ymax = 0
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if baricentr0(x, y, x0, y0, x1, y1, x2, y2) >= 0 and baricentr1(x, y, x0, y0, x1, y1, x2, y2) >= 0 and baricentr2(x, y, x0, y0, x1, y1, x2, y2) >= 0:
                image[y, x] = color

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array((x1-x2, y1-y2, z1-z2))
    v2 = np.array((x1-x0, y1-y0, z1-z0))
    return np.cross(v1, v2)
 
def R(a, b, c):
    arr1 = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
    arr2 = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    arr3 = np.array([[np.cos(c), np.sin(c), 0], [-np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return np.dot(np.dot(arr1, arr2), arr3)
f = open("model_1.obj")
R_arr = R(30, 96.1, 36)
tX=0.001
tY=0.03
tZ=0.25
v = []
_f = []
for s in f:
    sp = s.split()
    if(sp[0] == 'v'):
        x = float(sp[1])
        y = float(sp[2])
        z = float(sp[3])
        vArr = np.array([[x], [y], [z]])
        vArr = (R_arr @ vArr) + np.array([[tX],[tY],[tZ]])
        v.append([vArr[0,0], vArr[1,0], vArr[2,0]])
    if(sp[0] == 'f'):
       x = int(sp[1].split('/')[0])
       y = int(sp[2].split('/')[0])
       z = int(sp[3].split('/')[0])
       _f.append([x, y, z])   
img2 = np.zeros((2000, 2000, 3), dtype=np.uint8)
z_buf = np.full((2000, 2000), float(('inf')))
for l in _f:
    # x0=10000*v[l[0] - 1][0]+1000
    # y0=10000*v[l[0] - 1][1]+1000
    # z0=10000*v[l[0] - 1][2]+1000
    # x1=10000*v[l[1] - 1][0]+1000
    # y1=10000*v[l[1] - 1][1]+1000
    # z1=10000*v[l[1] - 1][2]+1000
    # x2=10000*v[l[2] - 1][0]+1000
    # y2=10000*v[l[2] - 1][1]+1000
    # z2=10000*v[l[2] - 1][2]+1000
    x0=v[l[0] - 1][0]
    y0=v[l[0] - 1][1]
    z0=v[l[0] - 1][2]
    x1=v[l[1] - 1][0]
    y1=v[l[1] - 1][1]
    z1=v[l[1] - 1][2]
    x2=v[l[2] - 1][0]
    y2=v[l[2] - 1][1]
    z2=v[l[2] - 1][2]
    px0=2500*x0/z0+1000
    py0=2500*y0/z0+1000
    px1=(2500*x1)/z1+1000
    py1=(2500*y1)/z1+1000
    px2=(2500*x2)/z2+1000
    py2=(2500*y2)/z2+1000
    x_min = max(0, int(min(px0, px1, px2)))
    x_max = min(1999, int(max(px0, px1, px2)))
    y_min = max(0, int(min(py0, py1, py2)))
    y_max = min(1999, int(max(py0, py1, py2)))
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    n_norm = np.linalg.norm(n)
    light = np.array([0, 0, 1])
    skalar = np.dot(n, light)
    light_norm = np.linalg.norm(light)
    color = (-255*skalar/(n_norm*light_norm), 0, 0)
    if (skalar/(n_norm*light_norm) < 0):
        for x in range(int(x_min), int(x_max)+1):
            for y in range(int(y_min), int(y_max)+1):
                lambda0 = baricentr0(x, y, px0, py0, px1, py1, px2, py2)
                lambda1 = baricentr1(x, y, px0, py0, px1, py1, px2, py2)
                lambda2 = baricentr2(x, y, px0, py0, px1, py1, px2, py2)
                if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:
                    z = lambda0*z0 + lambda1*z1 + lambda2*z2
                    if z < z_buf[y, x]:
                        img2[y,x] = color 
                        z_buf[y,x] = z
img = Image.fromarray(img2)
img.save("img3.png")
#img.show()