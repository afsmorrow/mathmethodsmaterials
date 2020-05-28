# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:47:55 2020

Monte Carlo Method, make a cirlce in a square to find an estimate of pi

Do so visually, showing selection of random pixels and if they are inside or outside the circle

@author: morroa2
"""

import random
from PIL import Image, ImageDraw

if __name__ =='__main__':

    corner1 = (0,0)
    corner2 = (100,0)
    corner3 = (0,100)
    corner4 = (100,100)
    circle1 = (0,0)
    circle2 = (100,100)
    
    im = Image.new("RGB",(100,100), (128,128,128))
    draw = ImageDraw.Draw(im)
    draw.rectangle((corner1, corner4), fill = (255,255,255), outline = (255,255,255))
    draw.ellipse((circle1, circle2), fill = (0,0,0), outline = (0,0,0))
    inside = 0
    count = 0
    
    
    
    for i in range(10000):
        randX = int(random.random()*100)
        #print(randX)
        randY = int(random.random()*100)
        color = im.getpixel((randX,randY))
        #print(color)
        draw.point((randX,randY),fill = (255,0,0))
        if i%1000 == 0:
            im.show()
        if color == (0,0,0) or color == (255,0,0):
            inside +=1
            draw.point((randX,randY),fill = (255,0,0))
        else:
            draw.point((randX,randY),fill = (0,255,0))
        count +=1
    
    #print(inside)
    ratio = inside/count
    #print(ratio)
    pi = (ratio)*((4))
    print(pi)
