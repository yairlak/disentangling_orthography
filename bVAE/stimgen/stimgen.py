from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import subprocess, shlex, shutil, io, os, random, gc, time
from tqdm import tqdm
import pickle

import numpy as np
words = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


####
def gen2(savepath='', text = 'text', index=1, mirror=False, invert=False, fontname='Arial', W = 64, H = 64, size=24, xshift=0, yshift=0, upper=0, show=None):
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new("RGB", (W,H), color = (255, 255, 255))
    fnt = ImageFont.truetype(fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)
    w, h = fnt.getsize(text)
    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')

    img.save(savepath+text+str(index)+'.jpg')


######################
def CreateWordSet(path_out='../letters/',num_train=100, num_val=10):
    #define words, sizes, fonts
    wordlist = words
    sizes = [15, 25, 35]
    fonts_tr = ['arial','times']
    fonts_val = ['comic','cour','calibri']

    xshift = [-6, -4, -2,0, 2,4,6]
    yshift = [-6, -4, -2,0, 2,4,6]

    #create train and val folders
    for m in ['train', 'val']:
        for f in wordlist:
            target_path = path_out+m+'/'+f
            os.makedirs(target_path)

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    for w in tqdm(wordlist):
        gc.collect()
        print (w,)
        for n in range(num_train + num_val):
            if n < num_train:
                path = path_out+'train/'+w+'/'
                fonts = fonts_tr
            else:
                path = path_out+'val/'+w+'/'
                fonts = fonts_val

            f = random.choice(fonts)
            s = random.choice(sizes)
            u = random.choice([0,1])
            x = random.choice(xshift)
            y = random.choice(yshift)
            gen2(savepath=path, text=w, index=n, fontname=f, size=s, xshift=x, yshift=y, upper=u)

    return 'done'
    
CreateWordSet()

