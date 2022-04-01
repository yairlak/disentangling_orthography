from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import subprocess, shlex, shutil, io, os, random, gc, time
from tqdm import tqdm
import pickle

import numpy as np
words = ['a','b','c','d','e','f','g','h','i','j',
         'k','l','m','n','o','p','q','r','s','t',
         'u','v','w','x','y','z']


####
def gen2(savepath='', text = 'text', index=1, mirror=False,
        invert=False, fontname='Arial', W = 64, H = 64, size=24,
        xshift=0, yshift=0, upper=0, show=None):
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new("RGB", (W,H), color = (255, 255, 255))
    fnt = ImageFont.truetype(fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)
    w, h = fnt.getsize(text)
    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')
    img.save(os.path.join(savepath, f'{text}{index}.jpg'))


######################
def CreateWordSet(path_out='../../../data/letters/',
                  n_train=1000,
                  n_val=100,
                  n_test=100):

    # set seed for replecability
    random.seed(1111)
    
    #define words, sizes, fonts
    wordlist = words
    sizes = [15, 25, 35]
    fonts = {}
    fonts['train'] = ['arial','times']
    fonts['val'] = ['comic','cour','calibri']
    fonts['test'] = ['comic','cour','calibri']

    xshift = [-6, -4, -2,0, 2,4,6]
    yshift = [-6, -4, -2,0, 2,4,6]

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    gc.collect()
    for dataset in ['train', 'val', 'test']:
        for w in wordlist:
            path = os.path.join(path_out, dataset, w)
            os.makedirs(path, exist_ok=True)

            n = {'train':n_train, # number of version from each word or letter
                 'val':n_val,
                 'test':n_test}[dataset]
            
            print(f'generating {n} images for {dataset.upper()} set, word {w.upper()}, and saving to: {path})')
            
            for i in range(n):
                f = random.choice(fonts[dataset])
                s = random.choice(sizes)
                u = random.choice([0, 1])
                x = random.choice(xshift)
                y = random.choice(yshift)
                gen2(savepath=path, text=w,
                     index=i, fontname=f, size=s,
                     xshift=x, yshift=y, upper=u)
        
    return 'done'
    
CreateWordSet()
