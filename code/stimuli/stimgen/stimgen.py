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
def gen2(savepath=None, text = 'text', index=1, mirror=False,
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
    if savepath:
        img.save(os.path.join(savepath, f'{text}{index}.jpg'))
    else:
        return img


def split_data(imgs, fraction_train, fraction_val):
    n_imgs = len(imgs)
    IXs = np.random.permutation(n_imgs)

    ed = int(n_imgs*fraction_train) # fraction of train data from total
    IXs_train = IXs[:ed] 
    IXs_test = IXs[ed:]

    n_train_imgs = len(IXs_train)
    ed = int(n_train_imgs*fraction_val) # fraction of val data from train
    IXs_val = IXs_train[ed:]
    IXs_train = IXs_train[:ed]
    return IXs_train, IXs_val, IXs_test


def dict2string(d):
    s = ''
    for key, value in d.items():
        s += f'{key}_{value}_'
    return s

######################
def CreateWordSet(path_out='../../../data/letters/',
                  n_train=1000,
                  n_val=100,
                  n_test=100):

    # set seed for replecability
    random.seed(1111)
    
    #define words, sizes, fonts
    wordlist = words
    sizes = range(15,31,2)
    fonts = ['arial', 'times', 'comic', 'cour', 'calibri']
    xshifts = range(-8, 8,2)
    yshifts = range(-8, 8,2)

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    gc.collect()
    
    for word in wordlist:
        print(f'Generating images for word: {word}')
        imgs, metas = [], []
        for size in sizes:
            for xshift in xshifts:
                for yshift in yshifts:
                    for font in fonts:
                        for upper in [0, 1]:
                            #print(size, xshift, yshift, font, upper)
                            img = gen2(savepath=None, index=None, # no saving
                                       text=word, fontname=font, size=size,
                                       xshift=xshift, yshift=yshift, upper=upper)
                            imgs.append(img)
                            metas.append({'word':word,
                                          'size':size,
                                          'xshift':xshift,
                                          'yshift':yshift,
                                          'font':font,
                                          'upper':upper})
        
        IXs_train, IXs_val, IXs_test = split_data(imgs, 0.8, 0.8)
        
        for dataset in ['train', 'val', 'test']:
            IXs = {'train':IXs_train,
                   'val':IXs_val,
                   'test':IXs_test}[dataset]
            for IX in IXs:
                path = os.path.join(path_out, dataset, metas[IX]['word'])
                os.makedirs(path, exist_ok=True)
                fn = dict2string(metas[IX])
                imgs[IX].save(os.path.join(path, fn + '.png'))

    return 'done'


def CreateFalseFonts(path_out='../../../data/letters/',
                     n_train=1000,
                     n_val=100,
                     n_test=100):

    # set seed for replecability
    random.seed(1111)
    
    #define words, sizes, fonts
    wordlist = words
    sizes = range(15,31,2)
    fonts = ['ff6'] # the corresponding real-letter font is deffvmonospaced3.ttf

    xshifts = range(-8, 8,2)
    yshifts = range(-8, 8,2)

    #for each word, create num_train + num_val exemplars, then split randomly into train and val.
    gc.collect()
    
    for word in wordlist:
        print(f'Generating images for word: {word}')
        imgs, metas = [], []
        for size in sizes:
            for xshift in xshifts:
                for yshift in yshifts:
                    for font in fonts:
                        for upper in [1]: # ff6 false font has only uppercase
                            #print(size, xshift, yshift, font, upper)
                            img = gen2(savepath=None, index=None, # no saving
                                       text=word, fontname=font, size=size,
                                       xshift=xshift, yshift=yshift, upper=upper)
                            imgs.append(img)
                            metas.append({'word':word,
                                          'size':size,
                                          'xshift':xshift,
                                          'yshift':yshift,
                                          'font':font,
                                          'upper':upper})
        
        
        for img, meta in zip(imgs, metas):
            path = os.path.join(path_out, 'false_fonts', meta['word'])
            os.makedirs(path, exist_ok=True)
            fn = dict2string(meta)
            img.save(os.path.join(path, fn + '.png'))

    return 'done'


CreateFalseFonts()
CreateWordSet()

