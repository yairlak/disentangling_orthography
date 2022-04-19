import os
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
import sys
import models
import utils
import torch
import numpy as np
import multiprocessing
from torchvision import datasets, transforms
from torchvision.utils import save_image
from analyze import get_z

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--latent-size', default=128, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--learning-rate', default=0.001, type=float)
parser.add_argument('--epochs', default=300, type=float)
parser.add_argument('--image-path',
                    default='../../data/letters/train/')
parser.add_argument('--model-path',
                    default='../../trained_models/checkpoints/')
parser.add_argument('--compare-path',
                    default='../../figures/pertubed_reconstructed/')
parser.add_argument('--use-cuda', default=False, action='store_true')
args = parser.parse_args()

hyperparams = utils.dict2string(args.__dict__,
                                ['beta', 'latent_size', 'batch_size',
                                 'learning_rate', 'epochs'])

use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])


def get_imgs(img_ids, list_imgs):
    list_imgs = [l[0] for l in list_imgs]
    list_imgs_fn = [os.path.basename(s) for s in list_imgs]
    IXs = []
    for img_id in img_ids:
        IX = list_imgs_fn.index(img_id + '.png')
        IXs.append(IX)
    return IXs




img_ids = ['word_a_size_15_xshift_0_yshift_2_font_arial_upper_0_',
           'word_c_size_15_xshift_0_yshift_2_font_arial_upper_0_',
           'word_h_size_15_xshift_0_yshift_2_font_arial_upper_0_']


n=3
kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}
data_dir = '../../data/letters/'
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                               transform = im_transform)
IXs_imgs = get_imgs(img_ids, dataset.imgs)

dataset_subset = torch.utils.data.Subset(dataset, IXs_imgs)
data_loader = torch.utils.data.DataLoader(dataset_subset, batch_size=1,
                                          shuffle=False, **kwargs)

model = models.BetaVAE(latent_size=args.latent_size,
                       beta = args.beta).to(device)
start_epoch = model.load_last_model(args.model_path, hyperparams) + 1

path_comparison = os.path.join(args.compare_path, hyperparams)
os.makedirs(path_comparison, exist_ok=True)

perts = range(-10, 11, 2)
original_images, rect_images = [], []
for batch_idx, (data, img_id) in enumerate(zip(data_loader, img_ids)):
    output, mu, logvar = model(data[0])
    z = get_z(data[0][0], model, device)
    
    pertubed_reconstructed = []
    for i_z in range(z.shape[1]):
        for pert in perts:
            z_pertubed = z.clone()
            z_pertubed[0, i_z] += pert
            with torch.no_grad():
                curr_img = model.decode(z_pertubed).cpu()
            pertubed_reconstructed.append(curr_img[0])
    
    save_image(pertubed_reconstructed,
                       os.path.join(path_comparison, f'{img_id}.png'),
                       padding=0, nrow=len(perts))
