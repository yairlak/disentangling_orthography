import os
import argparse
import models
import utils
import torch
import pandas as pd
import numpy as np
import multiprocessing
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser()
parser.add_argument('--data-type', default='test', type=str,
                    choices=['train', 'test', 'val'])
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--latent-size', default=16, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--learning-rate', default=0.01, type=float)
parser.add_argument('--epochs', default=300, type=float)
parser.add_argument('--image-path',
                    default='../../data/letters/train/')
parser.add_argument('--model-path',
                    default='../../trained_models/checkpoints/')
parser.add_argument('--output-path',
                    default='../../output/')
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


def get_imgs(target_data_type, list_imgs):
    data_types = [os.path.basename(os.path.dirname(os.path.dirname(l[0]))) \
                  for l in list_imgs]
    
    IXs = []
    for IX, data_type in enumerate(data_types):
        if data_type == target_data_type:
            IXs.append(IX)
    return IXs

# LOAD MODEL
model = models.BetaVAE(latent_size=args.latent_size,
                       beta = args.beta).to(device)
start_epoch = model.load_last_model(args.model_path, hyperparams) + 1

# LOAD DATA
data_dir = '../../data/letters/'
kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}
dataset = datasets.ImageFolder(os.path.join(data_dir, args.data_type),
                               transform = im_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=False, **kwargs)

# EXTRACT ACTIVATIONS
df = []
for i_data, (data, fn_img) in enumerate(zip(data_loader, dataset.imgs)):
    if i_data % 1000 == 0:
        print(f'image {i_data}/{len(data_loader)}')
    _, mu, logvar = model(data[0])
    _, letter, _, size, _, xshift, _, yshift, _, font, _, upper, _ = \
        os.path.basename(fn_img[0]).split('_')
    # ADPPEND TO DATAFRAME
    df.append([letter, size, xshift, yshift, font, upper, mu, logvar])
df = pd.DataFrame(df, columns=['letter', 'size', 'xshift', 'yshift', 'font', 'upper', 'mu', 'logvar'])

# SAVE
output_path = os.path.join(args.output_path, hyperparams[:-1])
os.makedirs(output_path, exist_ok=True)
fn_df = f'activations_{args.data_type}_{hyperparams[:-1]}.csv'
df.to_csv(os.path.join(args.output_path, fn_df))