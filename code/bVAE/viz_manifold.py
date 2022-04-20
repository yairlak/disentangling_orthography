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
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='MDS', type=str,
                    choices=['PCA', 'MDS'])
parser.add_argument('--n-components', default=2, type=int)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--latent-size', default=128, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--learning-rate', default=0.001, type=float)
parser.add_argument('--epochs', default=300, type=float)
parser.add_argument('--image-path',
                    default='../../data/letters/train/')
parser.add_argument('--model-path',
                    default='../../trained_models/checkpoints/')
parser.add_argument('--figure-path',
                    default='../../figures/viz_manifold/')
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


kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}
data_dir = '../../data/letters/'
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                               transform = im_transform)

subsample = True
if subsample:
    IXs = torch.tensor(np.random.choice(len(dataset), 5000, replace=False))
else:
    IXs = np.arange(len(dataset))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          # sampler=SubsetRandomSampler(IXs),
                                          shuffle=False, **kwargs)


model = models.BetaVAE(latent_size=args.latent_size,
                       beta = args.beta).to(device)
start_epoch = model.load_last_model(args.model_path, hyperparams) + 1


X, labels, uppers, fonts = [], [], [], []
for batch_idx, data in enumerate(data_loader):
    fn = os.path.basename(data_loader.dataset.samples[batch_idx][0])
    if batch_idx % 1000 == 0:
        print(f'sample {batch_idx}/{len(data_loader)}, {fn}')
    
    # print(fn)
    _, label, _, s, _, xshift, _, yshift, _, font, _, upper, _ = fn.split('_')
    z = get_z(data[0][0], model, device)
    # Append
    labels.append(label)
    uppers.append(upper)
    fonts.append(font)
    X.append(z.cpu()[0, :].numpy())
X = np.asarray(X)

labels = np.asarray(labels)[IXs]
uppers = np.asarray(uppers)[IXs]
fonts = np.asarray(fonts)[IXs]
X = X[IXs, :]
    
if args.method == 'MDS':
    embedding = MDS(n_components=args.n_components)
    X_transformed = embedding.fit_transform(X)
elif args.method == 'PCA':
    pca = PCA(n_components=args.n_components)
    X_transformed = pca.fit_transform(X)
    

def average_representations(X, labels, uppers):
    labels_ = np.asarray([l.upper() if upper=='1' else l for (l, upper) in zip(labels, uppers)])
    labels_set = np.asarray(sorted(list(set(labels_))))
    X_average, new_labels = [], []
    for l in labels_set:
        IXs = labels_ == l
        X_curr_label = X[IXs, :]
        # print(l, X_curr_label)
        X_curr_average = np.mean(X_curr_label, axis=0)
        X_average.append(X_curr_average)
        new_labels.append(l)
    return np.asarray(X_average), np.asarray(new_labels)


X_average, labels_average = average_representations(X_transformed, labels, uppers)

# PLOT
fig, ax = plt.subplots(figsize=(20,20))

# font_types = list(set(fonts))
for i, ((x, y), label) in enumerate(zip(X_average, labels_average)):
    # if upper:
    #     label = label.upper()
    print(x, y, label)
    ax.text(x, y, label, fontsize=30)

x_max = np.max((np.abs(X_average[:, 0].min()), np.abs(X_average[:, 0].max())))
y_max = np.max((np.abs(X_average[:, 1].min()), np.abs(X_average[:, 1].max())))
ax.set_xlim((-x_max, x_max))
ax.set_ylim((-y_max, y_max))
plt.axis('off')

os.makedirs(args.figure_path, exist_ok=True)
fn_fig = os.path.join(args.figure_path,
                      str(args.method) + '_' + hyperparams[:-1]+ '.png')


fig.savefig(fn_fig)
plt.close(fig)
print(f'Figure saved to: {fn_fig}')