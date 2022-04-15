import os
import argparse
import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import numpy as np
import models
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image


# generate n=num images using the model
def generate(model, num, device):
    model.eval()
    z = torch.randn(num, model.latent_size).to(device)
    with torch.no_grad():
        return model.decode(z).cpu()


# returns pytorch tensor z
def get_z(im, model, device):
    model.eval()
    im = torch.unsqueeze(im, dim=0).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(im)
        z = model.sample(mu, logvar)

    return z


def linear_interpolate(im1, im2, model, device):
    model.eval()
    z1 = get_z(im1, model, device)
    z2 = get_z(im2, model, device)

    factors = np.linspace(1, 0, num=10)
    result = []

    with torch.no_grad():

        for f in factors:
            z = (f * z1 + (1 - f) * z2).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)

    return result


def get_average_z(ims, model, device):
    model.eval()
    z = torch.unsqueeze(torch.zeros(model.latent_size), dim=0)

    for im in ims:
        z += get_z(im, model, device).cpu()

    return z / len(ims)


def latent_arithmetic(im_z, attr_z, model, device):
    model.eval()

    factors = np.linspace(0, 1, num=10, dtype=float)
    result = []

    with torch.no_grad():

        for f in factors:
            z = im_z + (f * attr_z).type(torch.FloatTensor).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)

    return result


def plot_loss(train_loss, test_loss, filepath):
    train_x, train_l = zip(*train_loss)
    test_x, test_l = zip(*test_loss)
    plt.figure()
    plt.title('Train Loss vs. Test Loss')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.plot(train_x, train_l, 'b', label='train_loss')
    plt.plot(test_x, test_l, 'r', label='test_loss')
    plt.legend()
    plt.savefig(filepath)


def get_attr_ims(attr, num=10):
    ids = prep.get_attr(attr_map, id_attr_map, attr)
    dataset = prep.ImageDiskLoader(ids)
    indices = np.random.randint(0, len(dataset), num)
    ims = [dataset[i] for i in indices]
    idx_ids = [dataset.im_ids[i] for i in indices]
    return ims, idx_ids

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=5, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--test-batch-size', default=10, type=int)
parser.add_argument('--epochs', default=1200, type=int)
parser.add_argument('--latent-size', default=100, type=int)
parser.add_argument('--learning-rate', default=1e-2, type=float)
parser.add_argument('--use-cuda', default=False, action='store_true')
parser.add_argument('--print-interval', default=100, type=int)
parser.add_argument('--log-path', default='./logs/log.pkl', type=str)
parser.add_argument('--model-path', default='../../trained_models/checkpoints/', type=str)
parser.add_argument('--compare-path', default='../../figures/comparisons/', type=str)
parser.add_argument('--plot-path', default='../../figures/analyses/', type=str)
args = parser.parse_args()


def dict2string(d, keys):
    s = ''
    for k in keys:
        s += f'{k}_{d[k]}_'
    return s


hyperparams = dict2string(args.__dict__,
                          ['beta', 'latent_size', 'batch_size', 'learning_rate', 'epochs'])


use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
model = models.BetaVAE(latent_size=args.latent_size).to(device)
print('latent size:', model.latent_size)

#attr_map, id_attr_map = prep.get_attributes()

if __name__ == "__main__":

    #model.load_last_model(args.model_path, hyperparams)

    '''
    generate images using model
    '''
    # samples = generate(model, 60, device)
    # save_image(samples, OUTPUT_PATH + MODEL + '.png', padding=0, nrow=10)

    train_losses, test_losses = utils.read_log(args.log_path, ([], []))
    print(train_losses)
    print(test_losses)
    fn_fig = os.path.join(args.plot_path, f'loss_{hyperparams}.png')
    plot_loss(train_losses, test_losses, fn_fig)
    print(f'Figure for loss vs. time saved to: {fn_fig}')

    '''
    get image ids with corresponding attribute
    '''
    #ims, im_ids = get_attr_ims('eyeglasses', num=20)
    # utils.show_images(ims, titles=im_ids, tensor=True)
    # print(im_ids)

    #man_sunglasses_ids = ['172624.jpg', '164754.jpg', '089604.jpg', '024726.jpg']
    #man_ids = ['056224.jpg', '118398.jpg', '168342.jpg']
    #woman_smiles_ids = ['168124.jpg', '176294.jpg', '169359.jpg']
    #woman_ids = ['034343.jpg', '066393.jpg']

    #man_sunglasses = prep.get_ims(man_sunglasses_ids)
    #man = prep.get_ims(man_ids)
    #woman_smiles = prep.get_ims(woman_smiles_ids)
    #woman = prep.get_ims(woman_ids)

    # utils.show_images(man_sunglasses, tensor=True)
    # utils.show_images(man, tensor=True)
    # utils.show_images(woman_smiles, tensor=True)
    # utils.show_images(woman, tensor=True)

    '''
    latent arithmetic
    '''
    #man_z = get_z(man[0], model, device)
    #woman_z = get_z(woman[1], model, device)
    #sunglass_z = get_average_z(man_sunglasses, model, device) - get_average_z(man, model, device)
    #arith1 = latent_arithmetic(man_z, sunglass_z, model, device)
    #arith2 = latent_arithmetic(woman_z, sunglass_z, model, device)

    #save_image(arith1 + arith2, OUTPUT_PATH + 'arithmetic-dfc' + '.png', padding=0, nrow=10)

    '''
    linear interpolate
    '''
    #inter1 = linear_interpolate(man[0], man[1], model, device)
    #inter2 = linear_interpolate(woman[0], woman_smiles[1], model, device)
    #inter3 = linear_interpolate(woman[1], woman_smiles[0], model, device)

    #save_image(inter1 + inter2 + inter3, OUTPUT_PATH + 'interpolate-dfc' + '.png', padding=0, nrow=10)
