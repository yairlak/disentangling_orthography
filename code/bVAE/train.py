import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import models
import utils
from torchvision.utils import save_image
import argparse


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data[0].to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--test-batch-size', default=10, type=int)
parser.add_argument('--epochs', default=1200, type=int)
parser.add_argument('--latent-size', default=100, type=int)
parser.add_argument('--learning-rate', default=1e-2, type=float)
parser.add_argument('--use-cuda', default=True, action='store_true')
parser.add_argument('--print-interval', default=100, type=int)
parser.add_argument('--log-path', default='./logs/log.pkl', type=str)
parser.add_argument('--model-path', default='../../trained_models/checkpoints/', type=str)
parser.add_argument('--compare-path', default='../../figures/comparisons/', type=str)
args = parser.parse_args()


def dict2string(d, keys):
    s = ''
    for k in keys:
        s += f'{k}_{d[k]}_'
    return s


hyperparams = dict2string(args.__dict__,
                          ['latent_size', 'batch_size', 'learning_rate', 'epochs'])

use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

# training code
# train_ids, test_ids = prep.split_dataset()
# print('num train_images:', len(train_ids))
# print('num test_images:', len(test_ids))

# data_train = prep.ImageDiskLoader(train_ids)
# data_test = prep.ImageDiskLoader(test_ids)

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}


###############################
from torchvision import datasets, transforms
import os
data_dir = '../../data/letters/'
im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

chosen_datasets = {}
data_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform =  im_transform)
data_test = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform =  im_transform)
################################

train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

print('latent size:', args.latent_size)

model = models.BetaVAE(latent_size=args.latent_size, beta = 5).to(device)
# model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if __name__ == "__main__":
    start_epoch = model.load_last_model(args.model_path) + 1
    train_losses, test_losses = utils.read_log(args.log_path, ([], []))

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, args.print_interval)
        test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

        path_comparison = os.path.join(args.compare_path, hyperparams)
        os.makedirs(path_comparison, exist_ok=True)
        save_image(original_images + rect_images,
                   os.path.join(path_comparison, f'{epoch}.png'),
                   padding=0, nrow=len(original_images))

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        utils.write_log(args.log_path, (train_losses, test_losses))

        model.save_model(os.path.join(args.model_path, f'{hyperparams}{epoch}.pt'))
