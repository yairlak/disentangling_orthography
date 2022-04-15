import os
import argparse
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=5, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--test-batch-size', default=10, type=int)
parser.add_argument('--epochs', default=1200, type=int)
parser.add_argument('--latent-size', default=100, type=int)
parser.add_argument('--learning-rate', default=1e-2, type=float)
parser.add_argument('--use-cuda', default=False, action='store_true')
parser.add_argument('--print-interval', default=100, type=int)
parser.add_argument('--log-path', default='logs/', type=str)
parser.add_argument('--model-path', default='../../trained_models/checkpoints/', type=str)
parser.add_argument('--compare-path', default='../../figures/comparisons/', type=str)
parser.add_argument('--plot-path', default='../../figures/analyses/', type=str)
args = parser.parse_args()


def dict2string(d, keys, remove_key_names=False):
    s = ''
    for k in keys:
        if remove_key_names:
            s+= f'{d[k]}_'
        else:
            s += f'{k}_{d[k]}_'
    return s


hyperparams = dict2string(args.__dict__,
                          ['beta', 'latent_size',
                           'batch_size', 'learning_rate'],
                          remove_key_names=False)

fn_log = os.path.join(args.log_path, f'log_{hyperparams[:-1]}.pkl')

if os.path.exists(fn_log):
    train_loss, test_loss = pickle.load(open(fn_log, 'rb'))
else:
    print(f'Log not found: {fn_log}')
    raise()



# PLOT AND SAVE
fn_fig = os.path.join(args.plot_path, f'loss_{hyperparams[:-1]}.png')

train_x, train_l = zip(*train_loss)
test_x, test_l = zip(*test_loss)

plt.figure(figsize=(10, 10))
plt.title(f'{hyperparams}')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.plot(train_x, train_l, 'b', label='Train loss', lw=3)
plt.plot(test_x, test_l, 'r', label='Test loss', lw=3)
plt.legend()
plt.savefig(fn_fig)


print(f'Figure for loss vs. time saved to: {fn_fig}')

