import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from zipfile import ZipFile
import gdown

from .models.deepv3 import DeepWV3Plus
from .models.mynn import Norm2d

import wget
import os


def init_pytorch_DeepWV3Plus(ckpt_path=None, num_classes=19):
    print("Load PyTorch model", end="", flush=True)
    network = nn.DataParallel(DeepWV3Plus(num_classes))
    print("... ok")
    if ckpt_path is not None:
        print("Checkpoint file: %s" % ckpt_path, end="", flush=True)
        network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
        print("... ok\n")
    network = network.cuda().eval()
    return network


def download_checkpoint(url, save_dir):
    print("Download PyTorch checkpoint")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = wget.download(url, out=save_dir)
    return filename


def download_zip(url, save_dir):
    print("Download .zip")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = wget.download(url, out=save_dir)
    with ZipFile(os.path.join(save_dir, filename), 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    return filename


def load_gdrive_file(file_id, save_dir, ending='zip'):
    print("Downloads files from google drive, caches files that are already downloaded.")
    filename = '{}.{}'.format(file_id, ending) if ending else file_id
    filename = os.path.join(os.path.expanduser('~/.keras/datasets'), filename)
    if not os.path.exists(filename):
        gdown.download('https://drive.google.com/uc?id={}'.format(file_id), filename, quiet=False)
    ZipFile(filename).extractall(save_dir)
    return filename


def get_softmax(network, image, transform=None, as_numpy=True):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).cuda()
    with torch.no_grad():
        y = network(x)
    probs = F.softmax(y, 1)
    if as_numpy:
        probs = probs.data.cpu().numpy()[0].astype("float32")
    return probs


def get_calibrated_softmax(network, image, magnitude, temperature, transform=None, as_numpy=False):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).cuda()
    p = get_gradient_wrt_input(network, image, transform)
    x = torch.sub(x, p, alpha=magnitude)
    with torch.no_grad():
        y = network(x)
    y = y / temperature
    probs = F.softmax(y, 1)
    if as_numpy:
        probs = probs.data.cpu().numpy()[0].astype("float32")
    return probs


def get_gradient_wrt_input(network, image, transform=None, as_numpy=False):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).requires_grad_().cuda()
    y = network(x)
    softmax = nn.Softmax(dim=1)(y)
    pred = torch.argmax(softmax, dim=1).detach()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(softmax, pred.cuda())
    loss.backward(retain_graph=True)
    grad = -torch.autograd.grad(outputs=loss, inputs=x, retain_graph=True)[0]
    grad = torch.sign(grad)
    if as_numpy:
        grad = grad.cpu().detach().numpy()[0].astype("float32")
    return grad


def mahalanobis_modification(network, ckpt_path):
    network.module.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))
    print("Checkpoint file: %s" % ckpt_path, end="", flush=True)
    network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    print("... ok\n")
    return network.cuda().eval()


def get_activations(network, image, transform=None, as_numpy=True):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).cuda()
    with torch.no_grad():
        y = network(x)
    if as_numpy:
        y = y.data.cpu().numpy()[0].astype("float32")
    return y


