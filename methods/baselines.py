import os
from .utils import download_checkpoint, download_zip, init_pytorch_DeepWV3Plus, get_softmax, get_calibrated_softmax, \
    mahalanobis_modification, get_activations, load_gdrive_file
import numpy as np
import torch
import h5py
from scipy.stats import entropy
import tensorflow as tf

load_dir = "/tmp/checkpoints/"


class Max_softmax(object):
    def __init__(self, modelid):
        checkpoint_path = os.path.join(load_dir, "DeepLabV3+_WideResNet38_baseline.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            filename = download_checkpoint(checkpoint_download_url, load_dir)
            os.rename(os.path.join(load_dir, filename), checkpoint_path)
        self.model = init_pytorch_DeepWV3Plus(checkpoint_path)

    def anomaly_score(self, image):
        probs = get_softmax(self.model, image)
        return 1 - np.max(probs, axis=0)


class ODIN(object):
    def __init__(self, modelid, magnitude=0.0001, temperature=3.0):
        self.magnitude = magnitude
        self.temperature = temperature
        checkpoint_path = os.path.join(load_dir, "DeepLabV3+_WideResNet38_baseline.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            filename = download_checkpoint(checkpoint_download_url, load_dir)
            os.rename(os.path.join(load_dir, filename), checkpoint_path)
        self.model = init_pytorch_DeepWV3Plus(checkpoint_path)

    def anomaly_score(self, image):
        calibrated_softmax = get_calibrated_softmax(self.model, image, self.magnitude, self.temperature, as_numpy=True)
        return 1 - np.max(calibrated_softmax, axis=0)


class Mahalanobis(object):
    def __init__(self, modelid):
        checkpoint_path = os.path.join(load_dir, "DeepLabV3+_WideResNet38_baseline.pth")
        estimates_path = os.path.join(load_dir, "cityscapes_train_estimates_global.h5")
        if not os.path.exists(checkpoint_path) or not os.path.exists(estimates_path):
            zip_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            download_zip(zip_download_url, load_dir)
        with h5py.File(estimates_path, "r") as data:
            self.arithmetic_means = np.array(data['means'])
            self.arithmetic_means = self.arithmetic_means.reshape(self.arithmetic_means.shape[:2] + (-1,))
            self.inverse_covariances = np.array(data['inverse'])
            self.inverse_covariances = torch.from_numpy(self.inverse_covariances.astype("float"))
        self.model = mahalanobis_modification(init_pytorch_DeepWV3Plus(), checkpoint_path)

    def forward_pass(self, image):
        return get_activations(self.model, image)

    def anomaly_score(self, image):
        activations = self.forward_pass(image)
        output_shape = activations.shape
        mahalanobis = np.zeros((len(self.arithmetic_means),) + output_shape[1:])
        for c in range(len(self.arithmetic_means)):
            z = activations.reshape(output_shape[0], -1) - self.arithmetic_means[c]
            z = torch.from_numpy(z.astype("float"))
            left = torch.einsum('ij,ik->kj', z, self.inverse_covariances[c])
            score = torch.einsum('ij,ij->j', left, z).numpy()
            mahalanobis[c] = score.reshape(output_shape[1:])
        return np.min(mahalanobis, axis=0).astype("float32")


class Entropy_max(object):
    def __init__(self, modelid):
        checkpoint_path = os.path.join(load_dir, "DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            filename = download_checkpoint(checkpoint_download_url, load_dir)
            os.rename(os.path.join(load_dir, filename), checkpoint_path)
        self.model = init_pytorch_DeepWV3Plus(checkpoint_path)

    def anomaly_score(self, image):
        probs = get_softmax(self.model, image)
        return entropy(probs, axis=0) / np.log(probs.shape[0]).astype("float32")


class voidclassifier(object):
    def __init__(self, modelid):
        load_gdrive_file(modelid, load_dir)
        tf.compat.v1.enable_resource_variables()
        self.model = tf.saved_model.load(load_dir)

    def anomaly_score(self, image):
        image = tf.cast(image, tf.float32)
        image_shape = image.shape[:2]
        image = tf.image.resize(image, (1024, 2048))
        out = self.model.signatures['serving_default'](image[tf.newaxis])['anomaly_score']
        out = tf.image.resize(out[..., tf.newaxis], image_shape)
        return tf.squeeze(out).numpy().astype("float32")


class dropout(object):
    def __init__(self, modelid):
        load_gdrive_file(modelid, load_dir)
        tf.compat.v1.enable_resource_variables()
        self.model = tf.saved_model.load(load_dir)

    def anomaly_score(self, image):
        image = tf.cast(image, tf.float32)
        image_shape = image.shape[:2]
        image = tf.image.resize(image, (1024, 2048))
        out = self.model.signatures['serving_default'](image[tf.newaxis])['anomaly_score']
        out = tf.image.resize(out[..., tf.newaxis], image_shape)
        return tf.squeeze(out).numpy().astype("float32")


class mindensity(object):
    def __init__(self, modelid):
        load_gdrive_file(modelid, load_dir)
        tf.compat.v1.enable_resource_variables()
        self.model = tf.saved_model.load(load_dir)

    def anomaly_score(self, image):
        image = tf.cast(image, tf.float32)
        image_shape = image.shape[:2]
        image = tf.image.resize(image, (1024, 2048))
        out = self.model.signatures['serving_default'](image[tf.newaxis])['anomaly_score']
        out = tf.image.resize(out[..., tf.newaxis], image_shape)
        return tf.squeeze(out).numpy().astype("float32")
