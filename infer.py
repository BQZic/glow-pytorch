"""Train script.

Usage:
    infer_celeba.py <hparams> <dataset_root> <z_dir>
"""
import os
import cv2
import random
import torch
import vision
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig

IMAGE_FILE_PATH = "/content/drive/MyDrive/Data/Test"

def select_index(name, l, r, description=None):
    index = None
    while index is None:
        print("Select {} with index [{}, {}),"
              "or {} for random selection".format(name, l, r, l - 1))
        if description is not None:
            for i, d in enumerate(description):
                print("{}: {}".format(i, d))
        try:
            line = int(input().strip())
            if l - 1 <= line < r:
                index = line
                if index == l - 1:
                    index = random.randint(l, r - 1)
        except Exception:
            pass
    return index


def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]).cuda(), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = img[:, :, ::-1]
    img = cv2.resize(img, (256, 256))
    return img


def save_images(images, names):
    if not os.path.exists("/content/drive/MyDrive/Data/Test2"):
        os.makedirs("/content/drive/MyDrive/Data/Test2")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("/content/drive/MyDrive/Data/Test2/{}.png".format(name), img)
        # cv2.imshow("img", img)
        # cv2.waitKey()


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    if not os.path.exists(z_dir):
        print("Generate Z to {}".format(z_dir))
        os.makedirs(z_dir)
        generate_z = True
    else:
        print("Load Z from {}".format(z_dir))
        generate_z = False

    hparams = JsonConfig("hparams/celeba.json")
    dataset = vision.Datasets["celeba"]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    # build
    graph = build(hparams, False)["graph"]
    dataset = dataset(dataset_root, transform=transform)

    # get Z
    if not generate_z:
        # try to load
        try:
            delta_Z = []
            for i in range(hparams.Glow.y_classes):
                z = np.load(os.path.join(z_dir, "detla_z_{}.npy".format(i)))
                delta_Z.append(z)
        except FileNotFoundError:
            # need to generate
            generate_z = True
            print("Failed to load {} Z".format(hparams.Glow.y_classes))
            quit()
    if generate_z:
        delta_Z = graph.generate_attr_deltaz(dataset)
        for i, z in enumerate(delta_Z):
            np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
        print("Finish generating")

    images = []
    names = []

    for t in range(200):
        base_index = random.randint(0, len(dataset)-1)
        attr_index = random.randint(0, len(delta_Z)-1)
        attr_name = dataset.attrs[attr_index]
        z_delta = delta_Z[attr_index]
        graph.eval()
        z_base = graph.generate_z(dataset[base_index]["x"])

        interplate_n = 5
        i = random.randint(0, interplate_n)
        
        d = z_delta * float(i) / float(interplate_n)
        
        rnd_sign = random.randint(0, 1)
        signs = [-1, 1]
        if i > 0:
          sign = signs[rnd_sign]
        else: sign = 1
        
        images.append(run_z(graph, z_base + d*sign))
        names.append("attr_{}_{}_{}".format(base_index, attr_name, interplate_n + i*sign))
        
        if (t+1) % 10 == 0:
          save_images(images, names)
          images = []
          names = []


