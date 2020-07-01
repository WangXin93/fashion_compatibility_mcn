import itertools
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from model import CompatModel
from polyvore_dataset import CategoryDataset, TripletDataset
from Resnet_18 import resnet18


# Hyperparameters
img_size = 112
emb_size = 64
device = torch.device("cuda")


# Helper functions
def test_compatibility_auc(test_auc_dataset, embeddingnet):
    """ Compute AUC of classifying compatibile and incompatible outfits
    """
    scores = []
    targets = []
    for i in range(len(test_auc_dataset)):
        print("#{}/{}\r".format(i, len(test_auc_dataset)), end="", flush=True)
        conditions = torch.tensor(
            [
                [-1, 0, 1, 2, 3],
                [0, -1, 4, 5, 6],
                [1, 4, -1, 7, 8],
                [2, 5, 7, -1, 9],
                [3, 6, 8, 9, -1],
            ],
            requires_grad=False,
        ).to(device)
        images, names, offsets, set_id, labels, is_compat = test_auc_dataset[i]
        images = images.to(device)
        labels = list(map(lambda e: 0 if "mean" in e else 1, labels))

        outfit_score = calc_outfit_score(images, labels, conditions, embeddingnet)
        scores.append(outfit_score)
        targets.append(is_compat)

    targets, scores = np.array(targets), np.array(scores)
    auc = metrics.roc_auc_score(1 - targets, scores)
    print()
    return auc


def test_fitb_quesitons(test_fitb_dataset, embeddingnet):
    """ Compute accuracy of correctly answering the fill-in-the-blank questions
    """
    is_correct = []
    for i in range(len(test_fitb_dataset)):
        print("#{}/{}\r".format(i, len(test_fitb_dataset)), end="", flush=True)
        outfit_scores = []
        conditions = torch.tensor(
            [
                [-1, 0, 1, 2, 3],
                [0, -1, 4, 5, 6],
                [1, 4, -1, 7, 8],
                [2, 5, 7, -1, 9],
                [3, 6, 8, 9, -1],
            ],
            requires_grad=False,
        ).to(device)
        items, labels, question_part, question_id, options, option_labels = test_fitb_dataset.get_fitb_quesiton(i)
        question_part = {
            "upper": 0,
            "bottom": 1,
            "shoe": 2,
            "bag": 3,
            "accessory": 4,
        }.get(question_part)

        images = items.to(device)
        labels = list(map(lambda e: 0 if "mean" in e else 1, labels))
        outfit_score = calc_outfit_score(images, labels, conditions, embeddingnet)
        outfit_scores.append(outfit_score)

        # Calculate distance for each options
        for option in options:
            images[question_part] = option
            outfit_score = calc_outfit_score(images, labels, conditions, embeddingnet)
            outfit_scores.append(outfit_score)

        # The original outfit should have lowest distance
        if min(outfit_scores) == outfit_scores[0]:
            is_correct.append(True)
        else:
            is_correct.append(False)
    print()
    return sum(is_correct) / len(is_correct)


def calc_outfit_score(images, labels, conditions, embeddingnet):
    """Calculate outfit score by calculate mean of all pair distance n among this outfit.

    Args:
        images: [5, 3, 224, 224] torch.FloatTensor
        labels: list of 5 element where 0 for mean_img, 1 for original image
        conditions: [5, 5] torch.tensor store condition for reach combination
        embeddingnet: A metric network get embedding for image
    """
    inputs = []
    conds = []
    mask = []
    outfit_score = 0.0

    for a, b in itertools.combinations(range(0, 5), 2):
        if labels[a] == 0 or labels[b] == 0:
            mask.append(0)
        else:
            mask.append(1)
        c = conditions[a][b]
        inputs.append(images[a])
        inputs.append(images[b])
        conds.append(c)
        conds.append(c)

    inputs = torch.stack(inputs)
    conds = torch.stack(conds)
    with torch.no_grad():
        embs = embeddingnet(inputs, conds)[0]
    embs = embs.reshape(10, 2, -1)
    embs = F.normalize(embs, dim=2)
    dist = F.pairwise_distance(embs[:, 0, :], embs[:, 1, :])
    mask = torch.tensor(mask).float().to(device)

    outfit_score = torch.sum(dist * mask) / mask.sum()
    return outfit_score.item()


def main():
    # Dataloader
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Scale((img_size, img_size)),
            torchvision.transforms.CenterCrop(112),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = TripletDataset(
        root_dir="../../data/images/",
        data_dir="../../data/",
        transform=transform
    )
    test_auc_dataset = CategoryDataset(
        root_dir="../../data/images/",
        data_dir="../../data/",
        transform=transform,
        use_mean_img=True,
        data_file="test_no_dup_with_category_3more_name.json",
        neg_samples=True,
    )

    # Model
    tnet = CompatModel(
        emb_size,
        n_conditions=len(train_dataset.conditions) // 2,
        learnedmask=True,
        prein=False,
    )
    tnet.load_state_dict(torch.load("./csn_model_best.pth"))
    tnet = tnet.to(device)
    tnet.eval()
    embeddingnet = tnet.embeddingnet

    # Test
    auc = test_compatibility_auc(test_auc_dataset, embeddingnet)
    print("AUC: {:.4f}".format(auc))
    fitb_accuracy = test_fitb_quesitons(test_auc_dataset, embeddingnet)
    print("Fitb Accuracy: {:.4f}".format(fitb_accuracy))
    # AUC: 0.8413 ACC: 0.5656


if __name__ == "__main__":
    main()
