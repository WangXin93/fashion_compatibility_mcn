""" Conditional Similarity Network
During training TripletDataset load anchor, positive, negative images, and condition
A Test Dataset load pair images, target and condition, conditions are:

    Conditions
    upper_bottom
    upper_shoe
    upper_bag
    upper_accessory
    bottom_shoe
    bottom_bag
    bottom_accessory
    shoe_bag
    shoe_accessory
    bag_accessory
"""

import sys
import torch
import logging
import torchvision
import numpy as np
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from polyvore_dataset import TripletDataset, CategoryDataset
from evaluate import test_compatibility_auc, test_fitb_quesitons
from utils import AverageMeter, prepare_dataloaders, config_logging
from model import CompatModel

# Leave a comment for this training, and it will be used for name suffix of log and saved model
comment = '_'.join(sys.argv[1:])

# Logger
config_logging(comment)

# Hyperparameters
img_size = 112
emb_size = 64
device = torch.device("cuda")

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
    transform=transform,
    data_file="train_no_dup_with_category_3more_name.json",
)
train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=4)
val_dataset = TripletDataset(
    root_dir="../../data/images/",
    data_dir="../../data/",
    transform=transform,
    data_file="valid_no_dup_with_category_3more_name.json",
    is_train=True,
)
val_loader = DataLoader(val_dataset, 32, shuffle=False, num_workers=4)
test_dataset = TripletDataset(
    root_dir="../../data/images/",
    data_dir="../../data/",
    transform=transform,
    data_file="test_no_dup_with_category_3more_name.json",
    is_train=True,
)
test_loader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4)

val_auc_dataset = CategoryDataset(
    root_dir="../../data/images/",
    data_dir="../../data/",
    transform=transform,
    use_mean_img=True,
    data_file="valid_no_dup_with_category_3more_name.json",
    neg_samples=True,
)

# Model
tnet = CompatModel(
    emb_size,
    n_conditions=len(train_dataset.conditions) // 2,
    learnedmask=True,
    prein=False,
)
tnet = tnet.to(device)

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum().item() / dista.size(0)

# Hyperparameters
criterion = torch.nn.MarginRankingLoss(margin=0.2)
parameters = filter(lambda p: p.requires_grad, tnet.parameters())
optimizer = torch.optim.Adam(parameters, lr=5e-5)
n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
logging.info(" + Number of params: {}".format(n_parameters))


# Train process
best_acc = -1
for epoch in range(1, 50 + 1):
    logging.info("** Epoch: {} **".format(epoch))
    # Train phase
    tnet.train()
    losses = AverageMeter()
    accs = AverageMeter()
    for batch_num, (a_img, p_img, n_img, c) in enumerate(train_loader, 1):
        a_img, p_img, n_img, c = (
            a_img.to(device),
            p_img.to(device),
            n_img.to(device),
            c.to(device),
        )
        # Original code need input triplet like: anchor, far, close
        dista, distb, mask_norm, embed_norm, mask_embed_norm = tnet(a_img, n_img, p_img, c)
        target = torch.FloatTensor(dista.size()).fill_(1).to(device)
        loss_triplet = criterion(dista, distb, target)
        loss_embed = embed_norm / np.sqrt(a_img.size(0))
        loss_mask = mask_norm / a_img.size(0)
        loss = loss_triplet + 5e-3 * loss_embed + 5e-4 * loss_mask
        losses.update(loss_triplet.item(), a_img.size(0))
        accs.update(accuracy(dista, distb), a_img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 50 == 0:
            logging.info("#{} Loss: {:.4f} (avg: {:.4f}), Accuracy: {:.4f}".format(
                batch_num, losses.val, losses.avg, accs.avg))
    logging.info("Train Loss: {:.4f}, Accuracy: {:.4f}".format(losses.avg, accs.avg))

    # Evaluation phase
    tnet.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    for batch_num, (a_img, p_img, n_img, c) in enumerate(val_loader, 1):
        a_img, p_img, n_img, c = (
            a_img.to(device),
            p_img.to(device),
            n_img.to(device),
            c.to(device),
        )
        # Original code need input triplet like: anchor, far, close
        with torch.no_grad():
            dista, distb, mask_norm, embed_norm, mask_embed_norm = tnet(a_img, n_img, p_img, c)

        target = torch.FloatTensor(dista.size()).fill_(1).to(device)
        loss_triplet = criterion(dista, distb, target)
        losses.update(loss_triplet.item(), a_img.size(0))
        accs.update(accuracy(dista, distb), a_img.size(0))

    # Valid AUC
    auc = test_compatibility_auc(val_auc_dataset, tnet.embeddingnet)
    acc = test_fitb_quesitons(val_auc_dataset, tnet.embeddingnet)
    logging.info("Valid Loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, FitbACC: {:.4f}".format(
        losses.avg, accs.avg, auc, acc))

    # Save Model
    if accs.avg > best_acc:
        best_acc = accs.avg
        torch.save(tnet.state_dict(), "csn_model_best.pth")
        logging.info("Found Best Accuracy {:.4f}, saved model to csn_model_best.pth".format(accs.avg))
