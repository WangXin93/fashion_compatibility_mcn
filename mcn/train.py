import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.optim import lr_scheduler
from torchvision import models

from model import CompatModel
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders

# Leave a comment for this training, and it will be used for name suffix of log and saved model
import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Training.')
parser.add_argument('--vse_off', action="store_true")
parser.add_argument('--pe_off', action="store_true")
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--conv_feats', type=str, default="1234")
parser.add_argument('--comment', type=str, default="")
args = parser.parse_args()

print(args)
comment = args.comment
vse_off = args.vse_off
pe_off = args.pe_off
mlp_layers = args.mlp_layers
conv_feats = args.conv_feats

# Logger
config_logging(comment)

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(batch_size=16)
)

# Device
device = torch.device("cuda:0")

# Model
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary),
                    vse_off=vse_off, pe_off=pe_off, mlp_layers=mlp_layers, conv_feats=conv_feats)

# Train process
def train(model, device, train_loader, val_loader, comment):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    saver = BestSaver(comment)
    epochs = 50
    for epoch in range(1, epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))
        scheduler.step()
        total_losses = AverageMeter()
        clf_losses = AverageMeter()
        vse_losses = AverageMeter()
        # Train phase
        model.train()
        for batch_num, batch in enumerate(train_loader, 1):
            lengths, images, names, offsets, set_ids, labels, is_compat = batch
            images = images.to(device)

            # Forward
            output, vse_loss, tmasks_loss, features_loss = model(images, names)

            # BCE Loss
            target = is_compat.float().to(device)
            output = output.squeeze(dim=1)
            clf_loss = criterion(output, target)

            # Sum all losses up
            features_loss = 5e-3 * features_loss
            tmasks_loss = 5e-4 * tmasks_loss
            total_loss = clf_loss + vse_loss + features_loss + tmasks_loss

            # Update Recoder
            total_losses.update(total_loss.item(), images.shape[0])
            clf_losses.update(clf_loss.item(), images.shape[0])
            vse_losses.update(vse_loss.item(), images.shape[0])

            # Backpropagation
            model.zero_grad()
            total_loss.backward()
            optimizer.step()
            if batch_num % 10 == 0:
                logging.info(
                    "[{}/{}] #{} clf_loss: {:.4f}, vse_loss: {:.4f}, features_loss: {:.4f}, tmasks_loss: {:.4f}, total_loss:{:.4f}".format(
                        epoch, epochs, batch_num, clf_losses.val, vse_losses.val, features_loss, tmasks_loss, total_losses.val
                    )
                )
        logging.info("Train Loss (clf_loss): {:.4f}".format(clf_losses.avg))

        # Valid Phase
        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        clf_losses = AverageMeter()
        outputs = []
        targets = []
        for batch_num, batch in enumerate(val_loader, 1):
            lengths, images, names, offsets, set_ids, labels, is_compat = batch
            images = images.to(device)
            target = is_compat.float().to(device)
            with torch.no_grad():
                output, _, _, _ = model._compute_score(images)
                output = output.squeeze(dim=1)
                clf_loss = criterion(output, target)
            clf_losses.update(clf_loss.item(), images.shape[0])
            outputs.append(output)
            targets.append(target)
        logging.info("Valid Loss (clf_loss): {:.4f}".format(clf_losses.avg))
        outputs = torch.cat(outputs).cpu().data.numpy()
        targets = torch.cat(targets).cpu().data.numpy()
        auc = metrics.roc_auc_score(targets, outputs)
        logging.info("AUC: {:.4f}".format(auc))
        predicts = np.where(outputs > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(predicts, targets)
        logging.info("Accuracy@0.5: {:.4f}".format(accuracy))
        positive_loss = -np.log(outputs[targets==1]).mean()
        logging.info("Positive loss: {:.4f}".format(positive_loss))
        positive_acc = sum(outputs[targets==1]>0.5) / len(outputs)
        logging.info("Positive accuracy: {:.4f}".format(positive_acc))

        # Save best model
        saver.save(auc, model.state_dict())

if __name__ == "__main__":
    train(model, device, train_loader, val_loader, comment)
