import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from model import CompatModel
from polyvore_dataset import CategoryDataset, lstm_collate_fn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import prepare_dataloaders, config_logging

# Leave a comment for this training, and it will be used for name suffix of log and saved model
comment = '_'.join(sys.argv[1:])

# Logger
config_logging(comment)

# Hyperparameters
epochs = 30
batch_size = 8
emb_size = 512
log_step = 2
device = torch.device("cuda")

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders(
    root_dir="../../data/images",
    data_dir="../../data",
    img_size=299,
    batch_size=12,
    use_mean_img=False,
    neg_samples=False,
    collate_fn=lstm_collate_fn,
)

# Model
model = CompatModel(emb_size=emb_size, need_rep=True, vocabulary=len(train_dataset.vocabulary))
mode = model.to(device)

# Train process
def train(model, device, train_loader, val_loader, comment):
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-1, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    for epoch in range(1, epochs + 1):
        # Train phase
        total_loss = 0
        scheduler.step()
        model.train()
        for batch_num, input_data in enumerate(train_loader, 1):
            lengths, images, names, offsets, set_ids, labels, is_compat = input_data
            image_seqs = images.to(device)  # (20+, 3, 224, 224)

            # forward propagation
            f_loss, b_loss, vse_loss = model(image_seqs, names, lengths)
            all_loss = f_loss + b_loss + 1. * vse_loss
            # backward propagation
            model.zero_grad()
            all_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # clip gradient
            optimizer.step()

            total_loss += all_loss.item()
            if batch_num % log_step == 0:
                logging.info(
                    "Epoch [{}/{}], Step #{}, F_loss: {:.4f}, B_loss: {:.4f}, VSE_Loss: {:.4f}, All_loss: {:.4f}".format(
                        epoch, epochs, batch_num, f_loss.item(), b_loss.item(), vse_loss.item(), all_loss.item(),
                    )
                )

        logging.info("**Epoch {}**, Train Loss: {:.4f}".format(epoch, total_loss / batch_num))
        torch.save(model.state_dict(), os.path.join("model{}.pth".format(comment)))

        # Validate phase
        model.eval()
        total_loss = 0
        for batch_num, input_data in enumerate(val_loader, 1):
            lengths, images, names, offsets, set_ids, labels, is_compat = input_data
            image_seqs = images.to(device)  # (20+, 3, 224, 224)
            with torch.no_grad():
                f_loss, b_loss, _ = model._forward_and_backward(image_seqs, lengths)
                all_loss = f_loss + b_loss
            total_loss += all_loss.item()

        logging.info("**Epoch {}**, Valid Loss: {:.4f}".format(epoch, total_loss / batch_num))


if __name__ == "__main__":
    train(model, device, train_loader, val_loader, comment)
