import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torchvision
import logging
import torch
import os
import sys
from tqdm import tqdm, trange
import pickle
import numpy as np
from sklearn import metrics
from polyvore_dataset import CategoryDataset, lstm_collate_fn
from model import EncoderCNN, LSTMModel, CompatModel
from utils import prepare_dataloaders

emb_size = 512
device = torch.device("cuda")

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders(
    root_dir="../../data/images",
    data_dir="../../data",
    img_size=299,
    use_mean_img=False,
    neg_samples=False,
    collate_fn=lstm_collate_fn,
)

# Restore model parameters
model = CompatModel(emb_size=emb_size, need_rep=False, vocabulary=len(train_dataset.vocabulary))
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

# Compute feature or Load extracted feature
if os.path.exists("test_features.pkl"):
    print("Found test_features.pkl...")
    test_features = pickle.load(open('./test_features.pkl', 'rb'))
else:
    print("Extract cnn features...")
    test_features = {}
    for input_data in tqdm(test_loader):
        lengths, images, names, offsets, set_ids, labels, is_compat = input_data
        image_seqs = images.to(device)
        with torch.no_grad():
            emb_seqs = model.encoder_cnn(image_seqs)
        batch_ids = []
        for set_id, items in zip(set_ids, labels):
            for item in items:
                batch_ids.append(item)
        for i, id in enumerate(batch_ids):
            test_features[id] = emb_seqs[i].cpu().detach().numpy()
    print()
    pickle.dump(test_features, open("test_features.pkl", "wb"))

test_features_ids = []
test_features_matrix = []
for k, v in test_features.items():
    test_features_matrix.append(v)
    test_features_ids.append(k)
test_features_matrix = torch.tensor(np.stack(test_features_matrix)).to(device)

# Compatibility AUC test
criterion = nn.CrossEntropyLoss()
f_losses, b_losses, truths = [], [], []
test_dataset.neg_samples=True # need negative samples now

for idx in trange(len(test_dataset)):
    input_data = test_dataset[idx]
    images, names, offsets, set_ids, labels, is_compat = input_data
    lengths = torch.tensor([len(labels)]).to(device) # 1, 2, 3 predicts 2, 3, 4, so length-1
    image_seqs = images.to(device)

    with torch.no_grad():
        f_output, b_output = model.extract(image_seqs, lengths)

    f_score = torch.matmul(f_output, test_features_matrix.t())
    b_score = torch.matmul(b_output, test_features_matrix.t())

    f_targets = [test_features_ids.index(i) for i in labels[1:]] # (2, 3, 4, 5) 
    b_targets = [test_features_ids.index(i) for i in labels[:-1][::-1]] # (4, 3, 2, 1)
    f_loss = criterion(f_score, torch.tensor(f_targets).to(device))
    b_loss = criterion(b_score, torch.tensor(b_targets).to(device))

    f_losses.append(f_loss.item())
    b_losses.append(b_loss.item())
    truths.append(is_compat)

f_losses, b_losses, truths = np.array(f_losses), np.array(b_losses), np.array(truths)
f_auc = metrics.roc_auc_score(truths, -f_losses)
b_auc = metrics.roc_auc_score(truths, -b_losses)
all_auc = metrics.roc_auc_score(truths, -f_losses-b_losses)
print('F_AUC: {:.4f}, B_AUC: {:.4f}, ALL_AUC: {:.4f}'.format(f_auc, b_auc, all_auc))
# F_AUC: 0.6587, B_AUC: 0.6656, ALL_AUC: 0.6651

# Fill in the blank test
criterion = nn.CrossEntropyLoss(reduction='none')
is_correct_f = []
is_correct_b = []
is_correct_all = []
for idx in trange(len(test_dataset)):
    items, labels, question_part, question_id, options, option_labels= test_dataset.get_fitb_quesiton(idx)
    lengths = torch.tensor([len(labels) for _ in range(4)]).to(device) # 4 options
    substitute_part = labels.index(question_id)

    images = [items]
    for option in options:
        new_outfit = items.clone()
        new_outfit[substitute_part] = option
        images.append(new_outfit)
    image_seqs = torch.cat(images).to(device)

    with torch.no_grad():
        f_output, b_output = model.extract(image_seqs, lengths)

    f_score = torch.matmul(f_output, test_features_matrix.t())
    b_score = torch.matmul(b_output, test_features_matrix.t())

    # construct target item id for each option
    # The order of targets in a batch should follow the rule of
    # torch.nn.utils.rnn.pack_padded_sequence
    f_targets = []
    b_targets = []
    option_labels = [labels[substitute_part]] + option_labels
    for i in range(1, len(labels)):
        for j in range(4):
            if i == substitute_part:
                f_targets.append(option_labels[j])
            else:
                f_targets.append(labels[i])
    for i in range(len(labels)-2, -1, -1):
        for j in range(4):
            if i == substitute_part:
                b_targets.append(option_labels[j])
            else:
                b_targets.append(labels[i])

    # Transform id to index
    f_targets = [test_features_ids.index(i) for i in f_targets] # (2, 3, 4, 5) 
    b_targets = [test_features_ids.index(i) for i in b_targets] # (4, 3, 2, 1)
    f_loss = criterion(f_score, torch.tensor(f_targets).to(device))
    b_loss = criterion(b_score, torch.tensor(b_targets).to(device))
    f_loss = f_loss.reshape(-1, 4)
    b_loss = b_loss.reshape(-1, 4)
    f_loss = f_loss.sum(dim=0)
    b_loss = b_loss.sum(dim=0)
 
    if f_loss.argmin().item() == 0:
        is_correct_f.append(True)
    else:
        is_correct_f.append(False)

    if b_loss.argmin().item() == 0:
        is_correct_b.append(True)
    else:
        is_correct_b.append(False)

    all_loss = f_loss + b_loss
    if all_loss.argmin().item() == 0:
        is_correct_all.append(True)
    else:
        is_correct_all.append(False)

print("F_ACC: {:.4f}, B_ACC: {:.4f}, All_ACC: {:.4f}".format(
    sum(is_correct_f) / len(is_correct_f),
    sum(is_correct_b) / len(is_correct_b),
    sum(is_correct_all) / len(is_correct_all)
))
# F_ACC: 0.4539, B_ACC: 0.4474, All_ACC: 0.4507
