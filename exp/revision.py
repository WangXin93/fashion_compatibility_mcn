import itertools
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn import metrics

import resnet
from utils import prepare_dataloaders
from diagnosis import *

warnings.filterwarnings('ignore')
plt.rc('font',family='Times New Roman')

def retrieve_sub(x, select, order):
    """ Retrieve the datset to substitute the worst item for the best choice.
    """
    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}
   
    best_score = -1
    best_img_path = dict()

    for o in order:
        if best_score > 0.9:
            break
        problem_part_idx = select[o]
        problem_part = all_names[problem_part_idx]
        for outfit in test_dataset.data:
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img).to(device)
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    score, *_ = model._compute_score(x)
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
    
        print('problem_part: {}'.format(problem_part))
        print('best substitution: {}'.format(best_img_path[problem_part]))
        print('After substitution the score is {:.4f}'.format(best_score))
        plt.imshow(plt.imread(best_img_path[problem_part]))
        plt.gca().axis('off')
        plt.show()
    
    show_imgs(x[0], select)
    return best_score, best_img_path

if __name__ == "__main__":
    # Dataloader
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        prepare_dataloaders()
    )
    iterator = iter(test_loader)
    batch = next(iterator)
    lengths, images, names, offsets, set_ids, labels, is_compat = batch

    # Load model weights
    from model import CompatModel
    device = torch.device("cuda:0")
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)
    model.load_state_dict(torch.load('./model_train_relation_vse_type_cond_scales.pth'))
    model.eval()
    
    for idx in [i for i, e in enumerate(is_compat) if e==0]:
        print("="*88)
        print("ID: {}".format(labels[idx]))
        x = images[idx].to(device).unsqueeze(0)
        select = [i for i, l in enumerate(labels[idx]) if 'mean' not in l]

        # Step 1: show images in an outfit
        show_imgs(x[0], select)

        # Step 2: show diagnosis results
        relation, out = defect_detect(x, model)
        relation = relation.squeeze().cpu().data
        show_rela_diagnosis(relation, select, cmap=plt.cm.Blues)
        result, order = item_diagnosis(relation, select)
        print("Predicted Score: {:.4f}\nProblem value of each item: {}\nOrder: {}\n".format(out, result, order))

        # Step 3: substitute the problem items for revision
        # best_score, best_img_path = retrieve_sub(x, select, order)
        print("="*88)