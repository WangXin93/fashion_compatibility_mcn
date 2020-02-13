import itertools
import os
import warnings

import cv2
import matplotlib; matplotlib.use('agg')
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
        # plt.imshow(plt.imread(best_img_path[problem_part]))
        # plt.gca().axis('off')
        # plt.show()
    
    show_imgs(x[0], select, "revised_outfit.pdf")
    return best_score, best_img_path

if __name__ == "__main__":
    # Load model weights
    from model import CompatModel
    model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)
    model.load_state_dict(torch.load('./model_train_relation_vse_type_cond_scales.pth'))
    model.eval()
    
    print("="*80)
    # Comment different line to choose different outfit as example.
    ID = ['178118160_1', 'bottom_mean', '199285568_4', '111355382_5', '209432387_4']
    # ID = ['204421067_1', 'bottom_mean', '202412456_3', '214716404_4', '187592500_5']
    # ID = ['140106066_1', '139731278_2', '215327132_4', 'bag_mean', '211697041_4']
    # ID = ['108112189_1', '216678271_2', '200786021_3', 'bag_mean', 'accessory_mean']
    # ID = ['127389151_1', 'bottom_mean', 'shoe_mean', '190117110_4', '171755122_4']
    # ID = ['187950801_1', '198450014_1', '129931699_5', '136842112_3', 'accessory_mean']
    x = loadimg_from_id(ID).to(device)
    # kick out the mean images for padding the sequence when making visualization
    select = [i for i, l in enumerate(ID) if 'mean' not in l]

    print("Step 1: show images in an outfit...")
    show_imgs(x[0], select)

    print("\nStep 2: show diagnosis results...")
    relation, out = defect_detect(x, model)
    relation = relation.squeeze().cpu().data
    show_rela_diagnosis(relation, select, cmap=plt.cm.Blues)
    result, order = item_diagnosis(relation, select)
    print("Predicted Score: {:.4f}\nProblem value of each item: {}\nOrder: {}".format(out, result, order))

    print("\nStep 3: substitute the problem items for revision, it takes a while to search...")
    best_score, best_img_path = retrieve_sub(x, select, order)
    print("="*80)
