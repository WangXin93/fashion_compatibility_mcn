import csv
import gzip
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class CategoryDataset(Dataset):
    """Dataset for polyvore with 5 categories(upper, bottom, shoe, bag, accessory),
    each suit has at least 3 items, the missing items will use a mean image.

    Args:
        root_dir: Directory stores source images
        data_file: A json file stores each outfit index and description
        data_dir: Directory stores data_file and mean_images
        transform: Operations to transform original images to fix size
        use_mean_img: Whether to use mean images to fill the blank part
        neg_samples: Whether generate negative sampled outfits
    """
    def __init__(self,
                 root_dir="../data/images/",
                 data_file='train_no_dup_with_category_3more_name.json',
                 data_dir="../data",
                 transform=None,
                 use_mean_img=True,
                 neg_samples=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]
        self.neg_samples = neg_samples # if True, will randomly generate negative outfit samples
    
        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open(os.path.join(self.data_dir, 'final_word_dict.txt')) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)


    def __getitem__(self, index):
        """It could return a positive suits or negative suits"""
        set_id, parts = self.data[index]
        if random.randint(0, 1) and self.neg_samples:
            #to_change = random.sample(list(parts.keys()), 3) # random choose 3 negative items
            to_change = list(parts.keys()) # random choose negative items
        else:
            to_change = []
        imgs = []
        labels = []
        names = []
        for part in ['upper', 'bottom', 'shoe', 'bag', 'accessory']:
            if part in to_change: # random choose a image from dataset with same category
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = os.path.join(self.root_dir, str(choice[0]), str(choice[1][part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
            elif part in parts.keys():
                img_path = os.path.join(self.root_dir, str(set_id), str(parts[part]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(parts[part]['name'])))
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                names.append(torch.LongTensor([])) # mean_img embedding
                labels.append('{}_{}'.format(part, 'mean'))
            else:
                continue
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        input_images = torch.stack(imgs)
        is_compat = (len(to_change)==0)

        offsets = list(itertools.accumulate([0] + [len(n) for n in names[:-1]]))
        offsets = torch.LongTensor(offsets)
        return input_images, names, offsets, set_id, labels, is_compat

    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]

    def get_fitb_quesiton(self, index):
        """Generate fill in th blank questions.
        Return:
            images: 5 parts of a outfit
            labels: store if this item is empty
            question_part: which part to be changed
            options: 3 other item with the same category,
                expect original composition get highest score
        """
        set_id, parts = self.data[index]
        question_part = random.choice(list(parts))
        question_id = "{}_{}".format(set_id, parts[question_part]['index'])
        imgs = []
        labels = []
        for part in ['upper', 'bottom', 'shoe', 'bag', 'accessory']:
            if part in parts.keys():
                img_path = os.path.join(self.root_dir, str(set_id), str(parts[part]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                labels.append('{}_{}'.format(part, 'mean'))
        items = torch.stack(imgs)

        option_ids = [set_id]
        options = []
        option_labels = []
        while len(option_ids) < 4:
            option = random.choice(self.data)
            if (option[0] in option_ids) or (question_part not in option[1]):
                continue
            else:
                option_ids.append(option[0])
                img_path = os.path.join(self.root_dir, \
                                        str(option[0]), \
                                        str(option[1][question_part]['index'])+'.jpg')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                options.append(img)
                option_labels.append("{}_{}".format(option[0], option[1][question_part]['index']))

        # Return 4 options for question, 3 incorrect options
        return items, labels, question_part, question_id, options, option_labels

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images,  names, offsets, set_ids, labels, is_compat = zip(*data)
    lengths = [i.shape[0] for i in images]
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    offsets = list(offsets)
    images = torch.stack(images)
    return (
        lengths,
        images,
        names,
        offsets,
        set_ids,
        labels,
        is_compat
    )

def lstm_collate_fn(data):
    """Need custom a collate_fn for LSTM DataLoader
    Batch images will be transformed to a long sequence.
    """
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images,  names, offsets, set_ids, labels, is_compat = zip(*data)
    lengths = [i.shape[0] for i in images]
    names = sum(names, [])
    offsets = list(offsets)
    images = torch.cat(images)
    return (
        lengths,
        images,
        names,
        offsets,
        set_ids,
        labels,
        is_compat
    )


class TripletDataset(Dataset):
    """Dataset will generate triplet to train conditional similarity network. Each 
     element in dataset should be anchor image, positive image, negative image and condition.
     
     Args:
         root_dir: Image directory
         data_file: A file record all outfit id and items
         data_dir: Directory which save mean image and data_file
         transform:
         is_train: Train phase will genrate triplet and condition, Evaluate phase will generate
             pair, condition and target.
     """
    def __init__(self,
             root_dir="/export/home/wangx/datasets/polyvore-dataset/images/",
             data_file='train_no_dup_with_category_3more_name.json',
             data_dir="../data",
             transform=None,
             is_train=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]
        self.is_train = is_train
        self.conditions = {
            'upper_bottom': 0,
            'bottom_upper': 0,
            'upper_shoe': 1,
            'shoe_upper': 1,
            'upper_bag': 2,
            'bag_upper': 2,
            'upper_accessory': 3,
            'accessory_upper': 3,
            'bottom_shoe': 4,
            'shoe_bottom': 4,
            'bottom_bag': 5,
            'bag_bottom': 5,
            'bottom_accessory': 6,
            'accessory_bottom': 6,
            'shoe_bag': 7,
            'bag_shoe': 7,
            'shoe_accessory': 8,
            'accessory_shoe': 8,
            'bag_accessory': 9,
            'accessory_bag': 9,
        }
        
    def load_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img
        
    def __getitem__(self, index):
        set_id, parts = self.data[index]
        choice = random.sample(list(parts.keys()), k=2)
        anchor_part, pos_part = choice[0], choice[1]
        anchor_img_path = os.path.join(self.root_dir, str(set_id), str(parts[anchor_part]['index'])+'.jpg')
        pos_img_path = os.path.join(self.root_dir, str(set_id), str(parts[pos_part]['index'])+'.jpg')

        neg_choice = self.data[index]
        while (pos_part not in neg_choice[1]) or (neg_choice[0] == set_id):
            neg_choice = random.choice(self.data)
        neg_img_path = os.path.join(self.root_dir, str(neg_choice[0]), str(neg_choice[1][pos_part]['index'])+'.jpg')

        pos_img = self.load_img(pos_img_path)
        anchor_img = self.load_img(anchor_img_path)
        neg_img = self.load_img(neg_img_path)

        condition = self.conditions['_'.join(choice)]
        
        if self.is_train:
            return anchor_img, pos_img, neg_img, condition
        else:
            target = random.randint(0, 1)
            if target == 0:
                return anchor_img, pos_img, target, condition
            elif target == 1:
                return anchor_img, neg_img, target, condition
            
    def __len__(self):
        return len(self.data)


# Test the loader
if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    d = CategoryDataset(transform=transform, use_mean_img=True)
    loader = DataLoader(d, 4, shuffle=True, num_workers=4, collate_fn=collate_fn)
