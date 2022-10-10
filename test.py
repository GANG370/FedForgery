#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from tqdm import tqdm
import torch
import cv2
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch.nn.functional as F
from options import args_parser
from src.models import CVAE_imagenet
import copy
from PIL import Image
from torchvision import datasets, transforms
import os
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import time
import pickle
from tensorboardX import SummaryWriter
from src import models
import sys
sys.path.append('.')

class Logger(object):
    def __init__(self, filename='test.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
def test_inference(args, model, test_dataset):
        """ Returns the test accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = 'cuda' if args.gpu else 'cpu'
   
        criterion = nn.CrossEntropyLoss().to(device) 
        testloader = DataLoader(test_dataset, batch_size=32,
                                shuffle=False) 

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)

            batch_loss = criterion(outputs[0], labels)  
            loss += batch_loss.item()

            # Prediction
 
            _, pred_labels = torch.max(outputs[0], 1)  
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

sys.stdout = Logger(stream=sys.stdout)
class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        self.transforms = transforms
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            self.imgs = imgs
        else:
            self.imgs = imgs 
    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = 1 if 'real' in img_path.split('/')[-1] else 0  
        else:
            label = 1 if 'real' in img_path.split('/')[-1] else 0  
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
if __name__ == '__main__':
    args = args_parser()
    transform = transforms.Compose([
        transforms.Resize((296, 296)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((296, 296)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])   
    test_dataset = DogCat(r'testset', transforms=transform_test, train=False)# testset

    # BUILD MODEL
    if args.dataset == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model =='FedForgery':
         if args.dataset == 'forgery_dataset':
            global_model = CVAE_imagenet(d=64, k=128, num_classes=2)
    else:
        exit('Error: unrecognized model')

    path = 'pretrained/model.pth'
    global_model.load_state_dict(torch.load(path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
