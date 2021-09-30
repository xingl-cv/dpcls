import os
from datetime import datetime 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tools.trainer import trainerize
from tools.valider import validate
from tools.evaluate import get_accuracy
from tools.plot import plot_losses

from models.lenet import lenet5
from models.alexnet import alexnet
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.mobilenet import mobilenetv1, mobilenetv2
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

from dataset.custom_dataset import CustomDataset


# Parameters
RANDOM_SEED = 42

MODEL_CHOICE = {
    'lenet5':lenet5,
    'alexnet':alexnet,
    'vgg11':vgg11,
    'vgg11_bn':vgg11_bn,
    'vgg13':vgg13,
    'vgg13_bn':vgg13_bn,
    'vgg16':vgg16,
    'vgg16_bn':vgg16_bn,
    'vgg19':vgg19,
    'vgg19_bn':vgg19_bn,
    'mobilenetv1': mobilenetv1,
    'mobilenetv2': mobilenetv2,
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet152':resnet152,
    'resnext50_32x4d':resnext50_32x4d,
    'resnext101_32x8d':resnext101_32x8d,
    'wide_resnet50_2':wide_resnet50_2,
    'wide_resnet101_2':wide_resnet101_2,

    }

GPU_ID = 1 # 0,1,2,3 or -1'
DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != -1 else torch.device('cpu')

IMG_SIZE = 224

DATASET = 'flowers'
DATASET_PATH = f'data/{DATASET}'
TRAIN_DATASET_PATH = f'{DATASET_PATH}/train'
VAILD_DATASET_PATH = f'{DATASET_PATH}/val'
CLASSES = dict(map(reversed, enumerate(sorted(os.listdir(TRAIN_DATASET_PATH)))))
N_CLASSES = len(CLASSES)

N_EPOCHS = 12
BATCH_SIZE = 2
LEARNING_RATE = 0.001

PRETRAINED = True
MODEL_NAME = 'resnet18'
MODEL = MODEL_CHOICE[MODEL_NAME]
CKPT_DIR = f'weights/{MODEL_NAME}'
CKPT_PATH = f'{CKPT_DIR}/{DATASET}.pt'

def main():
    # Data
    train_transform = transforms.Compose([transforms.CenterCrop(IMG_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          ])

    valid_transform = transforms.Compose([transforms.CenterCrop(IMG_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          ])

    train_dataset = CustomDataset(root=TRAIN_DATASET_PATH, transform=train_transform, img_size=IMG_SIZE)
    valid_dataset = CustomDataset(root=VAILD_DATASET_PATH, transform=valid_transform, img_size=IMG_SIZE)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
    # Model
    torch.manual_seed(RANDOM_SEED)
    model = MODEL(pretrained=PRETRAINED, num_classes=N_CLASSES)
    model.to(DEVICE)
    
    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # Loss
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    print(f'{datetime.now().time().replace(microsecond=0)} --- Start training!\n')
    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    
    for epoch in range(0, epochs):
        print(f'{datetime.now().time().replace(microsecond=0)} --- Epoch: {epoch}\t')
        # training
        # print(f'{datetime.now().time().replace(microsecond=0)} --- Epoch: {epoch}   Trainning...')
        model, optimizer, train_loss = trainerize(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        # print(f'{datetime.now().time().replace(microsecond=0)} --- Epoch: {epoch}   Validating...')
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            # print(f'{datetime.now().time().replace(microsecond=0)} --- Epoch: {epoch}   Calculate Train Accuracy...')
            train_acc = get_accuracy(model, train_loader, device=device, type='Train')
            # print(f'{datetime.now().time().replace(microsecond=0)} --- Epoch: {epoch}   Calculate Valid Accuracy...')
            valid_acc = get_accuracy(model, valid_loader, device=device, type='Valid')

            print(f'{datetime.now().time().replace(microsecond=0)} --- Epoch: {epoch}\t'   
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}\n')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


if __name__ == "__main__":
    main()