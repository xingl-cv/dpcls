import os
import os.path as osp
import glob
import torch
import torch.nn.functional as F
from models.lenet import lenet5
import cv2 as cv
from PIL import Image 
from torchvision import transforms

from models.lenet import lenet5
from models.alexnet import alexnet
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.mobilenet import mobilenetv1, mobilenetv2
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2


RANDOM_SEED = 42

MODEL_CHOICE = {
    'lenet5':lenet5, # img_size: 32
    'alexnet':alexnet, # img_size = 227
    'vgg11':vgg11, # img_size = 224
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

GPU_ID = 2 # 0,1,2,3 or -1'
DEVICE = torch.device(f'cuda:{GPU_ID}') if GPU_ID != -1 else torch.device('cpu')

DATASET = 'flowers'
DATASET_PATH = f'data/{DATASET}'
TEST_DATASET_PATH = f'{DATASET_PATH}/test'
CLASSES = dict(enumerate(sorted(os.listdir(TEST_DATASET_PATH))))
N_CLASSES = len(CLASSES)

SINGLE = False
IMG_SIZE = 224
IMG_DIR = f'data/{DATASET}/test/{CLASSES[0]}'
IMG_PATH = f'data/mnist/test/1/mnist_test_2.png'

MODEL_NAME = 'resnet18'
MODEL = MODEL_CHOICE[MODEL_NAME]
CKPT_DIR = f'weights/{MODEL_NAME}'
CKPT_PATH = f'{CKPT_DIR}/{DATASET}.pt'
print(CKPT_PATH)

def main():   
    torch.manual_seed(RANDOM_SEED)
    model = MODEL(num_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    if DATASET == 'mnist':
        IMG_SIZE = 32
        test_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                            transforms.ToTensor()])

    else:
        test_transform = transforms.Compose([transforms.CenterCrop(IMG_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])

    if SINGLE:
        img = load_img(IMG_PATH, test_transform)
        label, conf = test(model, img, DEVICE)
        print(CLASSES[int(label)], f'{conf:.0f}%')
    
    else:
        img_path_list = glob.glob(osp.join(IMG_DIR, '*.*'))
        for img_path in img_path_list:
            img = load_img(img_path, test_transform)
        
            label, conf = test(model, img, DEVICE)
            print(CLASSES[int(label)], f'{conf:.0f}%')


def load_img(img_path, test_transform):
    img = cv.imread(img_path)
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    img = test_transform(img)
    img = img.unsqueeze(0)
    return img


def test(model, img, device):
    with torch.no_grad():
        model.eval()
        x = img.to(device)
        y_pred = model(x)

        y_prob = F.softmax(y_pred, dim=1)
        # _, pred_label = torch.max(y_prob, 1)
        label = torch.argmax(y_prob).cpu().numpy()
        conf = torch.max(y_prob * 100).cpu().numpy()
        # title = f'{torch.argmax(y_prob)} ({torch.max(y_prob * 100):.0f}%)'
        # print(title)
    return label, conf


if __name__ == "__main__":
    main()
