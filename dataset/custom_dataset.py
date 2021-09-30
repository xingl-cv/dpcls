import glob
import os
import os.path as osp
import cv2 as cv
from PIL import Image 
from torch.utils.data import Dataset

        
def get_img_info(data_dir):
    data_info = []
    dirs = sorted(os.listdir(data_dir))
    classes = dict(map(reversed, enumerate(dirs)))
    for sub_dir in dirs:
        for path in glob.glob(osp.join(data_dir, sub_dir, '*.*')):    
            data_info.append((path, classes[sub_dir]))
    return data_info


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, img_size=32):
        super().__init__()
        self.data_info = get_img_info(root)
        self.transform = transform
        self.img_size = img_size
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size)
                   
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_path, label = self.data_info[index]
        img = cv.imread(img_path)
        cv.resize(img, self.img_size)
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    train_dataset = CustomDataset(root='data/mnist/mnist_train')

