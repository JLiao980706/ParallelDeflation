import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# train_dir = "ILSVRC/Data/CLS-LOC/train"

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Lambda(lambda x: x.view(-1)),  
#     transforms.Lambda(lambda x: x / 255.0)
# ])

class ImageNetTrainDataset(Dataset):
    def __init__(self, train_dir, transform):
        """
        train_dir : all train folders
        """
        super().__init__()
        self.image_paths = []
        class_names = sorted(os.listdir(train_dir))
        for class_name in tqdm(class_names):
            class_path = os.path.join(train_dir, class_name)
            img_names = sorted(os.listdir(class_path))
            for img_name in img_names:
                self.image_paths.append(os.path.join(class_path, img_name))
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path, ImageReadMode.GRAY)
        return self.transform(image)
    
    def __len__(self):
        return len(self.image_paths)

# dataset = ImageNetTrainDataset(train_dir, transform=transform)

class RandomBatchGetter:
    def __init__(self, dataset, batch_size):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    def getRandBatch(self):
        return np.array(next(iter(self.dataloader))) 




