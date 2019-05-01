from torch.utils.data import Dataset
import glob
import random
import cv2


class Phase1DataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.imagePaths = []
        self.labels = []

    def load_images(self, dir, image_extension, label):
        examples = glob.glob(F"{dir}/*{image_extension}")
        examples = [path for path in examples if 'mask' not in path]
        self.imagePaths.extend(examples)
        self.labels.extend([label] * len(examples))

    def shuffle(self):
        combined = list(zip(self.imagePaths, self.labels))
        random.shuffle(combined)
        self.imagePaths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        return (self.transform(cv2.imread(self.imagePaths[index])), self.labels[index])
