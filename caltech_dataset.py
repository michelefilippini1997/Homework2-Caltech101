from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from math import ceil

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
        self.samples = []
        self.class_indexes = {}
        
        dir_root = root.split('/')[0]
        file = open(f"{dir_root}/{split}.txt", 'r')

        class_count = 0
        
        for line in file.read().splitlines():
            if not line.startswith("BACKGROUND_Google"):
                class_name, image_path = line.split("/")[0], line
            
                if class_name not in self.class_indexes:
                    self.class_indexes[class_name] = class_count
                    class_count = class_count + 1
            
                sample = pil_loader(root + "/" + image_path), self.class_indexes[class_name]
                
                self.samples.append(sample)
        
    def stratified_subsets(self, percentage):
        
        if not (0 <= percentage <= 1):
            raise ValueError()
            
        first_split = []
        second_split = []
        
        index_by_class = {}
        
        for i in range(len(self.samples)):
            
            class_index, sample_index = self.samples[i][1], i
            
            if class_index not in index_by_class:
                index_by_class[class_index] = []
                
            index_by_class[class_index].append(sample_index)
            
        for key in index_by_class:
            class_indexes = index_by_class[key]
            
            n_first_split = ceil(len(class_indexes) * percentage)
            
            for i in range(len(class_indexes)):
                if i < n_first_split:
                    first_split.append(class_indexes[i])
                else:
                    second_split.append(class_indexes[i])
        
        return first_split, second_split
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.samples[index]
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return length
