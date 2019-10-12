import cv2
import numpy as np
import glob
import random


class Cute:

    def __init__(self):
        self.classes = {}
        '''
        classes: {
            classname: [filename] 
        }
        '''
        self.indexes = {}
        self.idx_to_class_name = []
        self.class_name_to_idx = {}

    def load_dir(self, dir_name, class_names=['cat', 'dog'], filename="*.jpg"):
        i = 0
        self.idx_to_class_name = class_names
        for class_name in class_names:
            self.indexes[class_name] = 0
            self.classes[class_name] = list(glob.glob(dir_name + "/" + class_name + "/" + filename))
            self.class_name_to_idx[class_name] = i
            i += 1
        return self

    def next(self, batch_size=16, height=128, width=128):
        labels = []
        images = []
        for i in range(batch_size):
            item = np.random.randint(len(self.classes))
            class_name = self.idx_to_class_name[item]
            label = item

            image = cv2.imread(self.classes[class_name][self.indexes[class_name]])[:, :, ::-1]
            image = cv2.resize(image, (width, height))
            image = image/255
            labels.append(label)
            images.append(image)
            self.indexes[class_name] = (self.indexes[class_name] + 1) % len(self.classes[class_name])
            if self.indexes[class_name] == 0:
                random.shuffle(self.classes[class_name])
        # return: (BS, n_classes), (BS, H, W, 3)
        return np.asarray(labels), np.asarray(images)
