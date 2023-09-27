import math
import os
import shutil
import numpy as np
from PIL import Image

class DataLoader:
    def __init__(self, A_training_size, B_test_size, positive_dataset_path, negative_dataset_path):
        super(DataLoader, self).__init__()
        self.A_training_size = A_training_size
        self.B_test_size = B_test_size
        self.positive_dataset_path = os.path.abspath(positive_dataset_path)
        self.negative_dataset_path = os.path.abspath(negative_dataset_path)
        self.base_path = os.path.abspath('.')

    def create(self):
        os.mkdir("./Dataset")
        os.mkdir("./Dataset/Client-A")
        os.mkdir("./Dataset/Client-B")
        os.mkdir("./Dataset/Client-A/Training")
        os.mkdir("./Dataset/Client-A/Training/Positive")
        os.mkdir("./Dataset/Client-A/Training/Negative")
        os.mkdir("./Dataset/Client-A/Validation")
        os.mkdir("./Dataset/Client-A/Validation/Positive")
        os.mkdir("./Dataset/Client-A/Validation/Negative")
        os.mkdir("./Dataset/Client-B/Test")
        os.mkdir("./Dataset/Client-B/Test/Positive")
        os.mkdir("./Dataset/Client-B/Test/Negative")

    def load(self):
        os.chdir(self.positive_dataset_path)
        positive_image_list = os.listdir()
        positive_train_size = math.floor(self.A_training_size/2)
        positive_test_size = math.floor(self.B_test_size/2)
        positive_validation_size = max(10, math.floor(0.1*positive_train_size))
        start = 0
        end = positive_train_size

        for image in positive_image_list[start:end]:
            image_path = os.path.join(self.positive_dataset_path, image)
            final_path = os.path.join(os.path.join(self.base_path, './Dataset/Client-A/Training/Positive'), image)
            shutil.move(image_path, final_path)
        start = end
        end += positive_validation_size
        
        print(start, end)
        for image in positive_image_list[start:end]:
            image_path = os.path.join(self.positive_dataset_path, image)
            final_path = os.path.join(os.path.join(self.base_path, './Dataset/Client-A/Validation/Positive'), image)
            shutil.move(image_path, final_path)
        start = end
        end += positive_test_size

        for image in positive_image_list[start:end]:
            image_path = os.path.join(self.positive_dataset_path, image)
            final_path = os.path.join(os.path.join(self.base_path, './Dataset/Client-B/Test/Positive'), image)
            shutil.move(image_path, final_path)
        
        os.chdir(self.negative_dataset_path)
        negative_image_list = os.listdir()
        negative_train_size = math.ceil(self.A_training_size/2)
        negative_test_size = math.ceil(self.B_test_size/2)
        negative_validation_size = max(10, math.floor(0.1*negative_train_size))
        start = 0
        end = negative_train_size

        for image in negative_image_list[start:end]:
            image_path = os.path.join(self.negative_dataset_path, image)
            final_path = os.path.join(os.path.join(self.base_path, './Dataset/Client-A/Training/Negative'), image)
            shutil.move(image_path, final_path)
        start = end
        end += negative_validation_size

        for image in negative_image_list[start:end]:
            image_path = os.path.join(self.negative_dataset_path, image)
            final_path = os.path.join(os.path.join(self.base_path, './Dataset/Client-A/Validation/Negative'), image)
            shutil.move(image_path, final_path)        
        start = end
        end += negative_test_size

        for image in negative_image_list[start:end]:
            image_path = os.path.join(self.negative_dataset_path, image)
            final_path = os.path.join(os.path.join(self.base_path, './Dataset/Client-B/Test/Negative'), image)
            shutil.move(image_path, final_path)

        os.chdir(self.base_path)
        
class NPZ_DataLoader:
    def __init__(self, A_training_size, B_test_size, file_name):
        super(NPZ_DataLoader, self).__init__()
        self.A_training_size = A_training_size
        self.B_test_size = B_test_size

        file = np.load(file_name)
        train_images = file[file.files[0]]
        train_labels = file[file.files[1]]

        self.positive_images = [train_images[i] for i in range(len(train_labels)) if train_labels[i]==1]
        self.negative_images = [train_images[i] for i in range(len(train_labels)) if train_labels[i]==0]
    
    # Condition 1 : The data type of NumPy Array is float
    # Condition 2 : The NumPy Array is normalized to range (0,1)
    def normalize(self):
        if self.positive_images[0].dtype == 'float32' and np.max(self.positive_images[0])<=1.0:
            for i in range(len(self.positive_images)):
                self.positive_images[i] = self.positive_images[i]*255               # Bring the NumPy Array range to (0,255)
                self.positive_images[i] = self.positive_images[i].astype(np.uint8)  # Cast the NumPy Array to Unsigned Int
            for i in range(len(self.negative_images)):
                self.negative_images[i] = self.negative_images[i]*255               # Bring the NumPy Array range to (0,255)
                self.negative_images[i] = self.negative_images[i].astype(np.uint8)  # Cast the NumPy Array to Unsigned Int

    def create(self):
        os.mkdir("./Dataset")
        os.mkdir("./Dataset/Client-A")
        os.mkdir("./Dataset/Client-B")
        os.mkdir("./Dataset/Client-A/Training")
        os.mkdir("./Dataset/Client-A/Training/Positive")
        os.mkdir("./Dataset/Client-A/Validation")
        os.mkdir("./Dataset/Client-A/Validation/Positive")
        os.mkdir("./Dataset/Client-B/Training")
        os.mkdir("./Dataset/Client-B/Test/Positive")
        os.mkdir("./Dataset/Client-B/Test/Negative")
        
    def load(self):
        train_size = math.floor(self.A_training_size/2)
        test_size = math.floor(self.B_test_size/2)
        validation_size = max(10, math.floor(0.1*train_size))

        start = 0
        end = train_size
        ClientA_training_positive_images = self.positive_images[start:end]
        ClientA_training_negative_images = self.negative_images[start:end]

        start = end
        end += validation_size
        ClientA_validation_positive_images = self.positive_images[start:end]
        ClientA_validation_negative_images = self.negative_images[start:end]

        start = end
        end += test_size
        ClientB_training_positive_images = self.positive_images[start:end]
        ClientB_training_negative_images = self.negative_images[start:end]

        os.chdir("./Dataset/Client-A/Training/Positive")
        for i in range(len(ClientA_training_positive_images)):
            data = Image.fromarray(ClientA_training_positive_images[i])
            data.save(f'{i}.png')

        os.chdir("./Dataset/Client-A/Training/Negative")
        for i in range(len(ClientA_training_negative_images)):
            data = Image.fromarray(ClientA_training_negative_images[i])
            data.save(f'{i}.png')

        os.chdir("./Dataset/Client-A/Validation/Positive")
        for i in range(len(ClientA_validation_positive_images)):
            data = Image.fromarray(ClientA_validation_positive_images[i])
            data.save(f'{i}.png')

        os.chdir("./Dataset/Client-A/Validation/Negative")
        for i in range(len(ClientA_validation_negative_images)):
            data = Image.fromarray(ClientA_validation_negative_images[i])
            data.save(f'{i}.png')

        os.chdir("./Dataset/Client-B/Training/Positive")
        for i in range(len(ClientB_training_positive_images)):
            data = Image.fromarray(ClientB_training_positive_images[i])
            data.save(f'{i}.png')

        os.chdir("./Dataset/Client-B/Training/Negative")
        for i in range(len(ClientB_training_negative_images)):
            data = Image.fromarray(ClientB_training_negative_images[i])
            data.save(f'{i}.png')