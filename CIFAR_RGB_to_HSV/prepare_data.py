import pickle
import numpy as np
from PIL import Image
import cv2

class ImageTransformer:

    def __init__(self, path):
        self.path = path


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            loaded_dict = pickle.load(fo, encoding='bytes')
            print(loaded_dict.keys())
        return loaded_dict


    def read_data(self): 
        batches = []
        for i in range(1, 6):
            batches.append(self.unpickle(self.path + 'data_batch_' + str(i)))
        return batches
        

    def rgb_to_hsl(self, data):
        transformed = []
        for batch in data:
            cur_batch = []
            cur_data = batch[b'data']
            cur_labels = batch[b'labels']
            for i in range(len(cur_data)):
                img=np.asarray(cur_data[i], dtype=np.uint8)
                hsv_img = cv2.cvtColor(img.reshape(32,32,3), cv2.COLOR_RGB2HSV)
                cur_batch.append(np.array(hsv_img).flatten())
            transformed.append({'data': cur_batch, 'labels': cur_labels})

        return transformed


if __name__ == "__main__":
    transformer = ImageTransformer('./cifar-10-batches-py/')
    data = transformer.read_data()
    transformed = transformer.rgb_to_hsl(data)
    for i in range(len(transformed)):
        with open('./hsv-cifar-10-batches-py/data_batch_' + str(i), 'wb+') as fo:
            pickle.dump(transformed[i], fo)
