from glob import glob
import os
from otsc.feat.bag_of_visual_words import BagOfVisualWords
import joblib

class CatsAndDogs(object):

    def __init__(self, config = None, n_images=1500):
        self.config = config
        self.n_images = n_images

    def read_image_paths(self):

        IMG_DIR = '/media/d1/data/cats_dogs/train'
        BAD_IMAGES = '/media/d1/data/cats_dogs/train/bad_images.csv'

        # Fetch the paths to images.
        zero_imgs = []
        one_imgs = []
        for img in glob(os.path.join(IMG_DIR,'*.jpg')):
            img_file_name = os.path.basename(img)

            if img_file_name.startswith('cat'):
                zero_imgs.append(img)
            elif img_file_name.startswith('dog'):
                one_imgs.append(img)

        #return zero_imgs[:self.n_images], one_imgs[:self.n_images]
        return zero_imgs[self.n_images:self.n_images + 500], one_imgs[self.n_images:self.n_images + 500]


    def load_data(self):

        A = joblib.load('/home/arun/code/github/otsc/data/cache/cats_dogs_images_600.dat')
        B = joblib.load('/home/arun/code/github/otsc/data/cache/cats_dogs_images_aux_600.dat')

        return A[0], A[1], B[0], B[1]

if __name__ == '__main__':

    obj = CatsAndDogs()

    zero_imgs, one_imgs = obj.read_image_paths()

    bovw = BagOfVisualWords()
    n_clusters = 600
    bovw.train_model(images=[zero_imgs,one_imgs], out_file_name='/home/arun/code/github/otsc/data/cache/cats_dogs_images_aux_%d.dat'%n_clusters,n_clusters=n_clusters)





