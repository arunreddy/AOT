from glob import glob
import os
from otsc.feat.bag_of_visual_words import BagOfVisualWords

class RectagleImages(object):

    def __init__(self, config = None, n_images=1000):
        self.config = config
        self.n_images = n_images

    def read_image_paths(self):

        IMG_DIR = '/media/d1/data/img-rectangles/imgs'
        bad_images = '/media/d1/data/img-rectangles/imgs/bad_images.csv'

        bad_images_list = []
        for bad_image in open(bad_images,'r').readlines():
            bad_images_list.append(bad_image.strip())

        # Fetch the paths to images.
        zero_imgs = []
        for img_0 in glob(os.path.join(IMG_DIR,'0','*.jpg')):
            img = os.path.basename(img_0)
            if img not in bad_images_list:
                zero_imgs.append(img_0)

        one_imgs = []
        for img_1 in glob(os.path.join(IMG_DIR, '1', '*.jpg')):
            img = os.path.basename(img_1)
            if img not in bad_images_list:
                one_imgs.append(img_1)

        return zero_imgs[:self.n_images], one_imgs[:self.n_images]


if __name__ == '__main__':

    obj = RectagleImages()

    zero_imgs, one_imgs = obj.read_image_paths()

    bovw = BagOfVisualWords()
    bovw.train_model(images=[zero_imgs,one_imgs])



