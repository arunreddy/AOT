import os
from glob import glob

import joblib

from otsc.feat.bag_of_visual_words import BagOfVisualWords


class ConvexImages(object):
    def __init__(self, config=None, n_images=1500):
        self.config = config
        self.n_images = n_images

    def read_image_paths(self):

        IMG_DIR = '/media/d1/data/img-convex-nonconvex/imgs'
        bad_images = '/media/d1/data/img-convex-nonconvex/imgs/bad_images.csv'

        bad_images_list = []
        for bad_image in open(bad_images, 'r').readlines():
            bad_images_list.append(bad_image.strip())

        # Fetch the paths to images.
        zero_imgs = []
        for img_0 in glob(os.path.join(IMG_DIR, '0', '*.jpg')):
            img = os.path.basename(img_0)
            if img not in bad_images_list:
                zero_imgs.append(img_0)

        one_imgs = []
        for img_1 in glob(os.path.join(IMG_DIR, '1', '*.jpg')):
            img = os.path.basename(img_1)
            if img not in bad_images_list:
                one_imgs.append(img_1)

        return zero_imgs[self.n_images:self.n_images + 500], one_imgs[self.n_images:self.n_images + 500]

    def load_data(self):

        A = joblib.load('/home/arun/code/github/otsc/data/cache/convex_images_100.dat')
        B = joblib.load('/home/arun/code/github/otsc/data/cache/convex_images_aux_100.dat')

        return A[0], A[1], B[0], B[1]


if __name__ == '__main__':
    obj = ConvexImages()

    zero_imgs, one_imgs = obj.read_image_paths()

    print('Total # of images: ', (len(zero_imgs) + len(one_imgs)))

    bovw = BagOfVisualWords()
    n_clusters = 100
    bovw.train_model(images=[zero_imgs, one_imgs],
                     out_file_name='/home/arun/code/github/otsc/data/cache/convex_images_aux_%d.dat' % n_clusters,
                     n_clusters=n_clusters)
