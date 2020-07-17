import argparse
import os
from PIL import Image

from datasets.coco.coco_dataset import CocoDataset


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))



if __name__ == '__main__':
    from coco_settings import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true', help='default settings for validation set')
    parser.add_argument('--image_size', type=int, default=resize_size, help='size for image after processing')
    args = parser.parse_args()

    args.image_dir = val_imdir if args.val else train_imdir
    args.output_dir = resized_val_imdir if args.val else resized_val_imdir

    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)
    cocodataset = CocoDataset.init(dataset_dir)

