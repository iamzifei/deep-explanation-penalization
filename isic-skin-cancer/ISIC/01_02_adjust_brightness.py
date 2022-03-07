import random
import torchvision.transforms as transforms
from PIL import Image
from os import listdir, mkdir
from os.path import isfile, join, isdir


image_path = "/Users/james/Dev/ai/master/data/ISIC/dark_subset/9"

ratio = 0.3


# get number of images in the folder image_path with ratio
def get_num_images(image_path, ratio):
    num_images = 0
    for f in listdir(image_path):
        if isfile(join(image_path, f)):
            num_images += 1
    return int(num_images * ratio)


random_files = random.sample(
    [f for f in list(listdir(image_path)) if isfile(join(image_path, f))], get_num_images(image_path, ratio))

for f in random_files:
    # make all images in random_files darker
    img = Image.open(join(image_path, f))
    img = transforms.ToTensor()(img)
    img = transforms.functional.adjust_brightness(img, random.uniform(0, 1))
    img = transforms.ToPILImage()(img)
    img.save(join(image_path, f))
