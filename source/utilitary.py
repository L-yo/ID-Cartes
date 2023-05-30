import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import csv
from PIL import Image
import imageio.v2 as imageio 
import matplotlib.pyplot as plt
import numpy as np


def extract_bb(img_name, csv_name):
    with open(csv_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        bb_list = []
        for row in spamreader:
            if row[5] == img_name:
                bb_list.append(BoundingBox(int(row[1]), 
                                            int(row[2]),
                                            int(row[1]) + int(row[3]), 
                                            int(row[2]) + int(row[4])))
        return bb_list




ia.seed(1)

rootdir = "../Data/"
image_name = "IMG_20230529_134632.jpg"

image = imageio.imread(rootdir + "photo-tarot/" + image_name)

bbs = BoundingBoxesOnImage(extract_bb(image_name, rootdir + 'labels_tarot-bb_2023-05-29-09-58-51.csv'),
                                        shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        rotate=(-45,45),
        mode = "constant"
        # scale=(0.2, 0.4)
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])


# Augment BBs and images.
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=4, color=[0, 0, 255])

h, w = image_after.shape[:2]

sem_map_after = np.dstack((image_after, np.zeros((h,w),dtype=np.uint8)+255))
mBlack = (sem_map_after[:, :, 0:3] == [0,0,0]).all(2)
sem_map_after[mBlack] = (0,255,0,0)


foreground = Image.fromarray(np.uint8(sem_map_after)).convert('RGBA')
background = Image.open(rootdir + "textures/braided/braided_0006.jpg")
#resize, first image
foreground = foreground.resize((180, 240))
foreground_size = foreground.size
background_size = background.size

background.paste(foreground, (100,100), foreground)

plt.imshow(np.asarray(background))
plt.show()

