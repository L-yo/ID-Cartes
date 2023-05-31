import os
import csv
from random import randrange
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio 
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
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

def shift_bbs(bbs, x, y):
    bb_list = []
    for bb in bbs:
        bb_list.append(BoundingBox(bb.x1_int+x, bb.y1_int+y, bb.x2_int+x, bb.y2_int+y))
    return bb_list

def random_texture(rootdir):
    listdir = os.listdir(path=rootdir+'/textures')
    randcat = listdir[randrange(len(listdir))]
    texlist = os.listdir(path=rootdir+'/textures/'+randcat)
    randtext = rootdir+'/textures/'+randcat+'/'+texlist[randrange(len(texlist))]
    
    return Image.open(randtext)

def augment_image(rootdir, image_name, csv_name):
    ia.seed(randrange(1000))
    
    image = imageio.imread(rootdir + "photo-tarot/" + image_name)
    bbs = BoundingBoxesOnImage(extract_bb(image_name, rootdir + csv_name), shape=image.shape)

    # Define transformations windows
    seq = iaa.Sequential([
        iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect BBs
        iaa.Resize((0.25,0.5)),
        iaa.Affine(
            rotate=(-180, 180),
            shear=(-50, 50),
            mode = "constant",
            scale=(0.2, 0.6))])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Create mask on newly created pixels
    h, w = image_aug.shape[:2]
    rgba = np.dstack((image_aug, np.zeros((h,w),dtype=np.uint8)+255))
    mBlack = (rgba[:, :, 0:3] == [0,0,0]).all(2)
    rgba[mBlack] = (0,255,0,0)

    # Pasting transformed card on random texture
    foreground = Image.fromarray(np.uint8(rgba)).convert('RGBA')
    background = random_texture(rootdir)
    background.paste(foreground, (50,50), foreground)   # RANDOM THIS TAKING INTO ACCOUNT 
                                                        # FOREGROUND SIZE TO NOT CUT CARD

    paste = np.asarray(background)

    bbs2 = BoundingBoxesOnImage(shift_bbs(bbs_aug, 50, 50), shape=paste.shape)
    img_final = bbs2.draw_on_image(paste, size=2, color=[0, 0, 255])

    plt.imshow(img_final)
    plt.show()
