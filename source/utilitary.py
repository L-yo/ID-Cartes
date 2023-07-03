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
import pandas as pd

def extract_bb(img_name, df):
    bb_list = []
    rows =  df[df["image_name"] == img_name] 
    for index, row in rows.iterrows():
        bb_list.append(BoundingBox(int(row["bbox_x"]), 
                                    int(row["bbox_y"]),  
                                    int(row["bbox_x"]) + int(row["bbox_width"]), 
                                    int(row["bbox_y"]) + int(row["bbox_height"])))

    label = rows.iloc[0]["label_name"]
    return bb_list, label

def shift_bbs(bbs, x, y):
    bb_list = []
    for bb in bbs:
        bb_list.append(BoundingBox(bb.x1_int+x, bb.y1_int+y, bb.x2_int+x, bb.y2_int+y))
    return bb_list

def random_texture(rootdir):
    listdir = os.listdir(path=rootdir+'/textures')
    randcat = listdir[randrange(len(listdir))]
    texlist = os.listdir(path=rootdir+'/textures/'+randcat)
    print(randcat)
    print(len(texlist))
    randtext = rootdir+'/textures/'+randcat+'/'+texlist[randrange(len(texlist))]
    
    return Image.open(randtext)

def foreground_card(rootdir, row_nb, df):
    ia.seed(randrange(1000))
    
    image_name = df.iloc[row_nb]["image_name"]
    image = imageio.imread(rootdir + "photo-tarot/" + image_name)


            
    bb_list, label = extract_bb(image_name, df)
    bbs = BoundingBoxesOnImage(bb_list, shape=image.shape)

    # Define transformations windows
    seq = iaa.Sequential([
        iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect BBs
        iaa.Resize((0.25,0.4)),
        iaa.Affine(
            rotate=(-180, 180),
            shear=(-40, 40),
            mode = "constant",
            scale=(0.4, 0.5))])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Create mask on newly created pixels
    h, w = image_aug.shape[:2]
    
    rgba = np.dstack((image_aug, np.zeros((h,w),dtype=np.uint8)+255))
    mBlack = (rgba[:, :, 0:3] == [0,0,0]).all(2)
    rgba[mBlack] = (0,255,0,0)

    #return label and all
    
    return Image.fromarray(np.uint8(rgba)).convert('RGBA'), bbs_aug


def augment_image(rootdir, nb_carte, csv_name):

    label = {}
    label["bbs"] = []
    df = pd.read_csv(rootdir + csv_name)
    nb_row = df.shape[0]

    background = random_texture(rootdir)

    for i in range(nb_carte):
        foreground, bbs_aug = foreground_card(rootdir, randrange(nb_row), df)
        w, h = foreground.size

        # pixels = foreground.load() # create the pixel map
        # for i in range(w): # for every pixel:
        #     for j in range(h):
        #         if pixels[i,j] == (255, 0, 0):
        #             pixels[i,j] = (254, 0 ,0)

        shift_x = randrange(background.size[0] - w)
        if background.size[1] - h > 0:
            shift_y = randrange(background.size[1] - h)
        else:
            shift_y = 0

        background.paste(foreground, (shift_x, shift_y), foreground)   
        background = np.asarray(background)

        bbs2 = BoundingBoxesOnImage(shift_bbs(bbs_aug, shift_x, shift_y), shape=background.shape)
        background = bbs2.draw_on_image(background, size=2, color=[0, 0, 255])
        background = Image.fromarray(np.uint8(background))
        
        label['bbs'].append(bbs2)

    label['img'] = background

    background = np.asarray(background)
    plt.imshow(background)
    plt.title(len(label['bbs']))
    plt.show()
