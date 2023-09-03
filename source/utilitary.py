import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as pltmg
import csv
import os
from random import randrange
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
                                    int(row["bbox_y"]) + int(row["bbox_height"]),
                                    label=rows.iloc[0]["label_name"]))

    bb_list.append(BoundingBox(0, 0, int(row["image_width"]), int(row["image_height"]), label="carte"))
    bb_list.reverse()
    return bb_list

def shift_bbs(bbs, x, y):
    bb_list = []
    for bb in bbs:
        bb_list.append(BoundingBox(bb.x1_int+x, bb.y1_int+y, bb.x2_int+x, bb.y2_int+y, bb.label))
    return bb_list

def random_texture(rootdir):
    listdir = os.listdir(path=rootdir+'textures')
    randcat = listdir[randrange(len(listdir))]
    while randcat == '.directory':
        randcat = listdir[randrange(len(listdir))]
    texlist = os.listdir(path=rootdir+'textures/'+randcat)
    randtext = texlist[randrange(len(texlist))]
    while randtext == '.directory':
        randtext = texlist[randrange(len(texlist))]

    textpath = rootdir+'textures/'+randcat+'/'+randtext
    return Image.open(textpath)

def foreground_card(rootdir, row_nb, df):
    ia.seed(randrange(1000))
    
    image_name = df.iloc[row_nb]["image_name"]
    image = imageio.imread(rootdir + "photo-tarot/" + image_name)

    mBlack = (image[:, :, 0:3] == [0,0,0]).all(2)
    image[mBlack] = (1,1,1)

            
    bb_list = extract_bb(image_name, df)
    bbs = BoundingBoxesOnImage(bb_list, shape=image.shape)

    # Define transformations windows
    seq = iaa.Sequential([
        iaa.Multiply((1, 1.5)), # change brightness, doesn't affect BBs
        iaa.Resize((0.25,0.3)),
        iaa.Affine(
            rotate=(-180, 180),
            shear=(-30, 30),
            mode = "constant",
            scale=(0.45, 0.52))])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Create mask on newly created pixels
    h, w = image_aug.shape[:2]
    
    rgba = np.dstack((image_aug, np.zeros((h,w),dtype=np.uint8)+255))
    mBlack = (rgba[:, :, 0:3] == [0,0,0]).all(2)
    rgba[mBlack] = (0,255,0,0)

    #return label and all
    
    return Image.fromarray(np.uint8(rgba)).convert('RGBA'), bbs_aug

def detect_overlap(labels, carte, newlabs):

    for label in labels:
        check = carte.intersection(label, default="no_overlap")
        if check != "no_overlap" and check.area > 35:
            labels.remove(label)

    for lab in newlabs:
        labels.append(lab)

    return labels

def remove_data_saved(dir):
    try:
        shutil.rmtree(dir)
        print("directory is removed successfully")
    except OSError as x:
        print("Error occured: %s : %s" % (dir, x.strerror))

def save_dataset(nb, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(dir+"/images")

    with open(dir+"labels.csv", 'w', newline='') as f:
        header = "image,label,x1,y1,x2,y2"
        f.write(row+'\n')
        for i in range(nb):
            print("Image n°"+str(i+1)+"/"+str(nb))
            img, labels = augment_image("Data/", randrange(1,3), "labels_tarot-bb_2023-05-29-09-58-51.csv")
            imgname = str(i)+'.png'
            pltmg.imsave(dir+"images/"+imgname, img)
            for bb in labels:
                row = imgname
                row+= ","+str(bb.label)+","+str(bb.x1)+","+str(bb.y1)+","+str(bb.x2)+","+str(bb.y2)
                f.write(row+'\n')


def augment_image(rootdir, nb_carte, csv_name):

    labels = []
    bb_carte = None
    df = pd.read_csv(rootdir + csv_name)
    nb_row = df.shape[0]

    background = random_texture(rootdir)
    while background.size[1] < 420:
        background = random_texture(rootdir)

    for i in range(nb_carte):
        foreground, bbs_aug = foreground_card(rootdir, randrange(nb_row), df)
        w, h = foreground.size

        print("texture height : " + str(background.size[1]) + " card height : "+ str(h))

        shift_x = randrange(background.size[0] - w)
        if background.size[1] - (h/5) >= 0:
            shift_y = randrange(background.size[1] - h)
        else:
            shift_y = 0

        background.paste(foreground, (shift_x, shift_y), foreground)   
        background = np.asarray(background)

        bbs = BoundingBoxesOnImage(shift_bbs(bbs_aug, shift_x, shift_y), shape=background.shape)
        bb_carte = bbs[0]
        new_labels = []
        for label in bbs[1:]:
            new_labels.append(label)

        labels = detect_overlap(labels, bb_carte, new_labels)
        background = Image.fromarray(np.uint8(background))

    background = np.asarray(background)

    # for bb in labels:
    #     background = bb.draw_on_image(background, size=1, color=[0, 0, 255])
    #     print(bb)
    return background, labels