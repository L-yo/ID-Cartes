import source.utilitary as uti
# import source.number_detector as ndec
# import source.custom_dataset as cust_data

nb_carte = 3
uti.augment_image("Data/", nb_carte, "labels_tarot-bb_2023-05-29-09-58-51.csv")


ELIMINATE (0, 0, 0) PIXELS THAT GET EATEN BY THE MASK