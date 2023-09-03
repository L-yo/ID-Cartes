import source.utilitary as uti
import source.custom_dataset as custom_data
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# uti.remove_data_saved("Data/custom_generated_V1/")
# uti.save_dataset(10000, "Data/custom_generated_V1/")

tarot_dataset = custom_data.TarotDataset(root_dir='Data/custom_generated_V1/',
                                            csv_file='labels.csv', 
                                            label_dict='label_dict.csv',
                                            transform=custom_data.ToTensor())


# test = tarot_dataset.__getitem__(70)
# print(test)
tarot_dataset.show_image(125)

dataloader = DataLoader(tarot_dataset, batch_size=1, shuffle=True, num_workers=2)
