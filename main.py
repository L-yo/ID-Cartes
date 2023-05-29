import source.utilitary as uti
import source.number_detector as ndec
import source.custom_dataset as cust_data

import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



continued_network = ndec.Net()
network_state_dict = torch.load('results/model.pth')
continued_network.load_state_dict(network_state_dict)


continued_optimizer = optim.SGD(continued_network.parameters(), lr=0.01,
                                momentum=0.5)
optimizer_state_dict = torch.load('results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)


test_dataset = cust_data.TarotDataset(csv_file='Data/test/labels.csv',
                                    root_dir='Data/test/',
                                    transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Grayscale(),
                                                  torchvision.transforms.Resize([28, 28], antialias=True),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        

taille = len(test_dataset)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('Data/', train=False, download=False,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
batch_size=1, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


test = test_dataset[2]['image'].unsqueeze(1)
print(test.shape)

with torch.no_grad():
    
    output = continued_network(example_data) #switch with test variable
    print(output.data.max(1, keepdim=True)[1][0].item())
    plt.imshow(example_data[0][0], cmap='gray', interpolation='none')  #switch with test variable
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][0].item()))
    plt.show()