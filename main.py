import source.utilitary as uti
import source.number_detector as ndec

import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

continued_network = ndec.Net()
network_state_dict = torch.load('results/model.pth')
continued_network.load_state_dict(network_state_dict)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('Data/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=40, shuffle=True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# with torch.no_grad():
#   output = continued_network(example_data)

#   pred = output.data.max(1, keepdim=True)[1][35].item()
#   fig = plt.figure()

#   plt.imshow(example_data[35][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}, Predicted: {}".format(example_targets[35], pred))
#   plt.show()


# TO DO : Try loading an external picture and predict it.
#         May need de define my own Dataset class, see more on pytorch


test_loader2 = torch.utils.data.DataLoader(data2, transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
  batch_size=40, shuffle=True)

# print(example_data[10][0].type)
