import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
from IPython.display import display, clear_output

model_file = 'mnist_model.pth'
def create_model():
    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))
    return model

model_file = 'mnist_model.pth'

def load_model():
    model = create_model()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

model = create_model()

if os.path.exists(model_file):
    print("Loading trained model...")
    model = load_model()
else:
    print("Training model...") # malli tallennetaan

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                  ])

    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    training_losses2 = []
    epochs = 5 # epoch arvo määrittää, montako kertaa datasetti käydään läpi. Mitä useamman kierrosta mallia opetetaan, sitä tarkemmin se tunnistaa numeron
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            training_loss = running_loss/len(trainloader)
            print(f"Training loss: {training_loss}")
            training_losses2.append(training_loss)

    torch.save(model.state_dict(), model_file)
def view(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
def run(b):
    clear_output(wait=True)
    display(button)

    images, labels = next(iter(trainloader))
    img = images[0].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    view(img.view(1, 28, 28), ps)
    print("Label:", labels[0].item())

button = widgets.Button(description="run")
button.on_click(run)
display(button)