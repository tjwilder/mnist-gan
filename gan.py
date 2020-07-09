import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '../data'
os.makedirs("images", exist_ok=True)

latent_dim = 100
image_shape = (28, 28)

g_iterations = 1
d_iterations = 1


class Generator(nn.Module):
  def __init__(self, input_size, output_size, hidden_layers):
    super(Generator, self).__init__()
    self.g_layers = []
    layer_size = input_size

    for layer in hidden_layers:
      self.g_layers.append(nn.Linear(layer_size, layer))
      self.g_layers.append(nn.BatchNorm1d(layer, 0.8))
      self.g_layers.append(nn.LeakyReLU(0.2))
      layer_size = layer

    self.g_layers.append(nn.Linear(layer_size, output_size))
    self.g_layers.append(nn.Tanh())
    # self.g_layers.append(nn.Sigmoid())
    self.g = nn.Sequential(*self.g_layers)

  def forward(self, z):
    outputs = self.g(z)

    return outputs.view(outputs.shape[0], 1, *image_shape)


class Discriminator(nn.Module):
  def __init__(self, input_size, output_size, hidden_layers):
    super(Discriminator, self).__init__()
    self.d_layers = []
    layer_size = input_size

    for layer in hidden_layers:
      self.d_layers.append(nn.Linear(layer_size, layer))
      self.d_layers.append(nn.LeakyReLU(0.2))
      self.d_layers.append(nn.Dropout(0.5))
      layer_size = layer

    self.d_layers.append(nn.Linear(layer_size, output_size))
    self.d_layers.append(nn.Sigmoid())
    self.d = nn.Sequential(*self.d_layers)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    outputs = self.d(x)

    return outputs


def load_dataset():
  global train_dataset, test_dataset, train_loader, test_loader
  batch_size = 100

  # MNIST dataset contains 60000 images in the training data and 10000 test
  # data images
  train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                             train=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     [0.5], [0.5])]),
                                             download=True)

  # Data loader divides the dataset into batches of batch_size=100 that
  # can be used for parallel computation on multi-processors
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)


def train(G, D, epochs=200, learning_rate=0.0002):
  criterion = nn.BCELoss()
  # criterion = nn.MSELoss()
  g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate,
                                 betas=(0.5, .99))
  d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate,
                                 betas=(0.5, .99))

  for i in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
      # labels = labels.to(device)
      valid = torch.ones((images.shape[0], 1))
      fake = torch.zeros((images.shape[0], 1))

      # Train the Generator
      for k in range(g_iterations):
        g_optimizer.zero_grad()

        z = torch.randn((images.shape[0], latent_dim))
        # z = Variable(torch.FloatTensor(
        #     np.random.normal(0, 1, (images.shape[0], latent_dim))))
        fake_outputs = G(z)
        fake_labels = D(fake_outputs)
        g_loss = criterion(fake_labels, valid)

        g_loss.backward()
        g_optimizer.step()

        _, best_labels = torch.topk(fake_labels, 25, dim=0)
        best_fakes = fake_outputs[best_labels]

        # print(torch.sum(fake_outputs[0] - fake_outputs[1]))

      # Train the Discriminator
      for k in range(d_iterations):
        d_optimizer.zero_grad()

        d_real_loss = criterion(D(images), valid)
        d_fake_loss = criterion(D(fake_outputs.detach()), fake)
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        d_optimizer.step()

      if (j + 1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], D_L: {:.4f}, G_L: {:.4f}'.format(
            i + 1, epochs, j + 1, len(train_loader),
            d_loss.item(), g_loss.item()))

      batches_done = i * len(train_loader) + j
      if batches_done % 100 == 0:
        save_image(best_fakes[:25].view(-1, 1, 28, 28),
                   "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == '__main__':
  load_dataset()
  G = Generator(latent_dim, 28 * 28, [128, 256, 512, 1024]).to(device)
  D = Discriminator(28 * 28, 1, [512, 256]).to(device)
  train(G, D)
