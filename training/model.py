import numpy as np
import torch
import torch.nn as nn
import torch.distributions
from torch.optim import AdamW 
import torch.nn.functional as F
from make_analogies.helper_functions import calculate_d_penalty, validate_reconstruction, validate_d_penalty


class VariationalAutoencoder(nn.Module):
    def __init__(self, img_channels=10, feature_dim=[128, 2, 2], latent_dim=128, use_dropout=False):
        super(VariationalAutoencoder, self).__init__()

        self.f_dim = feature_dim
        kernel_vae = 4
        stride_vae = 2

        self.use_dropout = use_dropout
        dropout_prob = 0.5

        # Initializing the convolutional layers and 2 full-connected layers for the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,
                      out_channels=128,
                      kernel_size=kernel_vae,
                      stride=stride_vae),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=kernel_vae,
                      stride=stride_vae),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=kernel_vae,
                      stride=stride_vae),
            nn.LeakyReLU())
        self.fc_mu = nn.Linear(np.prod(self.f_dim), latent_dim)
        self.fc_var = nn.Linear(np.prod(self.f_dim), latent_dim)

        # Initializing the fully-connected layer and convolutional layers for decoder
        self.dec_inp = nn.Linear(latent_dim, np.prod(self.f_dim))

        # Define layers in a list and conditionally add dropout layers
        layers = [
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=kernel_vae, stride=stride_vae),
            nn.LeakyReLU()
        ]

        if self.use_dropout:
            layers.append(nn.Dropout2d(p=dropout_prob))

        layers.extend([
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=kernel_vae, stride=stride_vae),
            nn.LeakyReLU()
        ])

        if self.use_dropout:
            layers.append(nn.Dropout2d(p=dropout_prob))

        layers.extend([
            nn.ConvTranspose2d(in_channels=128, out_channels=img_channels, kernel_size=kernel_vae, stride=stride_vae),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*layers)
        
    def encode(self, x):
        # Input is fed into convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.encoder(x)
        x = x.view(-1, np.prod(self.f_dim))
        mu = self.fc_mu(x)
        logVar = self.fc_var(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and samples the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        # z is fed back into a fully-connected layers and then into transpose convolutional layers
        # The generated output is the same size as the original input
        x = self.dec_inp(z)
        x = x.view(-1, self.f_dim[0], self.f_dim[1], self.f_dim[2])
        x = self.decoder(x)
        return x.squeeze()

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        out = self.decode(z)
        return out, mu, logVar
    

def train_encoder(model, 
          train_loader, 
          valid_loader,
          epochs=50, 
          rule_lambda=5e4, 
          lr=1e-3, 
          loss_fn = 'reconstruction + d',
          device='cuda'):
    optimizer = AdamW(model.parameters(), lr, weight_decay=0.2)
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (input, output, rule_ids) in enumerate(train_loader):

            # Combine input & output, adding noise, attaching to device
            in_out = torch.cat((input, output), dim=0)
            in_out = in_out.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = model(in_out)

            kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
            bce = F.binary_cross_entropy(out, in_out, reduction='sum') 

            if loss_fn == 'reconstruction + d':
                d_positive = calculate_d_penalty(mu, rule_ids)
                loss = bce + kl_divergence + rule_lambda*d_positive
            elif loss_fn == 'reconstruction':
                loss = bce + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
        d_valid = validate_d_penalty(model, valid_loader)
        print(f'Epoch {epoch+1}: Loss {round(bce.item() + kl_divergence.item(), 2)} - D {round(d_positive.item(), 2)} ({round(d_valid.item(), 2)})')

        acc_zero_train, acc_non_zero_train, = validate_reconstruction(model, train_loader)
        acc_zero_valid, acc_non_zero_valid, = validate_reconstruction(model, valid_loader)
        
        print('All Pixels : {0:.2f} ({2:.2f})% - Non-Zero Pixels : {1:.2f} ({3:.2f})%'.format(acc_zero_train*100, acc_non_zero_train*100, acc_zero_valid*100, acc_non_zero_valid*100))

    return model
