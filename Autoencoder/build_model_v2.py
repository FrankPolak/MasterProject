"""
PyTorch code to instantiate an Autoencoder.
"""

import torch
import torch.nn as nn

class AutoencoderV0(nn.Module):
    """
    Creates an autoencoder architecture.

    Args:
        input_features (int): The number of input/output features.
        hidden_features (int): The number of features in the latent space.

    Methods:
        forward(x): Passes input through the encoder and decoder to reconstruct the input.
        encode(x): Returns the latent space representation of the input data.
    """


    def __init__(self, input_features, hidden_features=125):
        super(AutoencoderV0, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=hidden_features),
            nn.ReLU(),
        )

        # Decoder
        decoder_layers = [
            nn.Linear(in_features=hidden_features, out_features=250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=input_features),
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent_space = self.encoder(x)  # Extract latent representation
        reconstructed = self.decoder(latent_space)  # Decode from latent space
        return reconstructed

    def encode(self, x):
        """Returns only the latent space representation."""
        return self.encoder(x)
