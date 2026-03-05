"""Tests for the Convolutional VAE."""

import torch
import pytest

from world_model.models.vae import ConvVAE, vae_loss


class TestConvVAE:
    def setup_method(self):
        self.model = ConvVAE(latent_dim=32)
        self.batch = torch.randn(4, 3, 64, 64)

    def test_forward_shapes(self):
        recon, mu, logvar = self.model(self.batch)
        assert recon.shape == (4, 3, 64, 64)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_encode_shapes(self):
        mu, logvar = self.model.encode(self.batch)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_decode_shapes(self):
        z = torch.randn(4, 32)
        recon = self.model.decode(z)
        assert recon.shape == (4, 3, 64, 64)

    def test_reconstruction_range(self):
        recon, _, _ = self.model(torch.sigmoid(self.batch))
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    def test_vae_loss(self):
        recon, mu, logvar = self.model(self.batch)
        loss, details = vae_loss(recon, self.batch, mu, logvar, beta=0.5)
        assert loss.requires_grad
        assert "recon_loss" in details
        assert "kl_loss" in details
        assert "total_loss" in details
        assert details["total_loss"] > 0

    def test_reparameterize_different_samples(self):
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)
        z1 = self.model.reparameterize(mu, logvar)
        z2 = self.model.reparameterize(mu, logvar)
        assert not torch.allclose(z1, z2)

    def test_different_latent_dims(self):
        for dim in [8, 16, 64]:
            model = ConvVAE(latent_dim=dim)
            recon, mu, _ = model(self.batch)
            assert mu.shape == (4, dim)
            assert recon.shape == (4, 3, 64, 64)
