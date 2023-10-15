import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     x = x.view(x.size(0), -1)
    #     z = self.encoder(x)
    #     x_hat = self.decoder(z)
    #     loss = nn.functional.mse_loss(x_hat, x)
    #     # Logging to TensorBoard (if installed) by default
    #     self.log("val_loss", loss)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     x = x.view(x.size(0), -1)
    #     z = self.encoder(x)
    #     x_hat = self.decoder(z)
    #     loss = nn.functional.mse_loss(x_hat, x)
    #     # Logging to TensorBoard (if installed) by default
    #     self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
print(autoencoder, autoencoder.device)

# setup data
transform = ToTensor()
train_set = MNIST(os.getcwd(), train=True, download=True, transform=transform)
train_set_size  = int(0.8 * len(train_set))
val_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, val_set = utils.data.random_split(train_set, [train_set_size, val_set_size], generator=seed)
train_loader = utils.data.DataLoader(train_set)
val_loader = utils.data.DataLoader(val_set)

test_set = MNIST(os.getcwd(), train=False, download=True, transform=transform)
test_loader = utils.data.DataLoader(test_set)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=5, min_epochs=1)
trainer.predict(model=autoencoder, dataloaders=test_loader)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(autoencoder.device)
# # test using the best model!
# trainer.test(model=autoencoder, dataloaders=test_loader)






# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)


# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=4-step=500.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)
# 直接读出来是mps上的，奇怪，难道是因为训练是在mps上训练的结果
# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

print(autoencoder.device)
fake_image_batch = fake_image_batch.to(autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

