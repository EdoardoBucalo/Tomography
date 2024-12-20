import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
from scipy.special import j0, j1, jn_zeros
import utils
from pytorch_lightning import LightningModule


class TomoModel(L.LightningModule):
    def __init__(self, input_size,  learning_rate,output_channels=1, feature_map_size=(25, 25)):
        super().__init__()
        self.lr = learning_rate
        self.input_size = input_size
        self.output_channels = output_channels
        self.feature_map_size = feature_map_size
        
        # Define model layers
        self.fc_layer = nn.Linear(input_size, feature_map_size[0] * feature_map_size[1])  # Fully connected layer
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=1, padding=1, output_padding=0),  # Transpose Convolution
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=1,output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1,output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, output_channels,kernel_size=6, stride=1, padding=1, output_padding=0),  # Final Transpose Conv
            
            
        )
        
        # Loss and metrics
        self.loss_rate = 0.2  # Define the loss rate
        self.loss_fn = nn.MSELoss()  # Define the loss function
        self.best_val_loss = torch.tensor(float('inf'))  # Initialize the best validation loss
        self.mse = torchmetrics.MeanSquaredError()  # Mean Squared Error metric
        self.mae = torchmetrics.MeanAbsoluteError()  # Mean Absolute Error metric
        self.r2 = torchmetrics.R2Score()  # R2 score metric
        self.md = torchmetrics.MinkowskiDistance(p=4)  # Minkowski Distance metric

        self.training_step_outputs = []  # Initialize a list to store training step outputs

    def forward(self, x):
        # Map input to feature map
        x = self.fc_layer(x)
        x = x.view(x.size(0),1, self.feature_map_size[0], self.feature_map_size[1])  # Reshape for convolution
        x = self.conv_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function

        mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
        mse = self.mse(y_hat,y)
    
        r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
        md = self.md(y_hat, y)  # Compute md using the y_hat (prediction) and target
        mse = self.mse(y_hat, y)
        self.training_step_outputs.append(loss.detach().cpu())  # Append the loss to the training step outputs list
        self.log_dict({'train_loss': loss,
                   'train_mae': mae,
                   'train_r2': r2,
                   'train_md': md,
                   'train_mse':mse},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
        return {"loss": loss, "preds": y_hat, "target": y}
  
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
        mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
        r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
        md = self.md(y_hat, y)  # Compute md using the y_hat (prediction) and target
        mse= self.mse(y_hat, y)
        self.log_dict({'val_loss': loss,
                   'val_mae': mae,
                   'val_r2': r2,
                   'val_md': md,
                   'val_mse':mse},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
        return loss
  
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
        mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
        r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
        md = self.md(y_hat, y)  # Compute md using the y_hat (prediction) and target
        mse = self.mse(y_hat, y)
        self.log_dict({'test_loss': loss,
                   'test_mae': mae,
                   'test_r2': r2,
                   'test_md': md,
                   'test_mse':mse},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, mae, and F1 score
        return loss
    def _common_step(self, batch, batch_idx):
        x, y = batch  # Supponendo che il batch contenga input (x) e target (y)
        y_hat = self(x)  # Calcola le predizioni del modello (output della CNN)
        y_hat = y_hat.squeeze(1)  # Rimuove la dimensione del canale: (batch_size, 1, 110, 110) -> (batch_size, 110, 110)
        # Se calcoli anche le mappe di emissività, usa calc_em
        # em, em_hat = self.calc_em(batch, y_hat)  # Calcola la mappa di emissività (se necessario)
    
        # Calcola la perdita (modificata se desideri combinare la perdita su y_hat e em_hat)
        loss = self.loss_fn(y_hat, y)  # La perdita di default tra predizioni e target
        # Se hai em_hat ed em, puoi combinarli così:
        # loss = ((1 - self.loss_rate) * self.loss_fn(y_hat, y)) + (self.loss_rate * self.loss_fn(em_hat, em))
    
        return loss, y_hat, y
  
    def predict_step(self, batch, batch_idx):
        x = batch[0]  # Prendi l'input dal batch (senza reshape se è un'immagine)
        y_hat = self(x)  # Calcola le predizioni passando l'input attraverso la CNN
    
        # Se stai cercando di fare una classificazione multi-classe, usa argmax per ottenere la classe predetta
        preds = torch.argmax(y_hat, dim=1)  # Classificazione: argmax lungo la dimensione delle classi
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
 
   
