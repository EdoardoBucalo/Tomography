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
    def __init__(self, input_size, output_channels, learning_rate, feature_map_size, kernel_size,k0_size,kf_size, padding,stride,stride2,output_padding):
        super().__init__()
        self.lr = learning_rate
        self.input_size = input_size
        self.output_channels = output_channels
        self.feature_map_size = feature_map_size
        
        # Define model layers
        self.fc_layer = nn.Linear(input_size, feature_map_size[0] * feature_map_size[1])  # Fully connected layer
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size, stride, padding, output_padding),  # Transpose Convolution
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, k0_size, stride2, padding, output_padding),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, output_channels, kf_size, stride, padding, output_padding),  # Final Transpose Conv
            
            nn.Sigmoid()  # Output normalized between 0 and 1
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
        x = x.view(x.size(0), 1, self.feature_map_size[0], self.feature_map_size[1])  # Reshape for convolution
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
        return optim.SGD(self.parameters(), lr=self.lr)
 
    def calc_em(self, batch, y_hat):
        _, y, j0, j1, em, em_hat, radii, angles = batch
        # # Print the shape of the radii and angles tensors
        # print(f"Radii shape: {radii.shape}")
        # print(f"Angles shape: {angles.shape}")

        # Split the coefficients for all samples in the batch along the last dimension (size 7)
        a0cl, a1cl, a1sl = torch.split(y, 7, dim=-1)
        a0cl_hat, a1cl_hat, a1sl_hat = torch.split(y_hat, 7, dim=-1)
        # Reshape coefficient tensors to match the dimensions of j0 and j1
        a0cl = a0cl.unsqueeze(1).unsqueeze(1)  # Shape: [32, 1, 1, 7]
        a1cl = a1cl.unsqueeze(1).unsqueeze(1)  # Shape: [32, 1, 1, 7]
        a1sl = a1sl.unsqueeze(1).unsqueeze(1)  # Shape: [32, 1, 1, 7]
    
        a0cl_hat = a0cl_hat.unsqueeze(1).unsqueeze(1)
        a1cl_hat = a1cl_hat.unsqueeze(1).unsqueeze(1)
        a1sl_hat = a1sl_hat.unsqueeze(1).unsqueeze(1)

        # # shape of the coefficients
        # print(f"Shape of a0cl: {a0cl.shape}")
        # print(f"Shape of a1cl: {a1cl.shape}")
        # print(f"Shape of a1sl: {a1sl.shape}")

        # Ensure j0, j1 are the same dtype as the coefficients
        j0 = j0.type(a0cl.dtype)
        j1 = j1.type(a1cl.dtype)
        # # shape of the j0, j1 tensors
        # print(f"Shape of j0: {j0.shape}")
        # print(f"Shape of j1: {j1.shape}")
    
        # Perform dot products over the last dimension (7) using einsum
        dot0_c = torch.einsum('bijl,bijl->bij', j0, a0cl)
        dot1_c = torch.einsum('bijl,bijl->bij', j1, a1cl)
        dot1_s = torch.einsum('bijl,bijl->bij', j1, a1sl)
        # # print the shape of the dots tensors
        # print(f"Dot0_c shape: {dot0_c.shape}")
        # print(f"Dot1_c shape: {dot1_c.shape}")
        # print(f"Dot1_s shape: {dot1_s.shape}")

        dot0_c_hat = torch.einsum('bijl,bijl->bij', j0, a0cl_hat)
        dot1_c_hat = torch.einsum('bijl,bijl->bij', j1, a1cl_hat)
        dot1_s_hat = torch.einsum('bijl,bijl->bij', j1, a1sl_hat)
    
        # Compute the emissivity maps for all samples in the batch
        em = dot0_c + dot1_c * torch.cos(angles) + dot1_s * torch.sin(angles)
        em_hat = dot0_c_hat + dot1_c_hat * torch.cos(angles) + dot1_s_hat * torch.sin(angles)
    
        # Apply the constraints
        em = torch.where(em < 0, torch.tensor(0.0, dtype=em.dtype, device=em.device), em)
        em = torch.where(radii > 1.0, torch.tensor(-10.0, dtype=em.dtype, device=em.device), em)
    
        em_hat = torch.where(em_hat < 0, torch.tensor(0.0, dtype=em_hat.dtype, device=em_hat.device), em_hat)
        em_hat = torch.where(radii > 1.0, torch.tensor(-10.0, dtype=em_hat.dtype, device=em_hat.device), em_hat)
    
        # Normalize the maps
        em = em / 0.459
        em_hat = em_hat / 0.459
        
        return em, em_hat

  # This is the same function as above but implemented using a for loop over the
  # samples in the batch! The above implementation is more efficient as it uses
  # einsum to perform the dot products over the last dimension of the tensors.
  # Or at least, it is more efficient on the memory side, as it does not require
  # to store the intermediate results of the dot products for each sample in the
  # batch. But it is not clear if it is more efficient in terms of computation time.
  # The for loop implementation is more readable and easier to understand.
    def calc_em_for_loop(self, batch, y_hat):
        _, y, j0, j1, em, em_hat, radii, angles = batch
        em_list = []
        em_hat_list = []
        for index in range(len(y)):
            # compute the map from the data
            a0cl, a1cl, a1sl = torch.split(y[index], 7)
            # change the j0, j1 to the same type as the coefficients
            j0 = j0.type(a0cl.dtype)
            j1 = j1.type(a1cl.dtype)
            # perform the dot product
            dot0_c = torch.matmul(j0, a0cl)
            dot1_c = torch.matmul(j1, a1cl)
            dot1_s = torch.matmul(j1, a1sl)
            # finally compute the emissivity map
            em = dot0_c + dot1_c * torch.cos(angles) + dot1_s * torch.sin(angles)
            em[em < 0] = 0 # get rid of negative values (unphysical)
            em[radii > 1.0] = -10 # get the values outside the circle to -10

        # compute the map from the model
            a0cl_hat, a1cl_hat, a1sl_hat = torch.split(y_hat[index], 7)
            # change the j0, j1 to the same type as the coefficients
            j0 = j0.type(a0cl_hat.dtype)
            j1 = j1.type(a0cl_hat.dtype)
            #  perform the dot product
            dot0_c_hat = torch.matmul(j0, a0cl_hat)
            dot1_c_hat = torch.matmul(j1, a1cl_hat)
            dot1_s_hat = torch.matmul(j1, a1sl_hat)
            # finally compute the emissivity map
            em_hat = dot0_c_hat + dot1_c_hat * torch.cos(angles) + dot1_s_hat * torch.sin(angles)
            em_hat[em < 0] = 0 # get rid of negative values (unphysical)
            em_hat[radii > 1.0] = -10 # set the values outside the circle to -10

            em_list.append(em / 0.459)
            em_hat_list.append(em_hat / 0.459)

        return torch.stack(em_list), torch.stack(em_hat_list) # return the normalized maps

