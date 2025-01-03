import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
from scipy.special import j0, j1, jn_zeros
import utils


class TomoModel(L.LightningModule):
  def __init__(self, inputsize, learning_rate, outputsize,fc_layer_size=124, activation='ReLU'):
    super().__init__()
    self.lr = learning_rate
    self.inputsize = inputsize
    self.outputsize= outputsize
    self.fc_layer_size = fc_layer_size
    self.activation_type = activation
    self.layers = nn.ModuleList()
    
    #######
    ######
    # Primo layer di input
    self.layers.append(nn.Linear(inputsize, fc_layer_size[0]))
    self.activations=nn.ModuleList([self._get_activation()])
    
    # Aggiungere layer successivi
    self.hidden_layers=nn.ModuleList()
    for i in range(len(fc_layer_size)-1):
        self.layers.append(nn.Linear(fc_layer_size[i], fc_layer_size[i+1]))
        self.activations.append(self._get_activation())

    # Ultimo layer di output
    self.output_layer = nn.Linear(fc_layer_size[-1], outputsize)
    #######
    ######
    
    self.loss_rate = 0.2  # Define the loss rate
    self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
    self.best_val_loss = torch.tensor(float('inf'))  # Initialize the best validation loss
    self.mse =torchmetrics.MeanSquaredError()
    self.mae = torchmetrics.MeanAbsoluteError() # Define Root Mean Squared Error metric
    self.r2 = torchmetrics.R2Score()  # Define R2 score metric, using the multioutput parameter the metric will return an array of R2 scores for each output
    self.md = torchmetrics.MinkowskiDistance(p=4)  # Define F1 score metric
    self.training_step_outputs = []  # Initialize an empty list to store training step outputs
    
  def _get_activation(self):
        # Dizionario per attivazioni senza parametri
        activation_dict = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'Softmax': nn.Softmax(dim=1),  # Specifica `dim=1` per Softmax
            'Swish': nn.SiLU(),  # Swish in PyTorch è implementato come SiLU
            'GELU': nn.GELU(),
            'ELU': nn.ELU()
        }
        

        if self.activation_type == 'ReLU':
            return nn.ReLU()
        elif self.activation_type == 'Sigmoid':
            return nn.Sigmoid()
        elif self.activation_type == 'Tanh':
            return nn.Tanh()
       
           
        elif self.activation_type == 'Softmax':
            return nn.Softmax(dim=1)  # Assicurati di impostare il parametro `dim` per Softmax
        elif self.activation_type == 'Swish':
            return nn.SiLU()  # Swish è spesso implementato come SiLU (Sigmoid Linear Unit)
        elif self.activation_type == 'GELU':
            return nn.GELU()
        elif self.activation_type == 'ELU':
            return nn.ELU()
        elif self.activation_type.startswith('LeakyReLU'):
            # Estrai il parametro (ad esempio 0.01)
            start_idx = self.activation_type.find('(') + 1
            end_idx = self.activation_type.find(')')
            
            if start_idx != -1 and end_idx != -1:
                alpha = float(self.activation_type[start_idx:end_idx])
            else:
                alpha = 0.01  # Imposta il valore predefinito se non specificato

            return nn.LeakyReLU(alpha)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_type}")

    
  
  def forward(self, x):
        # Passaggio attraverso i layer definiti
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))  # Applica il layer e l'attivazione

        # Passaggio finale attraverso il layer di output
        x = self.output_layer(x)
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
    x, y = batch[0], batch[1]
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
   #em, em_hat = self.calc_em(batch, y_hat) # Compute the tomography map using the coefficients
    
    # The following lines are used to set the value of the loss_rate variable (0.2)
    # print(f"MSE on the coefficients vector: {self.loss_fn(y_hat, y)}")
    # print(f"MSE on the emissivity maps: {self.loss_fn(em_hat, em)}")

    #loss = ((1 - self.loss_rate) * self.loss_fn(y_hat, y)) + (self.loss_rate * self.loss_fn(em_hat, em))  # Compute the loss using the y_hat (prediction) and target
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    preds = torch.argmax(y_hat, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate
  
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
      # perform the dot product
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
