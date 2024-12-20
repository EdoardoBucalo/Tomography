import lightning as L
import torch
import numpy as np
import os
from model import TomoModel
from dataset_c import TomographyDataModule
import config
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jn_zeros
from utils import compute_bessel_n_mesh
import time

def visualize(version_num):
  # Define an instance of the model
  model = TomoModel(config.INPUTSIZE, config.LEARNING_RATE, config.OUTPUTSIZE)
  # Load the best model
#version_num = 54
  assert os.path.exists(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt"), "The model does not exist"
  model.load_state_dict(torch.load(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt", weights_only=True)['state_dict'])
  print(model)
  # Define the data module
  data_module = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  # Load the data
  data_module.setup()
  # Define the dataloaders
  val_loader = data_module.val_dataloader()
  # print(f"Validation dataset: {val_loader.dataset[0]}")
  test_loader = data_module.test_dataloader()
  # input_data = val_loader.dataset[0][0].unsqueeze(0)
  # reference = val_loader.dataset[0][1]
  input_data = next(iter(val_loader))[0]
  val_reference = next(iter(val_loader))[1]
  # predict on the test dataset 
  test_data = next(iter(test_loader))[0]
  test_reference = next(iter(test_loader))[1]

  v = model(input_data)
  t = model(test_data)
  
  # print the validation predictions
  #print(f"Validation predictions: {v}")
  # print the validation reference
  #print(f"Validation reference: {val_reference}")
  # compute the error on the validation set
  #print(f"Validation error: {v - val_reference}")
  # compute the mean squared error
  print(f"Validation mean squared error: {torch.mean((v - val_reference)**2)}")

  # print some art to separate the results
  print("*" * 50)

  # print the test predictions
  #print(f"Test predictions: {t}")
  # print the test reference
  #print(f"Test reference: {test_reference}")
  # compute the error on the test set
  #print(f"Test error: {t - test_reference}")
  # compute the mean squared error
  print(f"Test mean squared error: {torch.mean((t - test_reference)**2)}")

  # plot the results
  plt.figure()
  plt.plot(val_reference[0].detach().numpy(), label="Reference", color='orange', marker='o')
  plt.plot(v[0].detach().numpy(), label="Prediction", color='blue', linestyle='dashed', marker='x')
  # plt.legend()
  plt.savefig("results.png")

  # print some art to separate the results
  print("*" * 50)
  # print(f"Test results: {t}")

def generate_maps(dataloader, model):
    """
    Generate the tomography maps directly from the model output.
    """
    # Preleva i dati dal dataloader
    data = next(iter(dataloader))[0]
    # Output del modello: mappe direttamente
    
    model_maps = model(data).detach().numpy()
    # Mappe precompute
    precomputed_maps = dataloader.dataset.dataset.target
    return model_maps, precomputed_maps
def plot_maps(model, dataloader):
    """
    Generate and plot emissivity maps.
    """
    val_loader = dataloader.val_dataloader()
    batch = next(iter(val_loader))
    # Le mappe generate dal modello
    model_maps = model(batch[0])
    # Le mappe precompute
    precomputed_maps = batch[1]  # Supponendo che siano le label

    return model_maps, precomputed_maps
def plot_maps_for_loop(model_maps, precomputed_maps, index, version_num):
    """
    Plot the model maps and precomputed maps side by side.
    """
    # Seleziona una mappa specifica
    model_map = model_maps[index].detach().numpy()
    precomputed_map = precomputed_maps[index].detach().numpy()
    diff_map = np.abs(model_map - precomputed_map)
    percentage_error = np.mean(diff_map / (np.abs(precomputed_map) + 1e-10)) * 100  # Evita divisioni per 0

    # Rimuove la dimensione del batch, se è presente
    model_map = model_map.squeeze(0)
    diff_map = diff_map.squeeze(0)    # Do the same for the diff_map

    # Traccia le mappe
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axs[0].imshow(model_map, cmap='viridis', interpolation='nearest')
    axs[0].set_title("Prediction map")
    fig.colorbar(im0, ax=axs[0], shrink=0.6)

    im1 = axs[1].imshow(precomputed_map, cmap='viridis', interpolation='nearest')
    axs[1].set_title("Model map")
    fig.colorbar(im1, ax=axs[1], shrink=0.6)

    im2 = axs[2].imshow(diff_map, cmap='viridis', interpolation='nearest')
    axs[2].set_title(f"Difference map\nError: {percentage_error:.2f}%")
    fig.colorbar(im2, ax=axs[2], shrink=0.6)

    # Salva il grafico
    save_dir=f"plots/maps/version_{version_num}"
    os.makedirs(save_dir, exist_ok=True)

    fig.savefig(f"{save_dir}/maps_{index}.png")
    plt.close()

if __name__ == "__main__":
  version_num = 1
  #visualize(version_num)

  # # Load the data and the model
  # datamodule = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  # datamodule.setup()
  # val_loader = datamodule.val_dataloader()

  # model = TomoModel(config.INPUTSIZE, config.LEARNING_RATE, config.OUTPUTSIZE)
  # version_num = 37
  # assert os.path.exists(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt"), "The model does not exist"
  # model.load_state_dict(torch.load(
  #   f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt",
  #   )['state_dict'])
  
  # # Generate the maps
  # model_map, pc_map = generate_maps(val_loader, model)
  # # plot the maps side by side
  # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  # im0 = axs[0].imshow(model_map, cmap='viridis', interpolation='nearest')
  # axs[0].set_title("Model map")
  # # plot the colorbar rescaled by 60%
  # fig.colorbar(im0, ax=axs[0], shrink=0.6)

  # im1 = axs[1].imshow(pc_map, cmap='viridis', interpolation='nearest')
  # axs[1].set_title("Precomputed map")
  # fig.colorbar(im1, ax=axs[1], shrink=0.6)
  # # compute the difference between the two maps
  # diff_map = np.abs(model_map - pc_map)#/np.max(pc_map)
  # im2 = axs[2].imshow(diff_map, cmap='viridis', interpolation='nearest')
  # axs[2].set_title("Difference map")
  # fig.colorbar(im2, ax=axs[2], shrink=0.6)

  # # save the figure
  # fig.savefig(f"maps_{version_num}.png")
  # plt.show()
  start = time.time()
  model = TomoModel(config.INPUTSIZE, config.LEARNING_RATE, config.OUTPUTSIZE)
  assert os.path.exists(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt"), "The model does not exist"
  model.load_state_dict(torch.load(f"my_Tomo_model_CNN/b9wtsyot/checkpoints/epoch=99-step=173400.ckpt",)['state_dict'])

  datamodule = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  datamodule.setup()

  em, em_hat = plot_maps(model, datamodule)
  stop = time.time()
  print(f"Time elapsed: {stop - start}")
  for i in range(32):
    plot_maps_for_loop(em, em_hat, i, version_num)
