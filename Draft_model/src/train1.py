import lightning as L
import torch
import wandb
import math
from math import log
from model import TomoModel
from dataset import TomographyDataModule
import config
from callbacks import PrintingCallback, SaveBest, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger





sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'batch_size': {'values': [16]},
        'epochs': {'value': 100},
        'fc_layer_size':{'values':[[512,512,512]]},
        'activation_function': {'values': ['Swish']},
        'learning_rate': {'values':[ 0.00008440371332291889]}, # {'distribution$
        'optimizer': {'values': ['ftrl']},
        'INPUTSIZE': {'value': 92},
        'OUTPUTSIZE': {'value': 21},
        'DATA_DIR': {'value': '../data/'},  # Aggiungi DATA_DIR all'interno di $
        'FILE_NAME': {'value': 'data_clean.npy'},  # Aggiungi FILE_NAME all'int$
        'NUM_WORKERS': {'value': 0},  # Aggiungi NUM_WORKERS all'interno di 'pa$
        'ACCELERATOR': {'value': 'gpu'},  # Aggiungi ACCELERATOR all'interno di$
        'DEVICES': {'value': [0]},  # Aggiungi DEVICES all'interno di 'paramete$
        'PRECISION': {'value': '16-mixed'},  # Aggiungi PRECISION all'interno d$
        }
}
# Impostazione della precisione per le operazioni con torch
torch.set_float32_matmul_precision("medium")
# Definisce la funzione principale che esegue l'addestramento
def main(W_config=None):
    # Inizializza WandB
    with wandb.init(project="my_Tomo_model", config=W_config):
        config = wandb.config
        wandb_logger = WandbLogger(project="my_Tomo_model")
        
        # Imposta il logger per TensorBoard
        logger = TensorBoardLogger("TB_logs", name="my_Tomo_model")

        # Inizializza il modello con i parametri dallo sweep
        model = TomoModel(
            inputsize=config.INPUTSIZE,
            learning_rate=config.learning_rate,
            outputsize=config.OUTPUTSIZE,
            fc_layer_size=config.fc_layer_size,
            activation=config.activation_function,
            
            
        )
        wandb.watch(model)
        # Inizializza il modulo di dati
        dm = TomographyDataModule(
            data_dir=config.DATA_DIR,
            file_name=config.FILE_NAME,
            batch_size=config.batch_size,
            num_workers=config.NUM_WORKERS,
        )

        # Configura il trainer
        trainer = L.Trainer(
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=1,
            max_epochs=config.epochs,
            precision=config.PRECISION,
            enable_progress_bar=True,
            callbacks=[
                PrintingCallback(),
                SaveBest(monitor="val_loss", logger=logger),
               # EarlyStopping(monitor="val_loss")
            ],
        )

        
        trainer.fit(model, dm)
        for epoch in range(trainer.current_epoch):

            wandb.log({
                "epoch":epoch,
                "train_loss": trainer.callback_metrics["train_loss"],  # Train loss
                "val_loss": trainer.callback_metrics["val_loss"],  # Validation loss
            })
        trainer.validate(model, dm)
        trainer.test(model, dm)
        return model, dm
        

# Esegue lo sweep
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="my_Tomo_model")
    wandb.agent(sweep_id, function=main, count=20)
