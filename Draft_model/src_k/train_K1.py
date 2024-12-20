import lightning as L
import torch
import wandb
from model import TomoModel
from dataset_K import TomographyDataModule
from callbacks import PrintingCallback, SaveBest, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold  # Importa KFold da scikit-learn

# Impostazione della precisione per le operazioni con torch
torch.set_float32_matmul_precision("medium")

# Sweep configuration per wandb
sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'batch_size': {'values': [16, 32, 64, 128]},
        'epochs': {'value': 10},
        'fc_layer_size': {'values': [64, 128, 256, 512]},
        'learning_rate': {'distribution': 'uniform', 'max': 0.1, 'min': 0},
        'optimizer': {'values': ['adam', 'sgd']},
        'INPUTSIZE': {'value': 92},
        'OUTPUTSIZE': {'value': 21},
        'DATA_DIR': {'value': '../data/'},
        'FILE_NAME': {'value': 'data_clean.npy'},
        'NUM_WORKERS': {'value': 0},
        'ACCELERATOR': {'value': 'gpu'},
        'DEVICES': {'value': [0]},
        'PRECISION': {'value': '16-mixed'},
        'FOLDS':{'value': 5}
    }
}

# Numero di fold per la Cross-Validation
#n_folds = 5

# Funzione principale per l'addestramento
def main(W_config=None):
    # Inizializza WandB
    with wandb.init(project="my_Tomo_model", config=W_config):
        config = wandb.config
        wandb_logger = WandbLogger(project="my_Tomo_model")
        
        # Imposta il logger per TensorBoard
        logger = TensorBoardLogger("TB_logs", name="my_Tomo_model")

        # Carica il modulo dati completo
        dm = TomographyDataModule(
            data_dir=config.DATA_DIR,
            file_name=config.FILE_NAME,
            batch_size=config.batch_size,
            num_workers=config.NUM_WORKERS,
            k_folds=config.FOLDS
        )

        for fold in range(config.FOLDS):
            print(f"Fold {fold + 1}/{config.FOLDS}")
            # Imposta il modulo dati per il fold corrente
            dm.setup(fold_idx=fold)

            # Inizializza il modello
            model = TomoModel(
                inputsize=config.INPUTSIZE,
                learning_rate=config.learning_rate,
                outputsize=config.OUTPUTSIZE,
                fc_layer_size=config.fc_layer_size
            )
            wandb.watch(model)

            # Configura il trainer
            trainer = L.Trainer(
                logger=wandb_logger,
                accelerator=config.ACCELERATOR,
                devices=config.DEVICES,
                min_epochs=1,
                max_epochs=config.epochs,
                precision=config.PRECISION,
                enable_progress_bar=True,
                callbacks=[
                    PrintingCallback(),
                    SaveBest(monitor="val_loss", logger=wandb_logger),
                    EarlyStopping(monitor="val_loss")
                ],
            )

            # Esegue il training
            trainer.fit(model, dm)
            
            

            # Log dei risultati per fold
            wandb.log({
                "fold": fold+1,
                "val_loss": trainer.callback_metrics.get("val_loss", None),
                "train_loss": trainer.callback_metrics.get("train_loss", None) })
               
               
            trainer.validate(model, dm)
            #trainer.test(model, dm)
          
           

    return model, dm

# Esegue lo sweep
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="my_Tomo_model")
    wandb.agent(sweep_id, function=main, count=20)
