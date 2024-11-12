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

# Impostazione della precisione per le operazioni con torch
torch.set_float32_matmul_precision("medium")

sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'batch_size': {
            'distribution': 'q_log_uniform',
            'max': math.log(256),
            'min': math.log(32),
            'q': 1
        },
        'epochs': {'value': 20},
        'fc_layer_size': {'values': [128, 256, 512]},
        'learning_rate': {'distribution': 'uniform', 'max': 0.1, 'min': 0},
        'optimizer': {'values': ['adam', 'sgd']},
        'INPUTSIZE': {'value': 92},
        'OUTPUTSIZE': {'value': 21},
        'DATA_DIR': {'value': '../data/'},  # Aggiungi DATA_DIR all'interno di 'parameters'
        'FILE_NAME': {'value': 'data_clean.npy'},  # Aggiungi FILE_NAME all'interno di 'parameters'
        'NUM_WORKERS': {'value': 0},  # Aggiungi NUM_WORKERS all'interno di 'parameters'
        'ACCELERATOR': {'value': 'gpu'},  # Aggiungi ACCELERATOR all'interno di 'parameters'
        'DEVICES': {'value': [0]},  # Aggiungi DEVICES all'interno di 'parameters'
        'PRECISION': {'value': '16-mixed'}  # Aggiungi PRECISION all'interno di 'parameters'
    }
}
# Definisce la funzione principale che esegue l'addestramento
def main(W_config=None):
    # Inizializza WandB
    with wandb.init(config=W_config):
        config = wandb.config

        # Imposta il logger per TensorBoard
        logger = TensorBoardLogger("TB_logs", name="my_Tomo_model")

        # Inizializza il modello con i parametri dallo sweep
        model = TomoModel(
            inputsize=config.INPUTSIZE,
            learning_rate=config.learning_rate,
            outputsize=config.OUTPUTSIZE,
            fc_layer_size=config.fc_layer_size
        )

        # Inizializza il modulo di dati
        dm = TomographyDataModule(
            data_dir=config.DATA_DIR,
            file_name=config.FILE_NAME,
            batch_size=config.batch_size,
            num_workers=config.NUM_WORKERS,
        )

        # Configura l'ottimizzatore in base al parametro dello sweep
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

        # Configura il trainer
        trainer = L.Trainer(
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=1,
            max_epochs=config.epochs,
            precision=config.PRECISION,
            enable_progress_bar=True,
            callbacks=[PrintingCallback(),
                       SaveBest(monitor="val_loss", logger=logger)],
                       #EarlyStopping(monitor="val_loss")],
        )

        # Esegue l'addestramento
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)

        # Logga la metrica di valutazione per WandB
        val_loss = trainer.callback_metrics.get("val_loss", None)
        wandb.log({"val_loss": val_loss})

# Esegue lo sweep
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="nome_del_tuo_progetto")
    wandb.agent(sweep_id, function=main, count=20)
