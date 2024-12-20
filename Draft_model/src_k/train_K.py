import lightning as L
import torch
import config
from model import TomoModel
from dataset_K import TomographyDataModule
from callbacks import PrintingCallback, SaveBest, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.model_selection import KFold  # Importa KFold da scikit-learn

# Impostazione della precisione per le operazioni con torch
torch.set_float32_matmul_precision("medium")

# Numero di fold per la Cross-Validation
n_folds = 5

# Funzione principale per l'addestramento
def main(W_config=None):
        # Imposta il logger per TensorBoard
        logger = TensorBoardLogger("TB_logs", name="my_Tomo_model")

        # Carica il modulo dati completo
        dm = TomographyDataModule(
            data_dir=config.DATA_DIR,
            file_name=config.FILE_NAME,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            k_folds=n_folds
        )

        for fold in range(n_folds):
            print(f"Fold {fold + 1}/{n_folds}")
            # Imposta il modulo dati per il fold corrente
            dm.setup(fold_idx=fold)

            # Inizializza il modello
            model = TomoModel(
                inputsize=config.INPUTSIZE,
                learning_rate=config.LEARNING_RATE,
                outputsize=config.OUTPUTSIZE,
                fc_layer_size=config.fc_layer_size,
                n_layer=config.n_layer,
                activation=config.activation_function
            )
           

            # Configura il trainer
            trainer = L.Trainer(
                logger=logger,
                accelerator=config.ACCELERATOR,
                devices=config.DEVICES,
                min_epochs=1,
                max_epochs=config.NUM_EPOCHS,
                precision=config.PRECISION,
                enable_progress_bar=True,
                callbacks=[
                    PrintingCallback(),
                    SaveBest(monitor="val_loss", logger=logger),
                    EarlyStopping(monitor="val_loss")
                ],
            )

            # Esegue il training
            trainer.fit(model, dm)
            trainer.validate(model, dm)
            #trainer.test(model, dm)
        return model, dm

# Esegue lo sweep
if __name__ == "__main__":
    model, dm = main()
