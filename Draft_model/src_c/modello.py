# main.py
from model import TomoModel  # Importa il modello definito nel file model.py
from torchsummary import summary
import config

# Crea una istanza del modello
model = TomoModel(
                input_size=config.INPUTSIZE,
                output_channels=config.OUTPUTSIZE,
                learning_rate=config.LEARNING_RATE,
                feature_map_size=(25, 25),
                
                
                kernel_size=4,
                k0_size=4,
                k1_size=4,
                k2_size=5,
                kf_size=6,
                
                
                padding=1,
                stride=1,
                stride2=2,
                output_padding=0
               
                
                )

# Visualizza il sommario del modello
summary(model, input_size=(92,))
