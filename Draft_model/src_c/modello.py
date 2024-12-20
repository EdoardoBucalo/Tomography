# main.py
from model_c import TomoModel  # Importa il modello definito nel file model.py
from torchsummary import summary
import config

# Crea una istanza del modello
model = TomoModel(
                input_size=config.INPUTSIZE,
                output_channels=config.OUTPUTSIZE,
                learning_rate=config.LEARNING_RATE,
                feature_map_size=(25, 25),
                
                
                
                k_0=3,
                k_1=4,
                k_2=4,

                k_5=4,
                k_6=5,
                k_7=6,
                k_8=6,
                
                s_1=2,
                s_2=1,
                s_3=1,

                s_6=2,
                s_7=1,
                s_8=1,
                s_9=1,
                
                
                )

# Visualizza il sommario del modello
summary(model, input_size=(92,))
