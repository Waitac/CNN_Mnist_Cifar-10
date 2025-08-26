import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model():
    model = Sequential([
        # Camada oculta com 128 neurônios e ativação ReLU
        Dense(128, activation="relu", input_shape=(784,)),
        
        # Camada de saída com 10 neurônios (softmax para classificação multiclasse)
        Dense(10, activation="softmax")
    ])

    # Compilar o modelo com otimizador, função de perda e métricas
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    # Teste rápido
    model = build_model()
    model.summary()

