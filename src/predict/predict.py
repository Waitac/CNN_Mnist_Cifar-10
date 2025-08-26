import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

def load_model_and_predict():
    # 1. Carregar o modelo treinado
    model = load_model("models/mnist_model.h5")
    print("Modelo carregado com sucesso!")

    # 2. Carregar alguns dados de teste do MNIST
    (_, _), (x_test, y_test) = mnist.load_data()

    # Normalizar e achatar para o formato do modelo
    x_test_norm = x_test.astype("float32") / 255.0
    x_test_norm = x_test_norm.reshape((x_test_norm.shape[0], 28 * 28))

    # 3. Escolher uma amostra aleatória
    idx = np.random.randint(0, x_test.shape[0])
    sample_image = x_test[idx]
    sample_input = x_test_norm[idx].reshape(1, 784)

    # 4. Fazer a predição
    prediction = model.predict(sample_input)
    predicted_label = np.argmax(prediction)

    # 5. Mostrar resultado
    plt.imshow(sample_image, cmap="gray")
    plt.title(f"Verdadeiro: {y_test[idx]} | Previsto: {predicted_label}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    load_model_and_predict()
