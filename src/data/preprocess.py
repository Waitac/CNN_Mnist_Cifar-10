import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    # 1. Carregar o dataset MNIST (já dividido em treino e teste)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Normalizar os valores dos pixels (0-255 -> 0-1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3. Achatar as imagens (28x28 -> 784) para redes densas
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))

    # 4. Converter os rótulos para one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    # Executar um teste rápido ao rodar o arquivo
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("Formato de x_train:", x_train.shape)
    print("Formato de y_train:", y_train.shape)
    print("Formato de x_test:", x_test.shape)
    print("Formato de y_test:", y_test.shape)

