import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Inicializa Flask
app = Flask(__name__)

# Carrega modelo treinado
mnist_model = load_model("models/mnist_model.h5")
cifar_model = load_model("models/cifar10_cnn.h5")

cifar_classes = ['avião','automóvel','pássaro','gato','cervo',
                 'cachorro','sapo','cavalo','navio','caminhão']


# Função para pré-processar a imagem - MODIFICADA
def preprocess_mnist(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # escala de cinza
    img = cv2.bitwise_not(img)  # inverte (MNIST = fundo preto, número branco)
    img = cv2.resize(img, (28, 28))  # redimensiona
    img = img / 255.0  # normaliza
    img = img.reshape(1, 784)  # ACHATA para [1, 784] - SOLUÇÃO DO ERRO
    return img


# Função para pré-processar CIFAR-10
def preprocess_cifar(img_path):
    img = cv2.imread(img_path)  # colorida
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte BGR → RGB
    img = cv2.resize(img, (32, 32))  # redimensiona
    img = img / 255.0  # normaliza
    img = img.reshape(1, 32, 32, 3)  # formato correto para CNN
    return img


@app.route("/")
def index():
    return render_template("index.html")


# Página MNIST
@app.route("/mnist", methods=["GET", "POST"])
def mnist():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return "Nenhum arquivo enviado"
        file = request.files["file"]
        if file.filename == "":
            return "Nenhum arquivo selecionado"
        
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        img = preprocess_mnist(filepath)
        pred = mnist_model.predict(img)
        prediction = int(pred.argmax())

        os.remove(filepath)

    return render_template("mnist.html", prediction=prediction)


# Página CIFAR-10
@app.route("/cifar", methods=["GET", "POST"])
def cifar():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return "Nenhum arquivo enviado"
        file = request.files["file"]
        if file.filename == "":
            return "Nenhum arquivo selecionado"
        
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        img = preprocess_cifar(filepath)
        pred = cifar_model.predict(img)
        prediction = cifar_classes[int(pred.argmax())]

        os.remove(filepath)

    return render_template("cifar.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)