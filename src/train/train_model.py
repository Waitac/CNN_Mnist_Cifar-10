from src.data.preprocess import load_and_preprocess_data
from src.models.model import build_model
import os

def train_and_save_model():
    # 1. Carregar e preprocessar os dados
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # 2. Construir o modelo
    model = build_model()

    # 3. Treinar o modelo
    history = model.fit(
        x_train, y_train,
        epochs=5,              # Treinar por 5 épocas (rápido e funcional)
        batch_size=32,         # Atualiza os pesos a cada 32 imagens
        validation_split=0.1,  # Separa 10% do treino para validação
        verbose=1
    )

    # 4. Avaliar no conjunto de teste
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")

    # 5. Salvar o modelo treinado
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_model.h5")
    print("Modelo salvo em 'models/mnist_model.h5'")

    return model, history

if __name__ == "__main__":
    train_and_save_model()

