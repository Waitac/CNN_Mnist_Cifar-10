# train_cifar10_improved.py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Carregar e preparar os dados
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. DATA AUGMENTATION (CRUCIAL)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),     # Rotação de até 10%
    layers.RandomZoom(0.1),         # Zoom de até 10%
    layers.RandomContrast(0.2),     # Variação de contraste
])

# 3. Arquitetura mais profunda e com Batch Normalization
model = models.Sequential([
    # Camada de aumento de dados
    layers.Input(shape=(32, 32, 3)),
    data_augmentation,
    
    # Bloco Convolucional 1
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),  # Dropout após pooling
    
    # Bloco Convolucional 2
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    # Bloco Convolucional 3
    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    
    # Camadas Densas
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Dropout mais alto nas camadas densas
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. Callbacks para treinamento inteligente
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# 5. Compilar com parâmetros ajustados
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Treinar por mais épocas
history = model.fit(
    x_train, y_train,
    epochs=100,  # Treinamos por mais tempo, mas os callbacks vão parar antes se necessário
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 7. Salvar o modelo
model.save("models/cifar10_cnn.keras")

# 8. Avaliar o modelo final
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final no teste: {test_acc:.4f}")
