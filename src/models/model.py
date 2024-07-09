import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, utils
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(data_dir, categories, limit=None):
    """Cargar datos del dataset de QuickDraw en formato .npy"""
    images = []
    labels = []
    for idx, category in enumerate(categories):
        file_path = os.path.join(data_dir, f"full_numpy_bitmap_{category}.npy").replace("\\", "/")
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            continue
        data = np.load(file_path)
        if limit:
            data = data[:limit]
        images.append(data)
        labels.append(np.full(data.shape[0], idx))
    
    if len(images) == 0:
        return None, None
    
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

def preprocess_data(images, labels, num_classes):
    """Preprocesar los datos: normalizar y one-hot encode"""
    x = images.astype('float32') / 255.0
    x = x.reshape(-1, 28, 28, 1)
    y = utils.to_categorical(labels, num_classes)
    return x, y

def data_generator(x, y, batch_size):
    """Generador para lotes de datos"""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)
    return dataset

def build_model(num_classes):
    """Construir un modelo de red neuronal convolucional"""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(5, 5), padding="valid", activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, kernel_size=(5, 5), padding="valid", activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(512),
        layers.Dropout(0.5),
        layers.Dense(128),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history):
    """Graficar la precisión del entrenamiento y validación"""
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=history.history['accuracy'], label='Accuracy')
    sns.lineplot(x=epochs, y=history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    data_dir = 'data'  # Modifica esta ruta según sea necesario
    categories = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer", "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]
    num_classes = len(categories)
    batch_size = 64
    limit = 10000 

    images, labels = load_data(data_dir, categories, limit)
    
    if images is None or labels is None:
        print("Failed to load data. Exiting...")
        return

    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_x, train_y = preprocess_data(train_x, train_y, num_classes)
    test_x, test_y = preprocess_data(test_x, test_y, num_classes)

    train_dataset = data_generator(train_x, train_y, batch_size)
    test_dataset = data_generator(test_x, test_y, batch_size)
    
    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    
    history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[early_stopping])
    plot_history(history)

    # Evaluación del modelo con los datos de prueba
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Accuracy: {accuracy}, Loss: {loss}')
    
    model.save('model.keras')

if __name__ == "__main__":
    main()
