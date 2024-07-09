import cv2 as cv
import numpy as np
import math
from keras.models import load_model

class CanvasProcessing:
    def __init__(self, model_path):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None
        self.categories = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer", "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]

    @staticmethod
    def refine_image(image):
        """Refina la imagen para asegurar que el tamaño y el padding sean correctos."""
        PADDING_SIZE, TARGET_SIZE = 22, 28
        rows, cols = image.shape
        
        # Determinar el factor de escala
        factor = PADDING_SIZE / max(rows, cols)
        rows, cols = int(round(rows * factor)), int(round(cols * factor))
        
        # Redimensionar la imagen
        resized_image = cv.resize(image, (cols, rows))
        
        cols_padding = (int(math.ceil((TARGET_SIZE - cols) / 2.0)), int(math.floor((TARGET_SIZE - cols) / 2.0)))
        rows_padding = (int(math.ceil((TARGET_SIZE - rows) / 2.0)), int(math.floor((TARGET_SIZE - rows) / 2.0)))
        
        padded_image = np.pad(resized_image, (rows_padding, cols_padding), 'constant')
        return padded_image
    
    def extract_and_refine(self, image):
        """Extrae el contorno de la imagen y lo refina."""
        # Convertir a escala de grises y umbralizar
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Crear una máscara en blanco
        mask = np.zeros_like(binary)
        
        # Dibujar el contorno más grande en la máscara
        if contours:
            c = max(contours, key=cv.contourArea)
            cv.drawContours(mask, [c], -1, 255, -1)
        
        # Encontrar las coordenadas del contorno más grande
        x, y, w, h = cv.boundingRect(mask)
        roi = binary[y:y+h, x:x+w]
        
        # Redimensionar la ROI a 28x28
        resized = cv.resize(roi, (28, 28), interpolation=cv.INTER_AREA)
        return resized

    def predict_image(self, image):
        """Realiza la predicción de un carácter usando el modelo cargado."""
        if self.model is None:
            return None
        image = image.astype('float32') / 255.0
        image = image.reshape(-1, 28, 28, 1)
        prediction = np.argmax(self.model.predict(image))
        return self.categories[prediction]  # Mapea el índice a la etiqueta correspondiente

    def process_image(self, path):
        """Procesa una imagen completa y devuelve la predicción del modelo."""
        image = cv.imread(path)
        if image is None:
            print(f"Error al leer la imagen: {path}")
            return "No se encontró dibujo"
        refined_image = self.extract_and_refine(image)
        if refined_image is None:
            return "No se encontró dibujo"
        predicted_label = self.predict_image(refined_image)
        return predicted_label
