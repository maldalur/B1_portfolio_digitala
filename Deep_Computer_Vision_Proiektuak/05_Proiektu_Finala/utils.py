"""
Utility Functions - Deep Computer Vision Proiektuak
Egilea: Mikel Aldalur Corta
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Irudia kargatu eta pre-prozesatu
    
    Args:
        image_path (str): Irudiaren bidea
        target_size (tuple): Irudi tamaina (width, height)
    
    Returns:
        np.array: Pre-prozesatutako irudia
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalizatu
    return img


def plot_training_history(history, metrics=['accuracy', 'loss']):
    """
    Entrenamendu historiala bistaratu
    
    Args:
        history: Keras history objektua
        metrics (list): Bistaratu beharreko metrikak
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].plot(history.history[metric], label=f'Train {metric}', linewidth=2)
        if f'val_{metric}' in history.history:
            axes[i].plot(history.history[f'val_{metric}'], 
                        label=f'Validation {metric}', 
                        linestyle='--', 
                        linewidth=2)
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel(metric.capitalize(), fontsize=12)
        axes[i].set_title(f'{metric.capitalize()} over Epochs', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Confusion matrix bistaratu
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): Klase izenak
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def predict_image(model, image_path, class_names, target_size=(224, 224)):
    """
    Irudi bat aurresan eta emaitza bistaratu
    
    Args:
        model: Keras eredua
        image_path (str): Irudiaren bidea
        class_names (list): Klase izenak
        target_size (tuple): Irudi tamaina
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Irudia kargatu
    img = load_and_preprocess_image(image_path, target_size)
    img_display = cv2.imread(image_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    # Aurresana
    img_array = np.expand_dims(img, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Bistaratu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Irudia
    ax1.imshow(img_display)
    ax1.set_title(f'Predicted: {class_names[predicted_class_idx]}\\nConfidence: {confidence:.2%}',
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Probabilitate barra
    ax2.barh(class_names, predictions[0], color='skyblue')
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return class_names[predicted_class_idx], confidence


def save_model_with_metadata(model, filepath, metadata=None):
    """
    Eredua gorde metadata-rekin
    
    Args:
        model: Keras eredua
        filepath (str): Gordetzeko bidea
        metadata (dict): Metadata informazioa
    """
    model.save(filepath)
    
    if metadata:
        import json
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    print(f"✅ Eredua gordeta: {filepath}")
    if metadata:
        print(f"✅ Metadata gordeta: {metadata_path}")


def visualize_feature_maps(model, image_path, layer_name, num_filters=16):
    """
    Feature maps bistaratu
    
    Args:
        model: Keras eredua
        image_path (str): Irudiaren bidea
        layer_name (str): Geruza izena
        num_filters (int): Bistaratu beharreko filtroen kopurua
    """
    from tensorflow.keras.models import Model
    
    # Irudia kargatu
    img = load_and_preprocess_image(image_path, target_size=(224, 224))
    img_array = np.expand_dims(img, axis=0)
    
    # Feature maps lortu
    layer_output = Model(inputs=model.input, 
                        outputs=model.get_layer(layer_name).output)
    feature_maps = layer_output.predict(img_array, verbose=0)
    
    # Bistaratu
    num_rows = int(np.sqrt(num_filters))
    num_cols = int(np.ceil(num_filters / num_rows))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < num_filters and i < feature_maps.shape[-1]:
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.set_title(f'Filter {i+1}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Feature Maps - {layer_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("✅ Utils module importatuta!")
    print("📚 Funtzio erabilgarriak:")
    print("  - load_and_preprocess_image()")
    print("  - plot_training_history()")
    print("  - plot_confusion_matrix()")
    print("  - predict_image()")
    print("  - save_model_with_metadata()")
    print("  - visualize_feature_maps()")
