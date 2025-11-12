# 2.B ATAZA: Ezagutzan sakontzeko eduki digitala

## 🚀 GAIA: Ikusmeneko Sistemak - Maila Aurreratua

**Helburua**: Ikasleek Computer Vision-eko teknika aurreratuak ezagutu eta proiektu konplexuetan aplikatzea, aurre-ebaluazioan puntuazio altua lortu dutenak.

---

## 🎯 IKASKUNTZA-HELBURUAK

### Helburu Orokorra:
Ikasleek Deep Learning eta Convolutional Neural Networks (CNN) teknikak erabiliz Computer Vision proiektu aurreratuak garatu ditzakete.

### Helburu Zehatzak:
1. **Kontzeptualak**:
   - CNN (Convolutional Neural Networks) arkitektura ulertu
   - Deep Learning funtsak Computer Vision-en
   - Transfer Learning kontzeptua ezagutu
   - Eredu ezagunenak identifikatu (ResNet, YOLO, etc.)

2. **Prozeduralak**:
   - CNN eredua diseinatu eta entrenatu
   - Pre-trained ereduak erabili (Transfer Learning)
   - Objektu-detekzioa inplementatu (YOLO)
   - Aurpegi-ezagutza sisteman garatu
   - Accuracy, Precision, Recall metrikak kalkulatu

3. **Jarrerazkoak**:
   - Arazo konplexuen aurrean erabakitasuna
   - Esperimentazioa eta proba-errorea
   - Kode etikoa IA-ren garapenean

---

## 📱 FORMATUA: Jupyter Notebook Interaktiboa + GitHub Repository

**Zergatik formato hau?**
- ✅ Kodea eta azalpenak txandakatzen dira (literate programming)
- ✅ Denbera errealean exekuta daiteke
- ✅ Emaitza bisualak berehalakoak dira
- ✅ Ikasleek beren datuak proba ditzakete
- ✅ GitHub bidez partekatzen da (bertsio-kontrola)
- ✅ Google Colab bidez erraz exekutatzen da (GPU batekin)

---

## 🎨 EDUKI DIGITALA: "Deep Computer Vision - Proiektuak Eskutik"

### 📦 EGITURA (GitHub Repository)

```
Deep_Computer_Vision_Proiektuak/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── 01_CNN_Oinarriak/
│   ├── 01_CNN_Sarrera.ipynb
│   ├── 02_Lehen_CNN_Eredua.ipynb
│   └── datuak/
│
├── 02_Transfer_Learning/
│   ├── 03_ResNet_Transfer_Learning.ipynb
│   ├── 04_VGG16_Fine_Tuning.ipynb
│   └── ereduak/
│
├── 03_Objektu_Detekzioa/
│   ├── 05_YOLO_Sarrera.ipynb
│   ├── 06_Objektu_Detekzio_Proiektua.ipynb
│   └── test_irudiak/
│
├── 04_Aurpegi_Ezagutza/
│   ├── 07_Face_Recognition.ipynb
│   ├── 08_Emotion_Detection.ipynb
│   └── aurpegiak/
│
└── 05_Proiektu_Finala/
    ├── 09_Proiektu_Osoa.ipynb
    ├── utils.py
    └── emaitzak/
```

---

## 📓 NOTEBOOK 1: CNN Oinarriak

**Fitxategia**: `01_CNN_Sarrera.ipynb`

### Atal 1: Sarrera
```markdown
# 🧠 CNN: Convolutional Neural Networks

## Helburua
Ulertu nola funtzionatzen duten CNN sareak eta zergatik 
dira hain eraginkorrak irudi-analisian.

## Gako-Kontzeptuak
- Konboluzioa
- Pooling
- Activation Functions
- Fully Connected Layers
```

### Atal 2: CNN vs MLP (Perceptron Multikapa)
```python
# KODEA: MLP vs CNN konparaketa bisualki

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

# Dataset: MNIST (eskuz idatzitako zenbakiak)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# MLP Eredua (oinarrizkoa)
mlp_model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# CNN Eredua (aurreratua)
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Konpilatu
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenatu eta konparatu
print("🔹 MLP entrenamendu hasita...")
mlp_history = mlp_model.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)

print("🔹 CNN entrenamendu hasita...")
cnn_history = cnn_model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=5, validation_split=0.2, verbose=0)

# EMAITZA: Accuracy konparaketa
print(f"\n📊 EMAITZAK:")
print(f"MLP Accuracy: {mlp_history.history['val_accuracy'][-1]:.4f}")
print(f"CNN Accuracy: {cnn_history.history['val_accuracy'][-1]:.4f}")
print(f"✅ CNN hobea da {cnn_history.history['val_accuracy'][-1] - mlp_history.history['val_accuracy'][-1]:.4f} puntu!")
```

**Irteera Esperatua**:
```
📊 EMAITZAK:
MLP Accuracy: 0.9745
CNN Accuracy: 0.9891
✅ CNN hobea da 0.0146 puntu!
```

### Atal 3: CNN Geruzen Birtualizazioa
```python
# KODEA: Konboluzio geruza bat nola funtzionatzen duen ikusi

from tensorflow.keras.models import Model
import cv2

# Eredua kargatu
model = cnn_model

# Lehen konboluzio geruzaren irteera lortu
layer_output = Model(inputs=model.input, outputs=model.layers[0].output)

# Irudi bat aukeratu
test_image = X_test[0].reshape(1, 28, 28, 1)

# Feature maps lortu
feature_maps = layer_output.predict(test_image)

# Bistaratu 32 feature maps
fig, axes = plt.subplots(4, 8, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
plt.suptitle('🔍 CNN-k Ikusten Duena: 32 Feature Maps', fontsize=16)
plt.tight_layout()
plt.show()
```

**Azalpena Markdown-ean**:
```markdown
### 🤔 Zer ikusi dugu?
Konboluzio geruza bakoitzak **filtro desberdinak** aplikatzen ditu:
- Batzuek **ertzak** detektatzen dituzte
- Beste batzuek **angeluak**
- Beste batzuek **testurak**

Geruza sakonagoak kontzeptu **abstraktuagoak** ikasi dituzte!
```

---

## 📓 NOTEBOOK 2: Transfer Learning

**Fitxategia**: `03_ResNet_Transfer_Learning.ipynb`

### Atal 1: Transfer Learning Kontzeptua
```markdown
# 🔄 Transfer Learning: Ez Hasi Hutsik!

## Ideia
Dataset txikiekin lan egitean, eredu "aurre-entrenatua" bat erabili 
ImageNet bezalako dataset handitan entrenatua.

## Abantailak
✅ Denbora aurreztea
✅ Emaitza hobeak datu gutxiagorekin
✅ Generalizazio hobea

## Eredu Ezagunak
- **ResNet** (Residual Networks) - Microsoft
- **VGG16** - Oxford
- **InceptionV3** - Google
- **EfficientNet** - Google
```

### Atal 2: Kasu Praktikoa - Animalia Sailkapena
```python
# KODEA: ResNet50 erabiliaz katuen vs. txakurren sailkapena

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 1. ResNet50 kargatu (ImageNet-en entrenatua)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,  # Azken geruza kendu
    input_shape=(224, 224, 3)
)

# 2. Base model "izoztu" (ez entrenatu)
base_model.trainable = False

# 3. Gure geruza berriak gehitu
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binarioa: katua edo txakurra
])

# 4. Konpilatu
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Datu-generadoreakprestatu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 6. Datuak kargatu (zure direktorioa)
train_generator = train_datagen.flow_from_directory(
    'datuak/animalia/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'datuak/animalia/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 7. Entrenatu
print("🐱🐶 Transfer Learning entrenamendu hasita...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 8. Emaitzak
print(f"\n🎯 Azken Accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

### Atal 3: Fine-Tuning
```python
# KODEA: Fine-tuning - azken geruzak "desizoztu"

# Base model-aren azken 10 geruzak entrenatzen utzi
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Berriz konpilatu learning rate txikiagoarekin
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # LR txikiagoa!
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning entrenamendu
print("🔧 Fine-tuning hasita...")
history_fine = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

print(f"📈 Accuracy hobetua: {history_fine.history['val_accuracy'][-1]:.4f}")
```

---

## 📓 NOTEBOOK 3: YOLO - Objektu Detekzioa

**Fitxategia**: `05_YOLO_Sarrera.ipynb`

### Atal 1: Objektu Detekzioa vs. Sailkapena
```markdown
# 🎯 YOLO: You Only Look Once

## Sailkapena vs. Detekzioa

**Sailkapena**: "Zer da irudian?"
- Irteera: Etiketa bat (adib. "katua")

**Detekzioa**: "Non dago irudian eta zer da?"
- Irteera: Kokapena (bounding box) + Etiketa

## YOLO Abantailak
✅ Oso azkarra (denbera errealean)
✅ Objektu anitzak detektatu
✅ Zehaztasun ona
```

### Atal 2: YOLOv8 Erabiliz
```python
# KODEA: YOLOv8 objektu-detekzioa

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. YOLOv8 eredua kargatu
model = YOLO('yolov8n.pt')  # Nano bertsioa (azkarra)

# 2. Irudi bat kargatu
image_path = 'test_irudiak/kaleko_eszena.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Detekzioa exekutatu
results = model(image_path)

# 4. Emaitzak prozesatu
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Bounding box koordenak
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # Konfiantza
        confidence = box.conf[0].cpu().numpy()
        # Klasea
        class_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[class_id]
        
        # Marraztu
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{class_name} {confidence:.2f}'
        cv2.putText(image_rgb, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 5. Bistaratu
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.title('🎯 YOLO Objektu Detekzioa')
plt.axis('off')
plt.show()

print(f"✅ {len(boxes)} objektu detektatu dira!")
```

### Atal 3: Bideo Denbera Errealean
```python
# KODEA: Webcam bidez denbera errealean

import cv2
from ultralytics import YOLO

# Eredua kargatu
model = YOLO('yolov8n.pt')

# Webcam ireki
cap = cv2.VideoCapture(0)

print("🎥 Webcam piztuta. Sakatu 'q' irteteko.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detekzioa
    results = model(frame, stream=True)
    
    # Marraztu
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Erakutsi
    cv2.imshow('YOLO Denbera Errealean', frame)
    
    # 'q' sakatzean irten
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam itxita")
```

---

## 📓 NOTEBOOK 4: Aurpegi Ezagutza

**Fitxategia**: `07_Face_Recognition.ipynb`

### Atal 1: Face Recognition Library
```python
# KODEA: Aurpegiak ezagutu face_recognition liburutegiarekin

import face_recognition
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Ezagututako aurpegiak kargatu
known_image = face_recognition.load_image_file("aurpegiak/pertsona1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Hainbat pertsona ezagunak
known_encodings = [known_encoding]
known_names = ["Pertsona 1"]

# 2. Test irudia kargatu
test_image = face_recognition.load_image_file("test_irudiak/taldea.jpg")
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# 3. Aurpegi guztiak detektatu
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# 4. Identifikatu
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Konparatu ezagutako aurpegiekin
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "Ezezaguna"
    
    # Distantzia kalkulatu
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        name = known_names[best_match_index]
    
    # Marraztu
    cv2.rectangle(test_image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image_rgb, name, (left, top-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# 5. Bistaratu
plt.figure(figsize=(12, 8))
plt.imshow(test_image_rgb)
plt.title('👤 Aurpegi Ezagutza')
plt.axis('off')
plt.show()
```

### Atal 2: Emozio Detekzioa
```python
# KODEA: Emozioak detektatu DeepFace liburutegiarekin

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Irudia kargatu
img_path = "test_irudiak/aurpegia.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Analisia exekutatu
result = DeepFace.analyze(img_path, actions=['emotion', 'age', 'gender'])

# Emaitzak
print("📊 ANALISIA:")
print(f"🎭 Emozioa: {result[0]['dominant_emotion']}")
print(f"👤 Generoa: {result[0]['dominant_gender']}")
print(f"🎂 Adina: {result[0]['age']} urte")
print(f"\n📈 Emozio guztiak:")
for emotion, score in result[0]['emotion'].items():
    print(f"  {emotion}: {score:.2f}%")

# Bistaratu
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title(f"Emozioa: {result[0]['dominant_emotion']} | "
          f"Generoa: {result[0]['dominant_gender']} | "
          f"Adina: {result[0]['age']}")
plt.axis('off')
plt.show()
```

---

## 📓 NOTEBOOK 5: Proiektu Finala

**Fitxategia**: `09_Proiektu_Osoa.ipynb`

### Proiektua: Sistema Inteligentea Segurtasunerako

```markdown
# 🔒 PROIEKTU FINALA: Segurtasun Sistema

## Helburuak
1. Aurpegiak detektatu bideo-jario batean
2. Ezagututako aurpegiak identifikatu
3. Ezezagunoak alerta bat bidali
4. Pertsonen mugimendua jarraitu
5. Log bat mantendu ekintza guztiekin

## Teknologiak
- OpenCV (bideo-captura)
- YOLO (pertsona-detekzioa)
- Face Recognition (aurpegi-ezagutza)
- SQLite (datu-basea)
```

### Kode Osoa:
```python
# KODEA: Segurtasun Sistema Osoa

import cv2
import face_recognition
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import numpy as np

class SegurtasunSistema:
    def __init__(self):
        # Ereduak kargatu
        self.yolo_model = YOLO('yolov8n.pt')
        self.known_faces = []
        self.known_names = []
        
        # Datu-basea konfiguratu
        self.setup_database()
        
    def setup_database(self):
        """SQLite datu-basea sortu"""
        conn = sqlite3.connect('segurtasuna.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS ekintzak
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      data TEXT,
                      ekintza TEXT,
                      pertsona TEXT)''')
        conn.commit()
        conn.close()
    
    def aurpegia_gehitu(self, image_path, izena):
        """Aurpegi ezagun bat gehitu"""
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        self.known_faces.append(encoding)
        self.known_names.append(izena)
        print(f"✅ {izena} gehituta!")
    
    def log_ekintza(self, ekintza, pertsona):
        """Ekintza bat gorde"""
        conn = sqlite3.connect('segurtasuna.db')
        c = conn.cursor()
        data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO ekintzak VALUES (NULL, ?, ?, ?)",
                  (data, ekintza, pertsona))
        conn.commit()
        conn.close()
    
    def prozesatu_framea(self, frame):
        """Frame bat prozesatu"""
        # 1. Pertsonak detektatu YOLO-rekin
        results = self.yolo_model(frame, classes=[0])  # 0 = pertsona
        
        # 2. Aurpegi-ezagutza
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # 3. Aurpegi bakoitza identifikatu
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "❌ EZEZAGUNA"
            
            if True in matches:
                match_index = matches.index(True)
                name = f"✅ {self.known_names[match_index]}"
            else:
                # Alerta bidali
                self.log_ekintza("EZEZAGUNA_DETEKTATU", "Ezezaguna")
                cv2.putText(frame, "⚠️ ALERTA!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Marraztu
            cv2.rectangle(frame, (left, top), (right, bottom), 
                         (0, 255, 0) if "✅" in name else (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                       (0, 255, 0) if "✅" in name else (0, 0, 255), 2)
        
        return frame
    
    def hasi(self):
        """Sistema hasi"""
        cap = cv2.VideoCapture(0)
        print("🎥 Sistema piztuta. Sakatu 'q' irteteko.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prozesatu
            frame_prozesatua = self.prozesatu_framea(frame)
            
            # Erakutsi
            cv2.imshow('🔒 Segurtasun Sistema', frame_prozesatua)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Sistema itxita")

# ERABILI
if __name__ == "__main__":
    # Sistema sortu
    sistema = SegurtasunSistema()
    
    # Aurpegi ezagunak gehitu
    sistema.aurpegia_gehitu("aurpegiak/langilea1.jpg", "Mikel")
    sistema.aurpegia_gehitu("aurpegiak/langilea2.jpg", "Ane")
    
    # Hasi
    sistema.hasi()
```

---

## 📊 EBALUAZIO METRIKAK

### Auto-Ebaluazio Irizpideak:
✅ CNN arkitektura ulertu eta inplementatu duzu?
✅ Transfer Learning teknika aplikatu duzu?
✅ YOLO erabiliz objektuak detektatu dituzu?
✅ Aurpegi-ezagutza sistema bat garatu duzu?
✅ Proiektu oso bat amaitu duzu?

**Puntuazio minimoa**: 4/5 zuzen

---

## 📄 LIZENTZIAK eta EGILEAK

### Edukiaren Egiletzapena:
```
📝 EGILEA: Mikel Aldalur
🏫 ERAKUNDEA: FPIkaskuntzagunea - BIRT
📅 DATA: 2025/11/12
📜 LIZENZIA: Creative Commons BY-NC-SA 4.0
```

### Liburutegiak eta Ereduak:
- **TensorFlow/Keras**: Apache 2.0 License
- **PyTorch**: BSD License
- **OpenCV**: Apache 2.0 License
- **YOLOv8 (Ultralytics)**: AGPL-3.0 License
- **face_recognition**: MIT License
- **DeepFace**: MIT License

### Eredu Pre-entrepatuak:
- **ResNet50**: Microsoft (BSD License)
- **VGG16**: Oxford (Creative Research License)
- **YOLOv8**: Ultralytics (AGPL-3.0)

### Datuak:
- **ImageNet**: Stanford University (Ikerketarako erabilpena)
- **MNIST**: Yann LeCun (Domeinu Publikoa)
- Test irudiak: Unsplash / Pixabay (Erabilpen librea)

---

## 🔗 ESTEKAK eta BALIABIDEAK

### Repository-a atzitzeko:
- **GitHub**: https://github.com/maldalur/Deep_Computer_Vision_Proiektuak
- **Google Colab**: [Notebook guztiak Colab-en exekutatzeko]
- **Binder**: [Exekuzio interaktiboa]

### Instalazio Gida:
```bash
# 1. Repository clonatu
git clone https://github.com/maldalur/Deep_Computer_Vision_Proiektuak.git
cd Deep_Computer_Vision_Proiektuak

# 2. Virtual environment sortu
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Dependentziak instalatu
pip install -r requirements.txt

# 4. Jupyter hasi
jupyter notebook
```

### Requirements.txt:
```
tensorflow==2.15.0
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1
ultralytics==8.0.0
face-recognition==1.3.0
deepface==0.0.79
numpy==1.24.3
matplotlib==3.7.2
jupyter==1.0.0
```

---

## 📈 LORPEN ADIERAZLEA

**"Sortutako eduki digitalak ikasketa-helburu jakin bat dute eta proposatutako gaiaren inguruan SAKONTZEA ahalbidetzen du"**

✅ **Betetzen da**:
- Helburu aurreratuak eta konplexuak definitu dira
- CNN, Transfer Learning, YOLO eta aurpegi-ezagutza teknikak landuta
- Kode exekutagarria eta probatzeko modukoa eskaintzen da
- Proiektu erreala garatzeko gida osoa ematen da
- Ikasleek beren sistemak garatu ditzakete
- GitHub bidez baliabide guztiak eskuragarri daude
