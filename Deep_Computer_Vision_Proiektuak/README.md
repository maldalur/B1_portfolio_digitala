# 🚀 Deep Computer Vision - Proiektuak Eskutik

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Maila aurreratuko Computer Vision proiektuak - Jupyter Notebooks with CNN, Transfer Learning, YOLO, eta Face Recognition

---

## 📖 Deskribapena

Baliabide hau **2.B ATAZA**ren parte da (Ezagutzan Sakontzeko Eduki Digitala) eta **maila aurreratuko ikasleentzat** diseinatuta dago. Aurre-ebaluazioan **8-10 puntu** lortu dutenek eduki hau erabiliko dute.

**Helburua**: Deep Learning eta CNN teknikak erabiliz Computer Vision proiektu konplexuak garatzea.

---

## 🎯 Ikaskuntza Helburuak

### Kontzeptualak:
- ✅ CNN (Convolutional Neural Networks) arkitektura ulertu
- ✅ Deep Learning funtsak Computer Vision-en
- ✅ Transfer Learning kontzeptua ezagutu
- ✅ Eredu ezagunenak identifikatu (ResNet, YOLO, VGG16)

### Prozeduralak:
- ✅ CNN eredua diseinatu eta entrenatu
- ✅ Pre-trained ereduak erabili (Transfer Learning)
- ✅ Objektu-detekzioa inplementatu (YOLO)
- ✅ Aurpegi-ezagutza sisteman garatu
- ✅ Accuracy, Precision, Recall metrikak kalkulatu

### Jarrerazkoak:
- ✅ Arazo konplexuen aurrean erabakitasuna
- ✅ Esperimentazioa eta proba-errorea
- ✅ Kode etikoa IA-ren garapenean

---

## 📁 Egitura

```
Deep_Computer_Vision_Proiektuak/
│
├── README.md                        # Dokumentu hau
├── requirements.txt                 # Python dependentziak
├── LICENSE                          # CC BY-NC-SA 4.0
│
├── 01_CNN_Oinarriak/               # CNN sarrera
│   ├── 01_CNN_Sarrera.ipynb
│   ├── 02_Lehen_CNN_Eredua.ipynb
│   └── datuak/
│
├── 02_Transfer_Learning/           # Transfer Learning
│   ├── 03_ResNet_Transfer_Learning.ipynb
│   ├── 04_VGG16_Fine_Tuning.ipynb
│   └── ereduak/
│
├── 03_Objektu_Detekzioa/          # YOLO objektu-detekzioa
│   ├── 05_YOLO_Sarrera.ipynb
│   ├── 06_Objektu_Detekzio_Proiektua.ipynb
│   └── test_irudiak/
│
├── 04_Aurpegi_Ezagutza/           # Face Recognition
│   ├── 07_Face_Recognition.ipynb
│   ├── 08_Emotion_Detection.ipynb
│   └── aurpegiak/
│
└── 05_Proiektu_Finala/            # Proiektu konplexua
    ├── 09_Proiektu_Osoa.ipynb
    ├── utils.py
    └── emaitzak/
```

---

## 📓 Jupyter Notebooks Zerrenda

### 1️⃣ CNN Oinarriak
- **01_CNN_Sarrera.ipynb**: CNN-en oinarriak, konboluzioa, pooling
- **02_Lehen_CNN_Eredua.ipynb**: MNIST dataset-arekin lehen CNN eredua

### 2️⃣ Transfer Learning
- **03_ResNet_Transfer_Learning.ipynb**: ResNet50 erabiliz transfer learning
- **04_VGG16_Fine_Tuning.ipynb**: VGG16 fine-tuning teknikak

### 3️⃣ Objektu Detekzioa
- **05_YOLO_Sarrera.ipynb**: YOLO (You Only Look Once) sarrera
- **06_Objektu_Detekzio_Proiektua.ipynb**: Objektu-detekzio proiektu osoa

### 4️⃣ Aurpegi Ezagutza
- **07_Face_Recognition.ipynb**: Aurpegi-ezagutza teknikak
- **08_Emotion_Detection.ipynb**: Emozio-detekzioa kamerarekin

### 5️⃣ Proiektu Finala
- **09_Proiektu_Osoa.ipynb**: Proiektu konplexua ikasitako guztia erabiliz

---

## 🚀 Nola Erabili

### 1. Klonatu Repositorioa
```bash
git clone https://github.com/maldalur/Deep_Computer_Vision_Proiektuak.git
cd Deep_Computer_Vision_Proiektuak
```

### 2. Instalatu Dependentziak
```bash
pip install -r requirements.txt
```

### 3. Ireki Jupyter Notebook
```bash
jupyter notebook
```

### 4. Google Colab-en Exekutatu (GPU-rekin)
Notebook bakoitzaren goiko aldean "Open in Colab" botoia sakatu

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maldalur/Deep_Computer_Vision_Proiektuak)

---

## 📦 Dependentziak

- Python 3.8+
- TensorFlow 2.0+
- Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- ultralytics (YOLO)
- face_recognition

Ikusi `requirements.txt` zerrenda osoa ikusteko.

---

## 📊 Dataset-ak

Proiektu hauek dataset ezberdinak erabiltzen dituzte:

1. **MNIST**: Eskuz idatzitako zenbakiak (28x28 pixel)
2. **CIFAR-10**: 10 klase (airplanes, cars, birds, etc.)
3. **ImageNet**: Transfer Learning-erako
4. **COCO**: Objektu-detekziorako
5. **LFW (Labeled Faces in the Wild)**: Aurpegi-ezagutzarako

Dataset guztiak automatikoki deskargatzen dira notebook-etan.

---

## 🎓 Proiektuaren Fluxua

```mermaid
graph TD
    A[1. CNN Oinarriak] --> B[2. Transfer Learning]
    B --> C[3. Objektu Detekzioa]
    B --> D[4. Aurpegi Ezagutza]
    C --> E[5. Proiektu Finala]
    D --> E
```

**Gomendatutako ordena**: 01 → 02 → 03 → 04 → ... → 09

---

## 💡 Gako Kontzeptuak

### CNN (Convolutional Neural Networks)
Sare neuronalak irudiak prozesatzeko diseinatuak:
- **Konboluzio geruza**: Karakteristikak atera (ertzak, testurak)
- **Pooling geruza**: Dimentsioa murriztu
- **Fully Connected geruza**: Klasifikazioa

### Transfer Learning
Pre-trained eredu bat hartu eta dataset berrira egokitu:
- **Fine-tuning**: Geruza batzuk berriz entrenatu
- **Feature extraction**: Geruza geldituak erabili

### YOLO (You Only Look Once)
Objektu-detekzio errealean denbora errealean:
- Irudi osoa behin bakarrik prozesatu
- Azkarragoa R-CNN baino
- Objektuak eta bounding box-ak detektatu

---

## 🔧 Troubleshooting

### GPU ez dago erabilgarri?
```python
import tensorflow as tf
print("GPU erabilgarri:", tf.config.list_physical_devices('GPU'))
```

Google Colab erabili GPU doako baterako.

### Memoria arazoak?
Batch size-a murriztu:
```python
model.fit(X_train, y_train, batch_size=16)  # 32 edo 64 ordez
```

### Dataset ez da deskargatu?
Manualki deskargatu eta `datuak/` karpetan jarri.

---

## 📝 Lizenzia

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

- ✅ Partekatu eta moldatu
- ✅ Aipamena eman
- ❌ Ez komertziala
- ✅ Lizentzia berbera mantendu

---

## 👤 Egilea

**Mikel Aldalur Corta**  
Irakaslea - Instituto de Formación Profesional BIRT  
📧 maldalur@birt.eus  
🌐 [Portfolio Digitala](https://maldalur.github.io/B1_portfolio_digitala/)

---

## 🌟 Eskerrak

Dataset-ak eta ereduak:
- TensorFlow & Keras
- PyTorch
- Ultralytics (YOLO)
- OpenCV
- face_recognition liburutegia

---

## 📚 Erreferentziak

1. **LeCun et al. (1998)** - Gradient-Based Learning Applied to Document Recognition
2. **Krizhevsky et al. (2012)** - ImageNet Classification with Deep CNNs
3. **He et al. (2015)** - Deep Residual Learning for Image Recognition
4. **Redmon et al. (2016)** - You Only Look Once: Unified, Real-Time Object Detection
5. **Schroff et al. (2015)** - FaceNet: A Unified Embedding for Face Recognition

---

## 🚀 Hurrengo Pausuak

Proiektu hauek osatu ondoren:
1. ✅ Zure dataset propioa erabili
2. ✅ Eredua produkziora eraman (Flask API)
3. ✅ Eredua optimizatu (TensorFlow Lite)
4. ✅ Cloud-era deploy egin (AWS, Google Cloud)

---

**Zorionak! Prest zaude Computer Vision proiektu aurreratuak garatzen hasteko! 🎉**
