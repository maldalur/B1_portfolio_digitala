# 🚀 Deep Computer Vision - Ikastaroa

**Egilea**: Mikel Aldalur Corta  
**Ikastetxea**: Instituto de Formación Profesional BIRT  
**Maila**: Aurreratua (2.B Ataza)  
**Iraupena**: 10-12 ordu

---

## 📚 Ikastaroaren Aurkezpena

### 🎯 Helburua

Ikastaro honek **maila aurreratuko ikasleei** Deep Learning eta Computer Vision teknika aurreratuak irakatsiko dizkie, proiektu errealetan aplikatzeko gai izan daitezen.

Aurre-ebaluazioan **8-10 puntu** lortu duten ikasleentzako edukia da.

---

## 🗂️ Ikastaroaren Egitura

Ikastaroa **5 atal nagusitan** banatuta dago, **9 Jupyter Notebook-ekin**:

### 📘 Atala 1: CNN Oinarriak
**Notebooks**: 2  
**Iraupena**: 2-3 ordu

- 🧠 **01_CNN_Sarrera.ipynb**
  - CNN kontzeptuaren sarrera
  - MLP vs CNN konparaketa
  - MNIST dataset-arekin praktika
  - Feature maps birtualizazioa

- 🎯 **02_Lehen_CNN_Eredua.ipynb**
  - CNN eredu bat zerotik eraikitzen
  - Hiper-parametroen optimizazioa
  - Confusion matrix eta metrikak
  - Model checkpoint eta early stopping

**Ikasiko duzuna**:
- ✅ CNN arkitektura oinarriak
- ✅ Konboluzio eta pooling geruzen funtzioa
- ✅ Lehen eredua entrenatu
- ✅ Eredua ebaluatu eta optimizatu

---

### 📗 Atala 2: Transfer Learning
**Notebooks**: 2  
**Iraupena**: 2-3 ordu

- 🔄 **03_ResNet_Transfer_Learning.ipynb**
  - Transfer Learning kontzeptua
  - ResNet50 ereduaren erabilera
  - Katuen vs. Txakurren sailkapena
  - Feature extraction vs Fine-tuning

- 🎨 **04_VGG16_Fine_Tuning.ipynb**
  - VGG16 arkitektura
  - Fine-tuning teknika aurreratuak
  - Data augmentation
  - Custom dataset-arekin lan egitea

**Ikasiko duzuna**:
- ✅ Transfer Learning teknikak
- ✅ Pre-trained ereduak erabili
- ✅ Fine-tuning aplikatu
- ✅ Dataset propiokin lan egin

---

### 📙 Atala 3: Objektu Detekzioa
**Notebooks**: 2  
**Iraupena**: 2-3 ordu

- 🎯 **05_YOLO_Sarrera.ipynb**
  - YOLO algoritmoaren sarrera
  - Objektu-detekzioaren oinarriak
  - Bounding boxes eta confidence scores
  - Non-Maximum Suppression (NMS)

- 🚀 **06_Objektu_Detekzio_Proiektua.ipynb**
  - YOLOv8 erabiliz objektuak detektatu
  - Irudietan eta bideoetan
  - Webcam-ekin denbora errealean
  - Custom objektuak entrenatu

**Ikasiko duzuna**:
- ✅ YOLO algoritmoaren funtzioa
- ✅ Objektuak irudietan detektatu
- ✅ Objektuak bideoetan detektatu
- ✅ Denbora errealean aplikatu

---

### 📕 Atala 4: Aurpegi Ezagutza
**Notebooks**: 2  
**Iraupena**: 2-3 ordu

- 👤 **07_Face_Recognition.ipynb**
  - Face Detection vs Face Recognition
  - Face embeddings (128D)
  - Aurpegiak sailkatu eta ezagutu
  - Face Recognition sistema osoa

- 😊 **08_Emotion_Detection.ipynb**
  - Emozio-detekzioa CNN-rekin
  - FER2013 dataset-a
  - 7 emozio: Happy, Sad, Angry, etc.
  - Kamerarekin denbora errealean

**Ikasiko duzuna**:
- ✅ Face Detection teknikak
- ✅ Face Recognition sistema garatu
- ✅ Emozioak detektatu
- ✅ Aplikazio errealak sortu

---

### 📓 Atala 5: Proiektu Finala
**Notebooks**: 1  
**Iraupena**: 2-3 ordu

- 🏆 **09_Proiektu_Osoa.ipynb**
  - Segurtasun Sistema konplexua
  - Multi-task learning
  - Ikasitako guztia konbinatu
  - Model deployment oinarriak
  - Flask API sortu (aukera)

**Ikasiko duzuna**:
- ✅ Proiektu osoa planifikatu
- ✅ Teknika guztiak konbinatu
- ✅ Sistema konplexua garatu
- ✅ Deployment oinarriak

---

## 🛠️ Behar diren Tresnak

### Software
- Python 3.8+
- Jupyter Notebook / JupyterLab
- Google Colab (GPU doakoa)

### Liburutegiak
```bash
pip install -r requirements.txt
```

**Liburutegi nagusiak**:
- TensorFlow 2.0+ / Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- scikit-learn
- Ultralytics (YOLO)
- face_recognition

### Hardware
- **Minimoa**: CPU (entrenamendua motela izango da)
- **Gomendatua**: GPU (NVIDIA CUDA)
- **Alternatiba**: Google Colab (GPU doan)

---

## 📊 Ebaluazio Sistema

### Notebook bakoitza:
- **Teoriatik**: Kontzeptuak ulertu (30%)
- **Praktika**: Kodea exekutatu eta ulertu (40%)
- **Proiektuak**: Jarduerak osatu (30%)

### Proiektu Finala:
- **Planifikazioa**: Proiektuaren diseinua (20%)
- **Inplementazioa**: Kode garbia eta eraginkorra (40%)
- **Emaitzak**: Accuracy, metrikak (30%)
- **Dokumentazioa**: README, komentarioak (10%)

---

## 🎓 Ikaskuntza Metodologia

### 1. **Teoria** (20%)
- Kontzeptuak azaldu Markdown gelaxketan
- Diagramak eta irudiak
- Erreferentziak paper zientifikoetara

### 2. **Praktika** (50%)
- Kode-adibideak gelaxketan
- Pausoz-pausoko azalpenak
- Emaitza bisualak

### 3. **Jarduerak** (30%)
- Erronkak notebook-aren amaieran
- Dataset desberdinekin probatu
- Parametroak aldatu eta konparatu

---

## 🚀 Nola Hastea

### 1. Klonatu Repositorioa
```bash
git clone https://github.com/maldalur/B1_portfolio_digitala.git
cd B1_portfolio_digitala/Deep_Computer_Vision_Proiektuak
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

---

## 📅 Ikastaroa Ordena Gomendatua

| Astea | Atala | Notebooks | Iraupena |
|-------|-------|-----------|----------|
| 1 | CNN Oinarriak | 01, 02 | 3h |
| 2 | Transfer Learning | 03, 04 | 3h |
| 3 | Objektu Detekzioa | 05, 06 | 3h |
| 4 | Aurpegi Ezagutza | 07, 08 | 3h |
| 5 | Proiektu Finala | 09 | 3h |

**GUZTIRA**: ~15 ordu (teoria + praktika)

---

## 💡 Aholkuak

### Ikasleentzat:
- ✅ Jarraitu notebook-en ordena
- ✅ Exekutatu gelaxka bakoitza banan-banan
- ✅ Aldatu parametroak eta esperimentatu
- ✅ Egin jarduera guztiak
- ✅ Galdetu zalantzak irakasleari

### Irakasleentzat:
- ✅ Gidatu ikasleei notebook-etan zehar
- ✅ Azaldu kontzeptu zailenak
- ✅ Lagundu debugging-ean
- ✅ Ebaluatu proiektu finala
- ✅ Eman feedback konstruktiboa

---

## 📚 Erreferentziak

1. **LeCun et al. (1998)** - Gradient-Based Learning Applied to Document Recognition
2. **Krizhevsky et al. (2012)** - ImageNet Classification with Deep CNNs (AlexNet)
3. **He et al. (2015)** - Deep Residual Learning for Image Recognition (ResNet)
4. **Redmon et al. (2016)** - You Only Look Once: Unified, Real-Time Object Detection (YOLO)
5. **Schroff et al. (2015)** - FaceNet: A Unified Embedding for Face Recognition

---

## 🏆 Zer Lortu

Ikastaro hau osatu ondoren gai izango zara:

✅ CNN ereduak zerotik eraikitzen  
✅ Transfer Learning aplikatzen  
✅ Objektuak denbora errealean detektatzen  
✅ Aurpegi-ezagutza sistemak garatzen  
✅ Emozio-detekzioa inplementatzen  
✅ Computer Vision proiektu konplexuak sortzen  
✅ Ereduak produkziora eramaten  

---

## 🌟 Hurrengo Pausuak

Ikastaro hau osatu ondoren:

1. **Zure Dataset Propioa**: Zure datu propioekin entrenatu
2. **API Sortu**: Flask/FastAPI erabiliz REST API
3. **Optimizatu**: TensorFlow Lite, ONNX
4. **Deploy**: AWS, Google Cloud, Azure
5. **Mobil App**: TensorFlow Lite erabiliz
6. **Portfolio**: GitHub-en argitaratu zure proiektuak

---

**Zorionak! Prest zaude Deep Computer Vision-en munduan sartzeko! 🎉**

**Hasteko**: Ireki `01_CNN_Oinarriak/01_CNN_Sarrera.ipynb` 🚀
