# 04 - Aurpegi Ezagutza

## 📖 Deskribapena

Face Recognition eta Emotion Detection teknikak erabiliz aurpegi-ezagutza sistemak garatuko ditugu.

## 📓 Notebooks

### 01_Aurpegi_Ezagutza_Sarrera.ipynb
- ✅ **Face Detection vs Face Recognition**: Bi teknika desberdintzea
- ✅ **Haar Cascades**: OpenCV aurpegi detekziorako (klasikoa)
- ✅ **DNN Face Detector**: Deep Learning aurpegi detekziorako (modernoa)
- ✅ **Face Embeddings**: 128 dimentsioko bektorea aurpegi bakoitzeko
- ✅ **Triplet Loss**: FaceNet ereduaren entrenamendua (antzekotasun ikaskuntza)

### 02_Face_Recognition_Praktika.ipynb
- ✅ **face_recognition liburutegia**: dlib-en gainean eraikitako liburutegia
- ✅ **68 facial landmarks**: Aurpegiaren puntu nagusiak (begiak, sudurra, ahoa...)
- ✅ **Aurpegi ezagutu**: Zein pertsona den zehaztu (database batekin)
- ✅ **Irudi estatikoak**: Argazki batean aurpegiak identifikatu
- ✅ **Webcam denbora errealean**: Kamerarekin zuzenean aurpegiak ezagutu

## 📂 aurpegiak/

Aurpegi irudiak karpeta honetan jarri ditzakezu Face Recognition entrenatzeko.

Karpeta egitura:
```
aurpegiak/
├── pertsona1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── pertsona2/
│   ├── 1.jpg
│   └── ...
```

## 🎯 Helburuak

- ✅ Face Detection teknikak ezagutu
- ✅ Face Recognition sistema garatu
- ✅ Emozioak detektatu
- ✅ Denbora errealean aplikatu
