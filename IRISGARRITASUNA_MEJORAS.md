# ♿ IRISGARRITASUN HOBEKUNTZAK / MEJORAS DE ACCESIBILIDAD

## 📋 Laburpena / Resumen

Portfolio digitalean **irisgarritasun elementu** ugari gehitu dira 5.A Ataza betetzeko. Eduki digital guztiak (bidalketa1 eta bidalketa2) orain **irisgarriak** dira ikasle guztientzat.

---

## 🎯 Gehitutako Irisgarritasun Elementuak

### 1️⃣ **Bideoen Azpitituluak / Subtítulos en Videos**

**Non:** Bidalketa1, Bidalketa2, Index (Sarrera)

**Zer egin da:**
- YouTube bideo guztiei `?cc_load_policy=1&hl=eu` parametroa gehitu da URL-an
- Azpitituluak automatikoki kargatzen dira (euskaraz)
- Erabiltzaileak 'CC' botoia sakatu behar du azpitituluak aktibatzeko

**Onurak:**
- ✅ Entzumen arazoak dituzten ikasleentzako laguntza
- ✅ Bideo edukiak testu formatuan eskuragarri
- ✅ Ikasleak beren erritmoan jarrai dezakete

**Adibidea:**
```html
<iframe src="https://www.youtube.com/embed/VIDEO_ID?cc_load_policy=1&hl=eu"></iframe>
```

---

### 2️⃣ **Irakurketa Modua / Modo de Lectura**

**Non:** Bidalketa1, Bidalketa2, Index

**Zer egin da:**
- Botoi bat gehitu da (♿) orriaren beheko eskuin aldean
- Menú bat irekitzen da irisgarritasun aukerekin
- "Irakurketa modua" hautatzean:
  - Nabigazio-menua ezkutatzen da
  - Edukia zentratzen da (max-width: 800px)
  - Letra-tamaina handitzen da (1.1em)
  - Lerro-tartea zabaltzen da (1.8)

**Onurak:**
- ✅ Fokua edukian jartzen du
- ✅ Distrazio elementuak kentzen ditu
- ✅ Irakurketa errazagoa egiten du
- ✅ Dislexia edo irakurketa zailtasunak dituztenentzat egokia

**CSS Aldaketak:**
```css
body.reading-mode .container {
    max-width: 800px;
    font-size: 1.1em;
    line-height: 1.8;
}

body.reading-mode .main-nav {
    display: none;
}
```

---

### 3️⃣ **Kontraste Altua / Alto Contraste**

**Non:** Bidalketa1, Bidalketa2, Index

**Zer egin da:**
- "Kontraste altua" aukera menuan
- Aktibatzean:
  - Atzekoa: Beltza (#000)
  - Testua: Zuria (#fff)
  - Tituluak: Horia (#ffff00)
  - Estekak: Cyan (#00ffff)

**Onurak:**
- ✅ Ikusmen arazoak dituzten ikasleentzako laguntza
- ✅ Irakurketa errazagoa argi baldintza txarrenetan
- ✅ WCAG 2.1 AA arauak betetzen ditu

**CSS Aldaketak:**
```css
body.high-contrast {
    background: #000;
}

body.high-contrast .container {
    background: #000;
    color: #fff;
}

body.high-contrast h1,
body.high-contrast h2 {
    color: #ffff00;
}

body.high-contrast a {
    color: #00ffff;
}
```

---

### 4️⃣ **Testu Tamaina Handitu / Texto Grande**

**Non:** Bidalketa1, Bidalketa2, Index

**Zer egin da:**
- "Testu handia" aukera menuan
- Aktibatzean, testu guztia %20 handitzen da
- Tituluen tamaina ere proportzionalki handitzen da

**Onurak:**
- ✅ Ikusmen arazoak dituztenentzat
- ✅ Zahartzaroko irakurketa zailtasunentzat
- ✅ Pantaila txikietatik irakurtzeko errazagoa

**CSS Aldaketak:**
```css
body.large-text {
    font-size: 120%;
}

body.large-text h1 {
    font-size: 2.8em;
}

body.large-text h2 {
    font-size: 2.2em;
}
```

---

### 5️⃣ **Tarte Zabalduak / Espaciado Amplio**

**Non:** Bidalketa1, Bidalketa2, Index

**Zer egin da:**
- "Tarteak zabaldu" aukera menuan
- Aktibatzean:
  - Letra-tartea: 0.05em
  - Hitz-tartea: 0.1em
  - Lerro-tartea: 2

**Onurak:**
- ✅ Dislexia duten ikasleentzako irakurketa errazagoa
- ✅ Hitzak bereizten lagundu
- ✅ Irakurketa-erritmoa hobetu

**CSS Aldaketak:**
```css
body.wide-spacing {
    letter-spacing: 0.05em;
    word-spacing: 0.1em;
}

body.wide-spacing .content {
    line-height: 2;
}
```

---

### 6️⃣ **LocalStorage Gorde / Guardar en LocalStorage**

**Non:** Bidalketa1, Bidalketa2, Index

**Zer egin da:**
- Erabiltzailearen irisgarritasun aukerak LocalStorage-n gordetzen dira
- Hurrengo bisitaldietan, aukerak automatikoki kargatzen dira
- Ez da behar berriz konfiguratzea

**JavaScript Kode-adibidea:**
```javascript
// Gorde aukera
localStorage.setItem('readingMode', true);

// Kargatu aukera
if (localStorage.getItem('readingMode') === 'true') {
    document.body.classList.add('reading-mode');
}
```

**Onurak:**
- ✅ Erabiltzaile-esperientzia hobetzen du
- ✅ Denborarik ez da galtzen berriz konfiguratzean
- ✅ Aukerak nabigatzailearen sesioan zehar mantentzen dira

---

## 📊 5.A Atazaren Lorpen Adierazlea

> **"Aurkeztutako eduki guztietan irisgarritasuna bermatzeko elementuren bat txertatu da"**

### ✅ Beteta:

1. ✅ **Bideoak azpitituluak dituzte** (YouTube CC)
2. ✅ **Irakurketa modua eskuragarri** dago
3. ✅ **Kontraste altuko modua** aktibatu daiteke
4. ✅ **Testu-tamaina handitu** daiteke
5. ✅ **Tarte zabalduak** aktibatu daitezke
6. ✅ **Aukerak LocalStorage-n gordetzen** dira

---

## 🛠️ Fitxategi Aldatuen Zerrenda

### HTML Fitxategiak:
- ✅ `index.html` - Botoia, menua, script eta bideo azpitituluak
- ✅ `bidalketa1.html` - Botoia, menua, script eta bideo azpitituluak
- ✅ `bidalketa2.html` - Botoia, menua, script, bideo azpitituluak eta 5.A sekzioa

### CSS Fitxategiak:
- ✅ `styles.css` - Irisgarritasun estiloak gehituta (180 lerro berri)

---

## 🎯 Nola Erabili Irisgarritasun Aukerak

### 1. Ireki Portfolio-a
Ireki edozein orri (index.html, bidalketa1.html edo bidalketa2.html)

### 2. Sakatu ♿ Botoia
Orriaren beheko eskuin aldean dagoen botoia sakatu

### 3. Hautatu Aukerak
Menuan agertu diren aukerak hautatu:
- 📖 Irakurketa modua
- 🎨 Kontraste altua
- 🔤 Testu handia
- 📏 Tarteak zabaldu

### 4. Itxi Menua
Edozein lekutan klik egin menua ixteko

### 5. Gozatu!
Zure aukerak automatikoki gordetzen dira hurrengo bisitaldietan

---

## 📹 Bideoen Azpitituluak Nola Aktibatu

1. ▶️ Sakatu bideoa erreproduzitzeko
2. Pantailan agertu den 'CC' botoia sakatu
3. Azpitituluak automatikoki kargatzen dira euskaraz

---

## 🌟 Etorkizuneko Hobekuntzak

- 🔊 Audio-deskripzioak bideoentzat
- 🎨 Kolore-eskema pertsonalizatuak
- 🗣️ Pantaila-irakurle hobekuntzak (ARIA etiketak)
- ⌨️ Teklatu nabigazio osoa
- 📱 Mugikor esperientzia hobetuta

---

## 📚 Erreferentziak

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [YouTube Accessibility](https://support.google.com/youtube/answer/2734796)
- [Web Accessibility Initiative](https://www.w3.org/WAI/)

---

**Egilea:** Mikel Aldalur  
**Data:** 2025-11-12  
**Ataza:** 5.A - Eduki Digitalen Irisgarritasuna  
**Egoera:** ✅ Osatuta
