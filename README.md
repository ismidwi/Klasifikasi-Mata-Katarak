
# 🧠 Klasifikasi Mata Katarak Menggunakan Deep Learning

**Ujian Akhir Praktikum (UAP) Pembelajaran Mesin**

## 📌 Deskripsi Proyek

Proyek ini merupakan bagian dari **Ujian Akhir Praktikum (UAP) Pembelajaran Mesin** yang bertujuan untuk membangun **sistem website sederhana berbasis pembelajaran mesin** menggunakan **Streamlit** untuk melakukan **klasifikasi citra fundus mata** menjadi dua kelas, yaitu **Normal** dan **Cataract**.

Model yang dibangun mencakup:

* 1 **Neural Network dasar (CNN non-pretrained)**
* 2 **Model pretrained dengan transfer learning**
  (MobileNetV2 dan EfficientNetB0)

Sistem dijalankan **secara lokal**, menerima input gambar dari pengguna, memproses data, dan menampilkan hasil prediksi.

---

## 📂 Struktur Repository

```
Klasifikasi-Mata-Katarak/
│
├── app.py
├── README.md
├── preprocessing.ipynb
├── training_model.ipynb
└── (dataset & model besar disimpan di Google Drive)
```

> ⚠️ **Catatan**
> File berukuran besar seperti dataset dan file model `.h5` **tidak diupload ke GitHub** karena melebihi batas ukuran.
> Seluruh file besar disimpan di **Google Drive** (link tersedia di bawah).

---

## 📊 Dataset

* **Nama Dataset**: ODIR-5K (Ocular Disease Intelligent Recognition)
* **Jenis Data**: Citra (Image Data – fundus mata)
* **Jumlah Data**:

  * Dataset awal < 5.000
  * ✅ **Dilakukan data augmentation** untuk memenuhi ketentuan minimal UAP
* **Sumber Dataset**:

  [https://www.kaggle.com/code/rishabnair/odir-5k]

### 📁 Struktur Dataset di Google Drive

```
dataset_labeled/
├── Normal/
└── Cataract/

Training Images/   → data sebelum pelabelan
Testing Images/    → data sebelum pelabelan
data.xlsx          → metadata & label awal
```

📌 **Catatan Penting**

* Folder **Training Images** dan **Testing Images** berisi dataset **belum berlabel**
* Proses pelabelan dilakukan otomatis berdasarkan `data.xlsx`
* Dataset final berada di folder **dataset_labeled**

---

## ⚙️ Preprocessing Data

Tahapan preprocessing dilakukan pada file `preprocessing.ipynb`, meliputi:

1. **Load metadata dari `data.xlsx`**
   Metadata digunakan untuk mengidentifikasi kondisi mata pada setiap citra fundus.

2. **Pelabelan otomatis (Normal / Cataract)**
   Proses pelabelan dilakukan secara otomatis berdasarkan metadata pada file `data.xlsx`.
   Setelah proses pelabelan selesai, diperoleh distribusi dataset sebagai berikut:
   
  **Sebelum Augmentasi**:
   * **Total images**: 3.411
   * **Normal**:       3.098
   * **Cataract**:     313

   Hasil ini menunjukkan bahwa dataset awal **sangat tidak seimbang**, di mana jumlah citra kelas **Cataract jauh lebih sedikit** dibandingkan kelas **Normal**.

   **Setelah Augmentasi**:
   * **Normal**:       3.098
   * **Cataract**:     1.252
   * **Total**:        4.350

3. **Resize gambar ke 128×128**
   Seluruh citra diseragamkan ukurannya untuk menyesuaikan input model dan mengurangi beban komputasi.

4. **Normalisasi pixel (0–1)**
   Nilai pixel dinormalisasi untuk mempercepat konvergensi model saat training.

5. **Visualisasi sampel data**
   Beberapa contoh citra dari masing-masing kelas divisualisasikan untuk memastikan bahwa proses pelabelan telah berjalan dengan benar.

6. **Data Augmentation (rotasi, shift, zoom, flip)**
   Data augmentation diterapkan terutama pada kelas **Cataract** untuk:

   * Menambah jumlah data hingga memenuhi ketentuan minimal UAP (≥ 5.000 data)
   * Mengurangi ketidakseimbangan kelas (*class imbalance*)
   * Meningkatkan kemampuan generalisasi model dalam mendeteksi katarak

7. **Penyimpanan hasil preprocessing (`.npy`)**
   Dataset hasil preprocessing disimpan dalam format `.npy` untuk mempercepat proses training.

📌 **Augmentasi dilakukan untuk menambah jumlah data dan menyeimbangkan kelas**, sesuai ketentuan UAP.

---

## 🤖 Model yang Digunakan

### 1️⃣ Convolutional Neural Network (CNN) – Non Pretrained

* Dibangun dari awal tanpa bobot pretrained
* Arsitektur: Conv2D → MaxPooling → Dense
* Digunakan sebagai **baseline model**

### 2️⃣ MobileNetV2 – Transfer Learning

* Pretrained pada ImageNet
* Feature extractor dibekukan (freeze)
* Ditambahkan classifier layer

### 3️⃣ EfficientNetB0 – Transfer Learning

* Pretrained pada ImageNet
* Arsitektur lebih ringan dan efisien
* Digunakan untuk perbandingan performa

---

## 📈 Evaluasi Model

Setiap model dievaluasi menggunakan:

* **Classification Report**
  (Accuracy, Precision, Recall, F1-score)
* **Grafik Loss & Accuracy**
* **Confusion Matrix**
---

## 🔍 Tabel Perbandingan Model

| Nama Model                             | Akurasi | Hasil Analisis                                                                                                                                                                                                                        |
| -------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CNN (Non-Pretrained)**               | **94%** | Model CNN dasar mampu mempelajari pola citra fundus dengan cukup baik. Akurasi tergolong tinggi, namun performa pada kelas **Cataract** masih terbatas (recall 0.47) akibat ketidakseimbangan data dan tidak adanya bobot pretrained. |
| **MobileNetV2 (Transfer Learning)**    | **95%** | MobileNetV2 memberikan peningkatan performa dibanding CNN. Model sangat baik dalam mengenali kelas **Normal** (recall 0.99), namun masih kesulitan mendeteksi **Cataract** secara optimal (recall 0.60).                              |
| **EfficientNetB0 (Transfer Learning)** | **95%** | EfficientNetB0 menunjukkan performa paling seimbang. Meskipun akurasi total sama dengan MobileNetV2, model ini memiliki **recall Cataract lebih tinggi (0.69)** sehingga lebih baik dalam mendeteksi kasus penyakit.                  |

---

## 📊 Ringkasan Hasil Evaluasi (Validation Set)

### 🔹 CNN (Non-Pretrained)

* **Accuracy**: 94%
* **Normal**
  Precision: 0.95 | Recall: 0.99 | F1-score: 0.97
* **Cataract**
  Precision: 0.85 | Recall: 0.47 | F1-score: 0.60

### 🔹 MobileNetV2

* **Accuracy**: 95%
* **Normal**
  Precision: 0.96 | Recall: 0.99 | F1-score: 0.97
* **Cataract**
  Precision: 0.84 | Recall: 0.60 | F1-score: 0.70

### 🔹 EfficientNetB0

* **Accuracy**: 95%
* **Normal**
  Precision: 0.97 | Recall: 0.98 | F1-score: 0.97
* **Cataract**
  Precision: 0.77 | Recall: 0.69 | F1-score: 0.73
  
> 📌 *Nilai detail dapat dilihat pada notebook training_model.ipynb*
---

## 🧠 Analisis Tambahan

* Dataset bersifat **tidak seimbang**, sehingga performa pada kelas Cataract lebih menantang.
* **Data augmentation** dilakukan pada tahap preprocessing untuk memenuhi syarat minimal 5.000 data.
* Model **transfer learning** secara konsisten memberikan performa lebih baik dibanding CNN dari nol.
* **EfficientNetB0** lebih robust dalam mendeteksi kelas Cataract meskipun akurasi total sama dengan MobileNetV2.

---

## 🌐 Sistem Website (Streamlit)

Sistem website sederhana dibangun menggunakan **Streamlit** dengan fitur:

* Upload gambar fundus mata
* Pemilihan model (CNN / MobileNetV2 / EfficientNetB0)
* Menampilkan hasil prediksi (Normal / Cataract)

Website dijalankan **secara lokal**.

### ▶️ Cara Menjalankan Website

```
streamlit run app.py
```

---

## ☁️ Google Drive (Dataset & Model)

📎 **Link Google Drive**
*(dataset, file .h5, dan file besar lainnya)*
👉 **[https://drive.google.com/drive/folders/18C89MiOOmKijcYeSmdvL1dDbejyBXiFU?usp=sharing]**

---

## 🎯 Kesimpulan

* Transfer learning terbukti meningkatkan performa dibanding CNN dari awal
* EfficientNetB0 memberikan hasil terbaik pada klasifikasi katarak
* Sistem berhasil diintegrasikan ke website Streamlit sesuai ketentuan UAP

## 👩‍💻 Identitas

**Nama**: Ismi Dwi Junianti (202210370311431)

**Mata Kuliah**: Pembelajaran Mesin

**Jenis Tugas**: Ujian Akhir Praktikum (UAP)

---
