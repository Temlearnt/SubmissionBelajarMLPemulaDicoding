# Customer Transaction Clustering & Classification

## Project Overview

Project ini bertujuan untuk melakukan analisis pola transaksi pelanggan menggunakan metode **Machine Learning** melalui dua tahap utama:

1. **Clustering** untuk mengelompokkan pelanggan berdasarkan karakteristik transaksi.
2. **Classification** untuk memprediksi cluster pelanggan menggunakan model klasifikasi.

Proyek ini dibuat sebagai bagian dari implementasi **Machine Learning Pipeline**, mulai dari **Exploratory Data Analysis (EDA), Data Preprocessing, Clustering, Interpretasi Model, hingga Classification**.

---

# Dataset

Dataset berisi informasi aktivitas transaksi pelanggan dengan beberapa fitur seperti:

* TransactionAmount
* CustomerAge
* TransactionDuration
* LoginAttempts
* AccountBalance
* MerchantCategory
* TransactionType
* dan fitur lainnya

Dataset diproses dengan melakukan pembersihan data, encoding fitur kategorikal, serta scaling fitur numerik sebelum digunakan dalam model machine learning.

---

# Tahapan Proyek

## 1. Exploratory Data Analysis (EDA)

Tahapan awal untuk memahami karakteristik data dengan beberapa langkah berikut:

* Menampilkan data menggunakan `head()`
* Menampilkan informasi dataset menggunakan `info()`
* Statistik deskriptif menggunakan `describe()`
* Visualisasi distribusi data menggunakan histogram
* Analisis hubungan antar fitur menggunakan **Correlation Matrix**

Tujuan tahap ini adalah untuk memahami distribusi data serta hubungan antar variabel.

---

## 2. Data Cleaning & Preprocessing

Tahap ini dilakukan untuk memastikan data siap digunakan oleh model machine learning.

Langkah-langkah yang dilakukan:

* Mengecek **missing value** menggunakan `isnull().sum()`
* Mengecek **data duplikat** menggunakan `duplicated().sum()`
* Menghapus data kosong menggunakan `dropna()`
* Menghapus data duplikat menggunakan `drop_duplicates()`
* Menghapus kolom yang tidak relevan seperti:

  * TransactionID
  * AccountID
  * DeviceID
  * IPAddress
  * MerchantID
  * TransactionDate

### Feature Engineering

Beberapa teknik preprocessing yang digunakan:

* **Label Encoding** untuk fitur kategorikal
* **Outlier Handling**
* **Feature Scaling** menggunakan `StandardScaler`
* **Binning** pada fitur numerik untuk membuat kategori nilai

---

# 3. Clustering Model

Model clustering digunakan untuk mengelompokkan pelanggan berdasarkan pola transaksi.

### Algoritma yang digunakan

* **K-Means Clustering**

### Tahapan model

1. Menentukan jumlah cluster menggunakan **Elbow Method**
2. Melatih model menggunakan `KMeans`
3. Evaluasi cluster menggunakan **Silhouette Score**
4. Visualisasi hasil clustering
5. Eksperimen tambahan menggunakan **PCA**

Model disimpan menggunakan:

```python
joblib.dump(model_clustering, "model_clustering.h5")
```

---

# 4. Interpretasi Hasil Clustering

Setelah model terbentuk, dilakukan interpretasi terhadap karakteristik setiap cluster.

Analisis dilakukan dengan menghitung statistik agregasi seperti:

* Mean
* Min
* Max

Interpretasi cluster dilakukan untuk memahami karakteristik masing-masing kelompok pelanggan.

Contoh interpretasi cluster:

**Cluster 0 – Customer Stabil**

* Rata-rata transaksi stabil
* Usia pelanggan relatif lebih tinggi
* Saldo akun cenderung lebih tinggi
* Pola transaksi lebih konsisten

**Cluster 1 – Customer Aktif**

* Nilai transaksi sedikit lebih tinggi
* Usia pelanggan lebih muda
* Aktivitas transaksi lebih dinamis

---

# 5. Inverse Transform

Untuk interpretasi bisnis yang lebih mudah dipahami, dilakukan:

* `inverse_transform()` pada data yang sudah discale
* Mengembalikan data ke nilai aslinya

Dataset hasil clustering kemudian disimpan sebagai:

```
data_clustering_inverse.csv
```

Dataset ini berisi:

* Data asli
* Hasil cluster pada kolom **Target**

---

# 6. Classification Model

Tahap selanjutnya adalah membangun model klasifikasi untuk memprediksi cluster pelanggan.

### Target Variable

```
Target
```

### Dataset Split

Dataset dibagi menggunakan:

```
train_test_split()
```

### Algoritma yang digunakan

* Decision Tree
* Model tambahan untuk eksplorasi klasifikasi

### Evaluasi Model

Model dievaluasi menggunakan metrik:

* Accuracy
* Precision
* Recall
* F1-Score

Model disimpan menggunakan:

```
joblib.dump(decision_tree_model, "decision_tree_model.h5")
```

---

# Conclusion

Melalui proses clustering, pelanggan dapat dikelompokkan berdasarkan pola transaksi mereka. Model klasifikasi kemudian dapat digunakan untuk memprediksi cluster pelanggan baru sehingga membantu dalam:

* memahami perilaku pelanggan
* membuat strategi pemasaran
* meningkatkan layanan berbasis segmentasi pelanggan

---

# Author

Project ini dibuat untuk memenuhi tugas akhir kelas Belajar Machine Learning Pemula Dicoding.
