# Head-shoulders segmentation

## Program sederhana untuk tes teknikal.
1. Haar cascade digunakan untuk mendapatkan head-shoulders pada suatu gambar. Karena penggunaan metode ini, terbatas hanya untuk wajah frontal saja dengan sedikit toleransi kemiringan. Digunakan implementasi Haar Cascade dari opencv.
2. Region yang tedeteksi oleh haar cascade di-crop, lalu gambar hasil crop tersebut dijadikan input kedalam model segmentasi.
3. Segmentasi dilakukan menggunakan pendekatan berbasis deep learning. Model yang digunakan adalah [BiSeNet](https://arxiv.org/abs/1808.00897) yang diimplementasi dalam framework Keras oleh pengguna github [Shaoanlu](https://github.com/shaoanlu/face_toolbox_keras). Segmentasi yang dilakukan adalah semantic segmentation, yang berarti semua kelas dalam 1 gambar tidak dibedakan seperti instance segmentation.
4. Hasil segmentasi kemudian ditampilkan.

**Usage**
```
python main.py [source file] --maxdim [MAXDIM]
```

- source file adalah path menuju file gambar sebagai input.
- maxdim adalah dimensi yang digunakan untuk resize. Dimensi tertinggi pada gambar diresize menjadi ukuran ini dengan aspect ratio terjaga.

## Memory + Timing measure

**Usage**
```
python timing.py
```

Pengukuran performa dilakukan pada Laptop DELL Inspiron 15 7565 dengan spesifikasi:
- CPU : Intel core i7-7700HQ 4 Physical Core @2.8Ghz dengan Hyperthreading
- GPU : NVidia GeForce 1050 Ti dengan VRAM 4GB
- RAM : 8GB

Pengukuran memory dilakukan menggunakan task manager.
- Untuk deteksi wajah, memerlukan space sebanyak sekitar 140 MB (di RAM)
- Untuk segmentasi, memerlukan space sekitar 1.5 GB (di Memory GPU)
- Keduanya pada gambar ukuran 640 x 480

Pengukuran waktu dilakukan secara programatis menggunakan fungsi timing dari OpenCV. Pengukuran dilakukan sebanyak 1001 kali (1 kali pertama untuk membiarkan parameter diload dahulu ke memory) dan diambil statistik mean, standar deviasi, dan range minimum hingga maximum.

**Pengukuran waktu deteksi wajah**
- First detection : 0.20438336581959277 seconds
- Avg : 0.0666391829989472 seconds
- Stdev : 0.010335157303854208 seconds
- Spanning 0.048444215105837136 to 0.20438336581959277 seconds

**pengukuran waktu segmentasi**
- First detection : 3.620750605812518 seconds
- Avg : 0.07437452360070528 seconds
- Stdev : 0.007538762860430644 seconds
- Spanning 0.06672890414031146 to 0.13456677795261807 seconds
  
## Performance Measure
Seberapa baik segmentasi berjalan? Kemarin diminta bisa mendapatkan metric seberapa baik hasil dari segmentasi. Untuk mendapatkan hasil tersebut, diperlukan sebuah test set external.

Secara naif bisa digunakan pixel-wise accuracy sebagai error. Akan tetapi pendekatan tersebut akan bermasalah ketika ada class imbalance di gambar. Misal rata-rata gambar didominasi oleh background. Dengan mengklasifikasi semua pixel sebagai background, akurasi tinggi bisa mudah dicapai. Pendekatan ini kurang baik karena sifatnya terlalu global.

Umumnya, ada dua pendekatan yang dipakai untuk mengukur performa dari semantic segmentation:
1. IoU / Jaccard Index

Pendekatan ini dilakukan untuk setiap kelas dan nantinya IoU / Jaccard Index dihitung untuk setiap kelas dan diambil reratanya. Dalam menghitung Jaccard Index, dihitung beberapa besaran terlebih dahulu, yakni True Positive (TP), False Positive(FP) dan False Negative(FN). Tiap-tiap pixel disini mengacu pada pixel yang menyimpan nilai 1 pada mask yang melingkupi objek.
- True Positive adalah tiap tiap pixel yang berada pada posisi sama di Label / Ground Truth (GT) dan pixel hasil prediksi. Dengan kata lain, interseksi antara Prediksi dan GT.
- False Positive adalah pixel yang ada pada prediksi, namun tidak memiliki interseksi dengan GT.
- False Negative adalah kebalikan dari FP, yakni pixel yang ada pada GT tetapi tidak muncul pada prediksi.

Setelah mendapatkan besaran-besaran tersebut, IoU atau Jaccard Index bisa dihitung dengan rumus

<img src="https://render.githubusercontent.com/render/math?math=IoU = \frac{TP}{TP + FP + FN}">

Sesuai namanya, Intersection Over Union, IoU juga bisa diformulasikan sebagai berikut :

<img src="https://render.githubusercontent.com/render/math?math=IoU = \frac{\mid Pred \cap GT\mid}{\mid Pred \cup GT\mid - \mid Pred \cap GT\mid}">

2. F1 Score / Dice Coefficient

Menggunakan definisi TP, FP, dan FN dari sebelumnya, F1 Score / Dice Coefficient bisa dihitung dengan menggunakan rumus

<img src="https://render.githubusercontent.com/render/math?math=IoU = \frac{2 \times TP}{2 \times TP %2b FP %2b FN}">

Atau bisa juga diformulasikan sebagai

<img src="https://render.githubusercontent.com/render/math?math=IoU = \frac{2 \times \mid Pred \cap GT \mid}{\mid Pred\mid %2b \mid GT\mid}">

**Perbedaan IoU dan F1 Score**

Kedua metric diatas berkorelasi. Akan tetapi, bila digunakan untuk mengukur average performance dari banyak test set sekaligus, akan terlihat perbedaannya.

Mirip seperti L1 dan L2 distance, IoU memberikan bobot lebih pada prediksi yang salah.

Umumnya, average performance yang diukur dengan F1 score akan memberikan expected performance, dan average performance yang diukur dengan IoU akan memberikan worst case performance.

## Requirements
(ada di requirements.txt)
- matplotlib version 2.2.2
- numpy version 1.16.4
- opencv-contrib-python version 4.1.0.25
- tensorflow-gpu version 2.0.0
- tqdm version 4.23.4