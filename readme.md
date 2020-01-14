# Head-shoulders segmentation

## Program sederhana untuk tes teknikal.
1. Haar cascade digunakan untuk mendapatkan head-shoulders pada suatu gambar. Karena penggunaan metode ini, terbatas hanya untuk wajah frontal saja dengan sedikit toleransi kemiringan. Digunakan implementasi Haar Cascade dari opencv.
2. Region yang tedeteksi oleh haar cascade di-crop, lalu gambar hasil crop tersebut dijadikan input kedalam model segmentasi.
3. Segmentasi dilakukan menggunakan pendekatan berbasis deep learning. Model yang digunakan adalah [BiSeNet](https://arxiv.org/abs/1808.00897) yang diimplementasi dalam framework Keras oleh pengguna github [Shaoanlu](https://github.com/shaoanlu/face_toolbox_keras).
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

## Requirements
(ada di requirements.txt)
- dlib version 19.8.1(tidak terlalu perlu jika hanya menguji yang segmentasi)
- matplotlib version 2.2.2
- numpy version 1.16.4
- opencv-contrib-python version 4.1.0.25
- tensorflow-gpu version 2.0.0
- tqdm version 4.23.4