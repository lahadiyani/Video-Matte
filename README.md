# Video Matte - Video Background Remover

## 📌 Apa Itu

`Video Matte` adalah program Python untuk **menghapus latar belakang video secara otomatis** dan menggantinya dengan warna tertentu (default hitam). Program ini menggunakan model **U^2-Net** (melalui library `rembg`) untuk segmentasi objek utama dari video, sehingga dapat bekerja pada video dengan latar belakang **apa saja** tanpa harus menggunakan green screen.

Alasan dibuat: Sulit menemukan tool yang mampu menghapus background video secara fleksibel untuk berbagai jenis video tanpa chroma key. Program ini memberikan solusi yang **otomatis, mudah digunakan, dan mendukung GPU untuk percepatan**.

## ⚡ Fitur Utama

* Menghapus background video apapun, tidak terbatas pada warna tertentu.
* Mengganti background dengan warna bebas (default hitam).
* Mendukung GPU untuk mempercepat pemrosesan.
* Menampilkan progress bar real-time dengan estimasi waktu tersisa.
* Post-processing smoothing untuk hasil mask yang lebih halus.
* Resize frame untuk mempercepat pemrosesan video panjang.

## 🛠️ Cara Menggunakan

Pastikan Python sudah terinstal dan library berikut tersedia: `opencv-python`, `numpy`, `rembg`, `tqdm`, `Pillow`.

### Contoh Penggunaan:

```bash
# Basic usage dengan CPU
python run.py -i input.mp4 -o output.mp4

# Dengan GPU (jika tersedia)
python run.py -i input.mp4 -o output.mp4 --gpu

# Dengan background merah dan GPU
python run.py -i input.mp4 -o output.mp4 --background "255,0,0" --gpu

# Dengan resize untuk percepatan dan GPU
python run.py -i input.mp4 -o output.mp4 --resize 0.5 --gpu

# Tanpa post-processing smoothing dengan GPU
python run.py -i input.mp4 -o output.mp4 --no-post-process --gpu
```

### Opsi Argumen:

* `-i, --input` : Path ke video input
* `-o, --output` : Path untuk video output
* `-b, --background` : Warna background baru (format R,G,B, default 0,0,0)
* `-r, --resize` : Faktor resize untuk percepatan (0.1-1.0, default 1.0)
* `--no-post-process` : Nonaktifkan smoothing mask
* `--gpu` : Gunakan GPU jika tersedia
* `--skip-model-check` : Lewati pengecekan/download model

## 🧩 Alur Program

1. **Cek model U^2-Net** di lokal, jika tidak ada akan didownload.
2. **Setup GPU/CPU** sesuai opsi.
3. **Buka video input** dan baca properti video.
4. **Proses frame per frame**:

   * Segmentasi objek utama
   * Buat mask alpha
   * Terapkan mask ke frame
   * Gabungkan dengan background warna yang diinginkan
5. **Tulis frame ke video output**
6. **Tampilkan summary** setelah selesai

## 📋 Catatan

* Model hanya perlu didownload **sekali**, tersimpan di cache lokal.
* Untuk video panjang, disarankan menggunakan GPU dan/atau resize frame.
* Format warna harus R,G,B, contoh: `255,0,0` untuk merah.
* Estimasi waktu pemrosesan ditampilkan secara realtime di progress bar.

## 🎯 Tujuan Pembuatan

* Mempermudah **pembuatan video dengan latar belakang hitam atau warna lain** tanpa perlu green screen.
* Mengurangi waktu dan usaha dibandingkan editing manual frame-by-frame.
* Memberikan solusi fleksibel untuk **video dengan latar belakang kompleks**.

---

### 🔗 Referensi

* [rembg](https://github.com/danielgatis/rembg) untuk model U^2-Net
* [OpenCV](https://opencv.org/) untuk manipulasi video
* [NumPy](https://numpy.org/) untuk operasi array
* [Pillow](https://python-pillow.org/) untuk konversi gambar
* [tqdm](https://github.com/tqdm/tqdm) untuk progress bar

### 📚 Referensi Jurnal

* [Zhang, X., et al., "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection," arXiv:2005.09007, 2020](https://arxiv.org/abs/2005.09007)
* [Chen, L.-C., et al., "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs," IEEE TPAMI, 2018](https://arxiv.org/abs/1606.00915)
* [Hu, J., et al., "Rethinking Image Matting: A Review and Benchmark," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021](https://ieeexplore.ieee.org/document/9216017)
* [Xu, N., et al., "Deep Image Matting," IEEE CVPR, 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Xu_Deep_Image_Matting_CVPR_2017_paper.html)
