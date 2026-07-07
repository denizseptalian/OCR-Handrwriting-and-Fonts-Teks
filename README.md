# Pembaca Patok Blok/TPH — Streamlit + CharCNN TFLite

Aplikasi Streamlit untuk membaca nomor Blok dan TPH dari foto patok:
- Deteksi garis pemisah horizontal (atas = Blok, bawah = TPH)
- Segmentasi karakter (threshold adaptif + merge fragmen)
- Klasifikasi per karakter dengan model CharCNN TFLite
- TPH dikunci angka saja (0-9)
- **Fitur suara**: hasil deteksi dibacakan otomatis (TTS Bahasa Indonesia)

## Struktur repo

```
├── app.py
├── requirements.txt
├── labels.txt
├── char_cnn_fp32.tflite   <-- taruh model kamu di sini (dari Google Drive)
└── README.md
```

## Deploy ke Streamlit Cloud

1. Download `char_cnn_fp32.tflite` dari Google Drive (`MyDrive/CharCNN_Android/`)
   dan taruh di root repo ini (sejajar dengan `app.py`).
2. Push semua file ke repo GitHub.
3. Buka https://share.streamlit.io -> New app -> pilih repo -> Main file: `app.py`.
4. **PENTING**: di Advanced settings, pilih **Python 3.11**
   (supaya `tflite-runtime` terpasang — jauh lebih ringan daripada tensorflow penuh).
5. Deploy.

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Catatan

- Fitur suara memakai gTTS (Google Text-to-Speech) — butuh koneksi internet.
  Di Streamlit Cloud otomatis tersedia.
- Audio autoplay butuh `streamlit >= 1.36`. Beberapa browser memblokir autoplay
  sampai pengguna berinteraksi sekali dengan halaman (klik apa saja).
- Kalau segmentasi kurang pas, sesuaikan parameter di sidebar
  (block size / konstanta C / tinggi minimum karakter).
