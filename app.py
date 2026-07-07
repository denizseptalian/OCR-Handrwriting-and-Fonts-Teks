# ============================================================
# Pembaca Patok Blok/TPH — CharCNN TFLite + Streamlit
# Fitur: deteksi garis pemisah, baca Blok (atas) & TPH (bawah),
#        output suara (TTS Bahasa Indonesia) saat teks terdeteksi
# ============================================================
import io
import os

import cv2
import numpy as np
import streamlit as st
from gtts import gTTS
from PIL import Image

# --- TFLite interpreter: LiteRT (pengganti resmi tflite-runtime) ---
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

# ============================================================
# Konfigurasi
# ============================================================
MODEL_PATH = "char_cnn_fp32.tflite"
LABELS_PATH = "labels.txt"
IMG_SIZE = 32

st.set_page_config(page_title="Pembaca Patok Blok/TPH", page_icon="🌴", layout="wide")


@st.cache_resource
def load_model():
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    return interp, inp, out, classes


interp, inp, out, CLASSES = load_model()


# ============================================================
# Pipeline deteksi — identik dengan notebook (Section 9)
# ============================================================
def x_overlap_ratio(a, b):
    left = max(a[0], b[0])
    right = min(a[0] + a[2], b[0] + b[2])
    if right <= left:
        return 0.0
    return (right - left) / min(a[2], b[2])


def y_gap(a, b):
    return max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))


def to_model_input(char_bin):
    inv = 255 - char_bin
    hh, ww = inv.shape
    side = int(max(hh, ww) * 1.3)
    canvas = np.full((side, side), 255, np.uint8)
    y0 = (side - hh) // 2
    x0 = (side - ww) // 2
    canvas[y0:y0 + hh, x0:x0 + ww] = inv
    small = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)[None, :, :, None] / 255.0


def predict_char(xin, digits_only=False):
    interp.set_tensor(inp["index"], xin)
    interp.invoke()
    prob = interp.get_tensor(out["index"])[0].copy()
    if digits_only:
        prob[10:] = 0
    k = int(prob.argmax())
    return CLASSES[k], float(prob[k])


def baca_patok(bgr, block_size=41, c_thresh=15, min_h_ratio=0.05):
    """Deteksi garis pemisah + baca Blok (atas) & TPH (bawah).

    Return: dict hasil + gambar visualisasi (RGB).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) Threshold: tulisan gelap di background terang
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block_size, C=c_thresh)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

    H_img, W_img = binary.shape

    # 2) Deteksi garis pemisah: kontur sangat lebar & pipih
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    separator_y = None
    sep_box = None
    char_contours = []
    for c in contours:
        x, y, w, hh = cv2.boundingRect(c)
        if w * hh < 0.001 * H_img * W_img:
            continue
        if w > 0.5 * W_img and w / max(hh, 1) > 4:
            separator_y = y + hh // 2
            sep_box = (x, y, w, hh)
            continue
        char_contours.append((x, y, w, hh))

    sep_found = separator_y is not None
    if not sep_found:
        separator_y = H_img // 2

    # 3) Filter + merge fragmen (tidak lintas garis pemisah)
    boxes = [list(b) for b in char_contours if b[3] > min_h_ratio * H_img]

    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                a_atas = (a[1] + a[3] / 2) < separator_y
                b_atas = (b[1] + b[3] / 2) < separator_y
                if a_atas != b_atas:
                    continue
                if x_overlap_ratio(a, b) > 0.4 and y_gap(a, b) < 0.4 * max(a[3], b[3]):
                    x0 = min(a[0], b[0])
                    y0 = min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break

    # 4) Bagi ke Blok (atas) & TPH (bawah), urut kiri->kanan
    blok_boxes = sorted([b for b in boxes if (b[1] + b[3] / 2) < separator_y], key=lambda b: b[0])
    tph_boxes = sorted([b for b in boxes if (b[1] + b[3] / 2) >= separator_y], key=lambda b: b[0])

    # 5) Prediksi
    blok_preds = [predict_char(to_model_input(binary[y:y + hh, x:x + w]))
                  for (x, y, w, hh) in blok_boxes]
    tph_preds = [predict_char(to_model_input(binary[y:y + hh, x:x + w]), digits_only=True)
                 for (x, y, w, hh) in tph_boxes]

    nomor_blok = "".join(c for c, _ in blok_preds)
    nomor_tph = "".join(c for c, _ in tph_preds)

    # 6) Visualisasi
    vis = rgb.copy()
    if sep_box is not None:
        sx, sy, sw, sh = sep_box
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 3)
    cv2.line(vis, (0, separator_y), (W_img, separator_y), (0, 0, 255), 2)
    for (x, y, w, hh), (ch, cf) in zip(blok_boxes, blok_preds):
        cv2.rectangle(vis, (x, y), (x + w, y + hh), (0, 200, 0), 3)
        cv2.putText(vis, ch, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 4)
    for (x, y, w, hh), (ch, cf) in zip(tph_boxes, tph_preds):
        cv2.rectangle(vis, (x, y), (x + w, y + hh), (255, 0, 0), 3)
        cv2.putText(vis, ch, (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)

    return {
        "nomor_blok": nomor_blok,
        "nomor_tph": nomor_tph,
        "blok_preds": blok_preds,
        "tph_preds": tph_preds,
        "sep_found": sep_found,
        "vis": vis,
        "binary": binary,
    }


# ============================================================
# TTS — suara Bahasa Indonesia
# ============================================================
def eja(teks):
    """Eja per karakter supaya jelas didengar: 'P67' -> 'P, 6, 7'."""
    return ", ".join(teks)


@st.cache_data(show_spinner=False)
def buat_audio(kalimat: str) -> bytes:
    """Generate MP3 dari teks (di-cache supaya tidak request ulang)."""
    tts = gTTS(text=kalimat, lang="id", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


# ============================================================
# UI
# ============================================================
st.title("🌴 Pembaca Patok Blok / TPH")
st.caption("CharCNN TFLite — atas garis = Nomor Blok, bawah garis = Nomor TPH. "
           "Hasil deteksi otomatis dibacakan lewat suara.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    sumber = st.radio("Sumber gambar", ["Upload file", "Kamera"])
    st.divider()
    st.subheader("Parameter deteksi")
    block_size = st.slider("Block size threshold (ganjil)", 21, 81, 41, step=2)
    c_thresh = st.slider("Konstanta C threshold", 5, 35, 15)
    min_h = st.slider("Tinggi minimum karakter (% gambar)", 2, 15, 5) / 100.0
    st.divider()
    suara_aktif = st.toggle("🔊 Bacakan hasil lewat suara", value=True)
    tampil_biner = st.toggle("Tampilkan gambar biner (debug)", value=False)

# Input gambar
img_file = None
if sumber == "Upload file":
    img_file = st.file_uploader("Upload foto patok", type=["png", "jpg", "jpeg", "bmp"])
else:
    img_file = st.camera_input("Ambil foto patok")

if img_file is None:
    st.info("⬆️ Upload atau ambil foto patok untuk memulai.")
    st.stop()

# Decode gambar
pil_img = Image.open(img_file).convert("RGB")
bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Proses
with st.spinner("Memproses gambar..."):
    hasil = baca_patok(bgr, block_size=block_size, c_thresh=c_thresh, min_h_ratio=min_h)

# --- Tampilkan hasil ---
col1, col2 = st.columns([3, 2])

with col1:
    st.image(hasil["vis"], caption="Hasil deteksi", use_container_width=True)
    if tampil_biner:
        st.image(hasil["binary"], caption="Gambar biner (debug)",
                 use_container_width=True, clamp=True)

with col2:
    if not hasil["sep_found"]:
        st.warning("Garis pemisah tidak terdeteksi — memakai tengah gambar sebagai batas.")

    terdeteksi = bool(hasil["nomor_blok"] or hasil["nomor_tph"])

    if terdeteksi:
        st.metric("Nomor Blok", hasil["nomor_blok"] if hasil["nomor_blok"] else "—")
        st.metric("Nomor TPH", hasil["nomor_tph"] if hasil["nomor_tph"] else "—")

        with st.expander("Detail confidence per karakter"):
            for ch, cf in hasil["blok_preds"]:
                st.write(f"Blok — **{ch}** : {cf*100:.0f}%")
            for ch, cf in hasil["tph_preds"]:
                st.write(f"TPH — **{ch}** : {cf*100:.0f}%")

        # --- FITUR SUARA: bacakan hasil saat teks terdeteksi ---
        if suara_aktif:
            bagian = []
            if hasil["nomor_blok"]:
                bagian.append(f"Nomor Blok, {eja(hasil['nomor_blok'])}")
            if hasil["nomor_tph"]:
                bagian.append(f"Nomor T P H, {eja(hasil['nomor_tph'])}")
            kalimat = "Terdeteksi. " + ". ".join(bagian)

            try:
                audio_bytes = buat_audio(kalimat)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                st.caption(f"🔊 \"{kalimat}\"")
            except Exception as e:
                st.warning(f"Suara gagal dibuat (butuh koneksi internet): {e}")
    else:
        st.error("Tidak ada karakter terdeteksi. Coba sesuaikan parameter di sidebar.")
