"""
==============================================================
  BANANA DATASET PREPARATION SCRIPT
  Untuk: merapikan dataset dari Label Studio -> siap Google Colab
==============================================================

CARA PAKAI:
  1. Setel path di bagian KONFIGURASI di bawah
  2. Pastikan DRY_RUN = True (preview dulu, tidak ada file diubah)
  3. Jalankan: python prepare_dataset_local.py
  4. Kalau hasilnya sudah OK, ganti DRY_RUN = False lalu jalankan lagi
"""

import os
import re
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Paksa output langsung tampil (fix Windows buffering)
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

print("=" * 60)
print("  BANANA DATASET PREPARATION - MULAI")
print("=" * 60)
sys.stdout.flush()

# ================================================================
# KONFIGURASI — SESUAIKAN PATH DI BAWAH INI
# ================================================================

IMAGES_DIR = r"D:\banana_dataset\images"   # folder images kamu
LABELS_DIR = r"D:\banana_dataset\labels"   # folder labels dari Label Studio
OUTPUT_DIR = r"D:\banana_dataset_colab"    # folder hasil (akan dibuat otomatis)

# True  = preview saja, TIDAK ada file yang diubah
# False = benar-benar proses dan salin file
DRY_RUN = True

VAL_RATIO   = 0.2   # 20% untuk validasi, 80% untuk train
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "mentah",
    1: "mengkal",
    2: "matang",
    3: "busuk",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP",
}

# ================================================================
print(f"\nKonfigurasi:")
print(f"  IMAGES_DIR : {IMAGES_DIR}")
print(f"  LABELS_DIR : {LABELS_DIR}")
print(f"  OUTPUT_DIR : {OUTPUT_DIR}")
print(f"  DRY_RUN    : {DRY_RUN}")
print(f"  VAL_RATIO  : {VAL_RATIO}")
sys.stdout.flush()

# ================================================================
# CEK FOLDER
# ================================================================
ok = True
for label, path in [("IMAGES_DIR", IMAGES_DIR), ("LABELS_DIR", LABELS_DIR)]:
    if os.path.isdir(path):
        print(f"\n[OK] {label} ditemukan: {path}")
    else:
        print(f"\n[ERROR] {label} TIDAK DITEMUKAN: {path}")
        print(f"  -> Pastikan path sudah benar di bagian KONFIGURASI!")
        ok = False
sys.stdout.flush()

if not ok:
    print("\nScript berhenti karena folder tidak ditemukan.")
    print("Setel IMAGES_DIR dan LABELS_DIR yang benar lalu coba lagi.")
    sys.exit(1)

# ================================================================
# FUNGSI
# ================================================================

def normalize_stem(stem):
    """'000001 (2)' -> '000001_2'"""
    s = stem.replace(" ", "_").replace("(", "").replace(")", "")
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def strip_uuid_prefix(stem):
    """
    Hapus UUID prefix Label Studio:
      'xxxxxxxx__xxxxxxxx-nama' -> 'nama'
      'xxxxxxxx_xxxxxxxx-nama'  -> 'nama'
    """
    # Pola double underscore (Label Studio terbaru)
    m = re.match(r'^[0-9a-f]+__[0-9a-f]+-(.+)$', stem, re.IGNORECASE)
    if m:
        return m.group(1)
    # Pola single underscore
    m = re.match(r'^[0-9a-f]{8}_[0-9a-f]{8}-(.+)$', stem, re.IGNORECASE)
    if m:
        return m.group(1)
    # Pola standard UUID (8-4-4-4-12)
    m = re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-(.+)$',
        stem, re.IGNORECASE
    )
    if m:
        return m.group(1)
    return stem


# ================================================================
# KUMPULKAN FILE
# ================================================================
print("\n" + "-" * 60)
print("Membaca folder images...")
sys.stdout.flush()

image_map = {}  # normalized_stem -> (path, ext)
for fname in os.listdir(IMAGES_DIR):
    ext = Path(fname).suffix
    if ext not in IMAGE_EXTENSIONS:
        continue
    stem = Path(fname).stem
    norm = normalize_stem(stem)
    path = os.path.join(IMAGES_DIR, fname)
    if norm not in image_map:
        image_map[norm] = (path, ext)

print(f"  -> {len(image_map)} gambar ditemukan")
print(f"  Contoh nama (5 pertama): {sorted(image_map)[:5]}")
sys.stdout.flush()

print("\nMembaca folder labels...")
sys.stdout.flush()

label_map = {}  # cleaned_stem -> path
for fname in os.listdir(LABELS_DIR):
    if not fname.lower().endswith(".txt"):
        continue
    stem = Path(fname).stem
    cleaned = strip_uuid_prefix(stem)
    path = os.path.join(LABELS_DIR, fname)
    if cleaned not in label_map:
        label_map[cleaned] = path

print(f"  -> {len(label_map)} label ditemukan")
print(f"  Contoh nama asli label (5 pertama, setelah strip UUID): {sorted(label_map)[:5]}")
sys.stdout.flush()

# ================================================================
# COCOKKAN
# ================================================================
print("\n" + "-" * 60)
print("Mencocokkan gambar dengan label...")
sys.stdout.flush()

matched_stems = sorted(set(image_map) & set(label_map))
img_only      = sorted(set(image_map) - set(label_map))
lbl_only      = sorted(set(label_map) - set(image_map))

print(f"\n  Pasangan COCOK (image + label) : {len(matched_stems)}")
print(f"  Gambar TANPA label             : {len(img_only)}")
print(f"  Label TANPA gambar             : {len(lbl_only)}")
sys.stdout.flush()

if len(matched_stems) == 0:
    print("\n[ERROR] Tidak ada pasangan yang cocok!")
    print()
    print("  Kemungkinan penyebab:")
    print("  Gambar di-rename manual (000001, 000002..) tapi label masih")
    print("  pakai nama asli dari Label Studio (green_banana, IMG_2026...).")
    print()
    print("  SOLUSI:")
    print("  Gunakan folder 'images' langsung dari export Label Studio YOLO,")
    print("  JANGAN rename gambar sebelum jalankan script ini.")
    print()
    print("  Contoh nama gambar yang ada:")
    for s in sorted(image_map)[:8]:
        print(f"    {s}")
    print("  Contoh nama label yang ada (setelah strip UUID):")
    for s in sorted(label_map)[:8]:
        print(f"    {s}")
    sys.exit(1)

if len(img_only) > 0:
    print(f"\n  [INFO] {len(img_only)} gambar tidak punya label (akan dilewati):")
    for s in img_only[:5]:
        print(f"    {s}")
    if len(img_only) > 5:
        print(f"    ... dan {len(img_only)-5} lainnya")

if len(lbl_only) > 0:
    print(f"\n  [INFO] {len(lbl_only)} label tidak punya gambar (akan dilewati):")
    for s in lbl_only[:5]:
        print(f"    {s}")
    if len(lbl_only) > 5:
        print(f"    ... dan {len(lbl_only)-5} lainnya")

# ================================================================
# DISTRIBUSI KELAS
# ================================================================
print("\n" + "-" * 60)
print("Menghitung distribusi kelas...")
sys.stdout.flush()

class_counts = defaultdict(int)
for stem in matched_stems:
    try:
        with open(label_map[stem]) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1
    except Exception:
        pass

total_bbox = sum(class_counts.values())
print(f"\n  Total bbox : {total_bbox}")
for cls_id, name in CLASS_NAMES.items():
    count = class_counts.get(cls_id, 0)
    pct   = count / total_bbox * 100 if total_bbox else 0
    bar   = "=" * int(pct / 2)
    print(f"  {cls_id} {name:8s}: {count:5d} bbox ({pct:5.1f}%) [{bar}]")
sys.stdout.flush()

# ================================================================
# SPLIT TRAIN / VAL
# ================================================================
random.seed(RANDOM_SEED)
stems_shuffled = list(matched_stems)
random.shuffle(stems_shuffled)
n_val       = max(1, int(len(stems_shuffled) * VAL_RATIO))
val_stems   = set(stems_shuffled[:n_val])
train_stems = set(stems_shuffled[n_val:])

print(f"\n  Split train/val:")
print(f"    Train : {len(train_stems)} gambar ({100-int(VAL_RATIO*100)}%)")
print(f"    Val   : {len(val_stems)} gambar ({int(VAL_RATIO*100)}%)")
sys.stdout.flush()

# ================================================================
# DRY RUN ATAU EKSEKUSI
# ================================================================
if DRY_RUN:
    print("\n" + "=" * 60)
    print("  DRY RUN SELESAI - tidak ada file yang diubah")
    print()
    print("  Kalau hasilnya sudah OK:")
    print("  1. Buka script ini")
    print("  2. Ganti  DRY_RUN = True  menjadi  DRY_RUN = False")
    print("  3. Jalankan lagi: python prepare_dataset_local.py")
    print("=" * 60)
    sys.exit(0)

# ================================================================
# BUAT FOLDER OUTPUT
# ================================================================
print("\n" + "-" * 60)
print(f"Membuat folder output: {OUTPUT_DIR}")
sys.stdout.flush()

for split in ("train", "val"):
    for sub in ("images", "labels"):
        Path(os.path.join(OUTPUT_DIR, split, sub)).mkdir(parents=True, exist_ok=True)

# ================================================================
# SALIN FILE
# ================================================================
print("Menyalin file...")
sys.stdout.flush()

errors = 0

def copy_pair(stem, split):
    global errors
    img_src, ext = image_map[stem]
    lbl_src = label_map[stem]
    img_dst = os.path.join(OUTPUT_DIR, split, "images", stem + ext)
    lbl_dst = os.path.join(OUTPUT_DIR, split, "labels", stem + ".txt")
    try:
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)
    except Exception as e:
        print(f"  [ERROR] {stem}: {e}")
        errors += 1

for i, stem in enumerate(sorted(train_stems), 1):
    copy_pair(stem, "train")
    if i % 100 == 0 or i == len(train_stems):
        print(f"  Train: {i}/{len(train_stems)} selesai")
        sys.stdout.flush()

for i, stem in enumerate(sorted(val_stems), 1):
    copy_pair(stem, "val")
    if i % 50 == 0 or i == len(val_stems):
        print(f"  Val  : {i}/{len(val_stems)} selesai")
        sys.stdout.flush()

# ================================================================
# BUAT dataset.yaml
# ================================================================
yaml_content = (
    "# Banana Detection Dataset - siap Google Colab\n"
    "# PENTING: Sesuaikan path train & val saat di Google Colab!\n"
    "#          Contoh: /content/drive/MyDrive/banana_dataset_colab/train/images\n\n"
    f"train: {os.path.join(OUTPUT_DIR, 'train', 'images')}\n"
    f"val:   {os.path.join(OUTPUT_DIR, 'val', 'images')}\n\n"
    f"nc: {len(CLASS_NAMES)}\n\n"
    "names:\n"
    "  0: mentah\n"
    "  1: mengkal\n"
    "  2: matang\n"
    "  3: busuk\n"
)
yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"\n  dataset.yaml dibuat: {yaml_path}")

# ================================================================
# BUAT classes.txt
# ================================================================
classes_path = os.path.join(OUTPUT_DIR, "classes.txt")
with open(classes_path, "w", encoding="utf-8") as f:
    for cls_id in sorted(CLASS_NAMES):
        f.write(CLASS_NAMES[cls_id] + "\n")
print(f"  classes.txt  dibuat: {classes_path}")

# ================================================================
# BUAT report.txt
# ================================================================
def dist_for(stems_set):
    d = defaultdict(int)
    for s in stems_set:
        try:
            with open(label_map[s]) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        d[int(parts[0])] += 1
        except Exception:
            pass
    return d

train_dist = dist_for(train_stems)
val_dist   = dist_for(val_stems)

report_lines = [
    "=" * 55,
    "  BANANA DATASET REPORT",
    "=" * 55,
    f"\nTotal gambar berlabel : {len(matched_stems)}",
    f"Train                 : {len(train_stems)}",
    f"Val                   : {len(val_stems)}",
    f"Gambar tanpa label    : {len(img_only)}",
    f"Label tanpa gambar    : {len(lbl_only)}",
    f"\nDistribusi kelas (TRAIN):",
]
for cls_id, name in CLASS_NAMES.items():
    report_lines.append(f"  {cls_id} {name:8s}: {train_dist.get(cls_id,0):5d} bbox")
report_lines.append(f"\nDistribusi kelas (VAL):")
for cls_id, name in CLASS_NAMES.items():
    report_lines.append(f"  {cls_id} {name:8s}: {val_dist.get(cls_id,0):5d} bbox")

report_path = os.path.join(OUTPUT_DIR, "report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  report.txt   dibuat: {report_path}")
sys.stdout.flush()

# ================================================================
# RINGKASAN AKHIR
# ================================================================
print("\n" + "=" * 60)
if errors == 0:
    print("  SELESAI TANPA ERROR!")
else:
    print(f"  Selesai dengan {errors} error (cek log di atas)")
print()
print(f"  Output folder  : {OUTPUT_DIR}")
print(f"  Train          : {len(train_stems)} gambar + label")
print(f"  Val            : {len(val_stems)} gambar + label")
print()
print("  LANGKAH SELANJUTNYA:")
print("  1. Buka report.txt untuk cek ringkasan")
print(f"  2. Upload folder ini ke Google Drive:")
print(f"       {OUTPUT_DIR}")
print("  3. Buka banana_training_colab.ipynb di Google Colab")
print("  4. Sesuaikan DATASET_PATH lalu Run All!")
print("=" * 60)
sys.stdout.flush()
