"""
Script untuk menyinkronkan nama file images dan labels dari export Label Studio (YOLO format).

Masalah:
  - Label Studio menambahkan UUID prefix (double underscore) pada file label:
      74eaf18c__ef0a5cb9-000001_2.txt
  - Image masih nama asli dengan spasi/kurung: 000001 (2).jpg
    ATAU image sudah pakai nama asli (tidak di-rename manual)

Solusi:
  - Strip UUID prefix dari nama label
  - Normalisasi nama image (spasi & kurung -> underscore)
  - Verifikasi berapa pasang yang cocok

CATATAN PENTING:
  Pastikan folder images yang kamu pakai adalah gambar yang di-export dari
  Label Studio (bukan yang di-rename manual ke 000001, 000002, dst).
  Kalau gambar sudah di-rename manual, nama image dan label tidak akan bisa
  dicocokkan secara otomatis.

Cara pakai:
  1. Setel IMAGES_DIR dan LABELS_DIR di bawah
  2. Jalankan: python fix_dataset_names.py
     -> DRY RUN (preview saja, tidak mengubah file)
  3. Jika hasilnya sudah sesuai, ubah DRY_RUN = False lalu jalankan lagi
"""

import os
import re
import shutil
from pathlib import Path

# ============================================================
# KONFIGURASI - sesuaikan path di bawah ini
# ============================================================
IMAGES_DIR = r"D:\banana_dataset\images"
LABELS_DIR = r"D:\banana_dataset\labels"
OUTPUT_DIR = r"D:\banana_dataset_fixed"   # folder output bersih (boleh sama dg input)

DRY_RUN = True   # True = preview saja | False = benar-benar rename file
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_stem(stem: str) -> str:
    """
    Normalisasi nama file (tanpa ekstensi):
      '000001 (2)' -> '000001_2'
      'IMG 20260430 (3)' -> 'IMG_20260430_3'
    """
    stem = stem.replace(" ", "_")
    stem = stem.replace("(", "").replace(")", "")
    stem = re.sub(r"_+", "_", stem)
    return stem.strip("_")


def strip_uuid_prefix(stem: str) -> str:
    """
    Hapus UUID prefix dari nama label Label Studio.

    Pola 1 (double underscore): xxxxxxxx__xxxxxxxx-nama_asli
    Pola 2 (single underscore): xxxxxxxx_xxxxxxxx-nama_asli
    Pola 3 (standard UUID):     xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-nama_asli

    Contoh:
      '74eaf18c__ef0a5cb9-000001_2'     -> '000001_2'
      '002da5db__c819c789-green_banana'  -> 'green_banana'
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

    return stem  # tidak ada prefix, kembalikan apa adanya


def collect_images(images_dir):
    """Kumpulkan image: normalized_stem -> (original_path, ext)"""
    result = {}
    for fname in os.listdir(images_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        norm = normalize_stem(stem)
        path = os.path.join(images_dir, fname)
        if norm in result:
            print(f"  [DUPLIKAT IMAGE] '{norm}': '{result[norm][0]}' vs '{path}'")
        result[norm] = (path, ext)
    return result


def collect_labels(labels_dir):
    """Kumpulkan label: cleaned_stem -> original_path"""
    result = {}
    for fname in os.listdir(labels_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".txt":
            continue
        cleaned = strip_uuid_prefix(stem)
        path = os.path.join(labels_dir, fname)
        if cleaned in result:
            print(f"  [DUPLIKAT LABEL] '{cleaned}': '{result[cleaned]}' vs '{path}'")
        result[cleaned] = path
    return result


def split_train_val(matched_stems, val_ratio=0.2, seed=42):
    """Split matched stems into train/val sets."""
    import random
    random.seed(seed)
    stems = sorted(matched_stems)
    random.shuffle(stems)
    split_idx = max(1, int(len(stems) * val_ratio))
    val = set(stems[:split_idx])
    train = set(stems[split_idx:])
    return train, val


def main():
    print("=" * 70)
    print("  Fix Dataset Names - Label Studio YOLO Export")
    print(f"  Mode: {'DRY RUN (tidak ada perubahan)' if DRY_RUN else 'AKTIF - file akan direname/disalin'}")
    print("=" * 70)

    for d in [IMAGES_DIR, LABELS_DIR]:
        if not os.path.isdir(d):
            print(f"\n[ERROR] Folder tidak ditemukan: {d}")
            return

    print(f"\nImages : {IMAGES_DIR}")
    print(f"Labels : {LABELS_DIR}")
    print(f"Output : {OUTPUT_DIR}\n")

    image_map = collect_images(IMAGES_DIR)
    label_map = collect_labels(LABELS_DIR)

    print(f"Total gambar ditemukan : {len(image_map)}")
    print(f"Total label ditemukan  : {len(label_map)}\n")

    matched    = set(image_map) & set(label_map)
    img_only   = set(image_map) - set(label_map)
    lbl_only   = set(label_map) - set(image_map)

    print(f"Pasangan COCOK (image+label) : {len(matched)}")
    print(f"Image tanpa label             : {len(img_only)}")
    print(f"Label tanpa image             : {len(lbl_only)}\n")

    if not matched:
        print("[ERROR] Tidak ada pasangan yang cocok!")
        print("Kemungkinan penyebab:")
        print("  1. Gambar di-rename manual (000001, 000002, dst) sementara label")
        print("     masih pakai nama asli (green_banana_0020, IMG_20260430, dst).")
        print("  2. Solusi: gunakan folder 'images' dari export Label Studio langsung,")
        print("     jangan di-rename manual sebelum dijalankan script ini.")
        return

    train_stems, val_stems = split_train_val(matched)
    print(f"Split -> Train: {len(train_stems)} | Val: {len(val_stems)}\n")

    if img_only:
        print(f"[INFO] {len(img_only)} gambar tidak punya label (akan diabaikan):")
        for s in sorted(img_only)[:10]:
            print(f"  {s}")
        if len(img_only) > 10:
            print(f"  ... dan {len(img_only)-10} lainnya")
        print()

    if lbl_only:
        print(f"[INFO] {len(lbl_only)} label tidak punya gambar (akan diabaikan):")
        for s in sorted(lbl_only)[:10]:
            print(f"  {s}")
        if len(lbl_only) > 10:
            print(f"  ... dan {len(lbl_only)-10} lainnya")
        print()

    if DRY_RUN:
        print("=" * 70)
        print("  DRY RUN selesai. Tidak ada file yang diubah.")
        print(f"  Jika sudah OK, ubah DRY_RUN = False lalu jalankan lagi.")
        print("=" * 70)
        return

    # ---- Buat struktur output ----
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            Path(os.path.join(OUTPUT_DIR, split, sub)).mkdir(parents=True, exist_ok=True)

    errors = 0

    def copy_pair(stem, split):
        nonlocal errors
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

    for stem in train_stems:
        copy_pair(stem, "train")
    for stem in val_stems:
        copy_pair(stem, "val")

    # ---- Buat dataset.yaml ----
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    yaml_content = f"""train: {os.path.join(OUTPUT_DIR, 'train', 'images')}
val: {os.path.join(OUTPUT_DIR, 'val', 'images')}

nc: 4

names:
  0: mentah
  1: mengkal
  2: matang
  3: busuk
"""
    if not DRY_RUN:
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

    print("=" * 70)
    if errors == 0:
        print(f"  SELESAI!")
        print(f"  Train : {len(train_stems)} pasang -> {OUTPUT_DIR}/train/")
        print(f"  Val   : {len(val_stems)} pasang  -> {OUTPUT_DIR}/val/")
        print(f"  YAML  : {yaml_path}")
        print()
        print("  Langkah selanjutnya:")
        print("  1. Upload folder OUTPUT_DIR ke Google Drive")
        print("  2. Buka banana_training_colab.ipynb di Google Colab")
        print("  3. Sesuaikan DATASET_PATH di notebook, lalu run all!")
    else:
        print(f"  Selesai dengan {errors} error. Cek log di atas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
