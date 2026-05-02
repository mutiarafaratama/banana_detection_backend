"""
==============================================================
  BANANA DATASET PREPARATION SCRIPT
  Untuk: merapikan dataset dari Label Studio -> siap Google Colab
==============================================================

CARA PAKAI:
  1. Setel IMAGES_DIR, LABELS_DIR, dan OUTPUT_DIR di bagian KONFIGURASI
  2. Jalankan: python prepare_dataset_local.py --dry-run
     -> Preview hasilnya, tidak ada file yang diubah
  3. Jika sudah OK: python prepare_dataset_local.py
     -> Dataset bersih tersimpan di OUTPUT_DIR

STRUKTUR OUTPUT (siap upload ke Google Drive):
  OUTPUT_DIR/
    train/
      images/   (80% data)
      labels/
    val/
      images/   (20% data)
      labels/
    dataset.yaml
    classes.txt
    report.txt  <- ringkasan hasil
"""

import os
import re
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ================================================================
# KONFIGURASI - SESUAIKAN PATH DI BAWAH INI
# ================================================================

# Folder images dari Label Studio YOLO export
# (atau folder gambar yang kamu punya)
IMAGES_DIR = r"D:\banana_dataset\images"

# Folder labels dari Label Studio YOLO export
LABELS_DIR = r"D:\banana_dataset\labels"

# Folder output (akan dibuat otomatis)
OUTPUT_DIR = r"D:\banana_dataset_colab"

# Rasio split train/val (0.2 = 20% untuk validasi)
VAL_RATIO = 0.2

# Random seed untuk hasil split yang konsisten
RANDOM_SEED = 42

# ================================================================
# KELAS - SESUAIKAN JIKA PERLU
# 0=mentah, 1=mengkal, 2=matang, 3=busuk
# ================================================================
CLASS_NAMES = {
    0: "mentah",
    1: "mengkal",
    2: "matang",
    3: "busuk",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


# ================================================================
# FUNGSI UTILITY
# ================================================================

def normalize_stem(stem: str) -> str:
    """
    Normalisasi nama file:
      '000001 (2)' -> '000001_2'
      'IMG 20260430 (3)' -> 'IMG_20260430_3'
    """
    s = stem.replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def strip_uuid_prefix(stem: str) -> str:
    """
    Hapus UUID prefix dari nama label Label Studio.

    Label Studio menggunakan berbagai format prefix:
      Pola A: xxxxxxxx__xxxxxxxx-nama_file    (double underscore)
      Pola B: xxxxxxxx_xxxxxxxx-nama_file     (single underscore)
      Pola C: standard UUID + nama_file

    Contoh:
      '74eaf18c__ef0a5cb9-000001_2'           -> '000001_2'
      '002da5db__c819c789-green_banana_0020'   -> 'green_banana_0020'
      '006fe92d__1fce6d24-IMG_20260430_174642' -> 'IMG_20260430_174642'
    """
    # Pola A: double underscore (Label Studio versi terbaru)
    m = re.match(r'^[0-9a-f]+__[0-9a-f]+-(.+)$', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Pola B: single underscore (8hex_8hex-)
    m = re.match(r'^[0-9a-f]{8}_[0-9a-f]{8}-(.+)$', stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Pola C: standard UUID (8-4-4-4-12)
    m = re.match(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-(.+)$',
        stem, re.IGNORECASE
    )
    if m:
        return m.group(1)

    return stem  # tidak ada prefix dikenali


def collect_images(images_dir: str) -> dict:
    """
    Kumpulkan semua gambar.
    Return: { normalized_stem: (absolute_path, extension) }
    """
    result = {}
    duplicates = []
    for fname in os.listdir(images_dir):
        ext = Path(fname).suffix
        if ext not in IMAGE_EXTENSIONS:
            continue
        stem = Path(fname).stem
        norm = normalize_stem(stem)
        path = os.path.join(images_dir, fname)
        if norm in result:
            duplicates.append((norm, result[norm][0], path))
        else:
            result[norm] = (path, ext)
    return result, duplicates


def collect_labels(labels_dir: str) -> dict:
    """
    Kumpulkan semua label, strip UUID prefix.
    Return: { cleaned_stem: absolute_path }
    """
    result = {}
    duplicates = []
    for fname in os.listdir(labels_dir):
        if not fname.lower().endswith(".txt"):
            continue
        stem = Path(fname).stem
        cleaned = strip_uuid_prefix(stem)
        path = os.path.join(labels_dir, fname)
        if cleaned in result:
            duplicates.append((cleaned, result[cleaned], path))
        else:
            result[cleaned] = path
    return result, duplicates


def validate_label_file(label_path: str, num_classes: int) -> list:
    """
    Validasi isi file label YOLO.
    Return: list of errors (kosong = valid)
    """
    errors = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            errors.append("file kosong")
            return errors
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                errors.append(f"baris {i}: harusnya 5 kolom, ada {len(parts)}")
                continue
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= num_classes:
                errors.append(f"baris {i}: class_id {cls_id} diluar range 0-{num_classes-1}")
            for j, val in enumerate(parts[1:], 2):
                v = float(val)
                if not (0.0 <= v <= 1.0):
                    errors.append(f"baris {i}: kolom {j} nilai {v} diluar range 0-1")
    except Exception as e:
        errors.append(f"tidak bisa dibaca: {e}")
    return errors


def analyze_class_distribution(label_paths: list) -> dict:
    """Hitung distribusi kelas dari semua label."""
    counts = defaultdict(int)
    for path in label_paths:
        try:
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        counts[int(parts[0])] += 1
        except Exception:
            pass
    return dict(counts)


# ================================================================
# MAIN
# ================================================================

def main(dry_run: bool):
    print("=" * 65)
    print("  BANANA DATASET PREPARATION")
    print(f"  Mode: {'DRY RUN (preview saja)' if dry_run else 'AKTIF - file akan disalin'}")
    print("=" * 65)

    # --- Cek folder ---
    for label, path in [("IMAGES_DIR", IMAGES_DIR), ("LABELS_DIR", LABELS_DIR)]:
        if not os.path.isdir(path):
            print(f"\n[ERROR] Folder {label} tidak ditemukan: {path}")
            print("  -> Setel path yang benar di bagian KONFIGURASI script ini.")
            return

    print(f"\nImages : {IMAGES_DIR}")
    print(f"Labels : {LABELS_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Split  : {int((1-VAL_RATIO)*100)}% train / {int(VAL_RATIO*100)}% val\n")

    # --- Kumpulkan file ---
    print("Membaca folder...")
    image_map, img_dupes = collect_images(IMAGES_DIR)
    label_map, lbl_dupes = collect_labels(LABELS_DIR)

    print(f"  Gambar ditemukan : {len(image_map)}")
    print(f"  Label ditemukan  : {len(label_map)}")

    if img_dupes:
        print(f"\n[PERINGATAN] {len(img_dupes)} duplikat nama gambar setelah normalisasi:")
        for norm, p1, p2 in img_dupes[:5]:
            print(f"  '{norm}': {os.path.basename(p1)} vs {os.path.basename(p2)}")
        if len(img_dupes) > 5:
            print(f"  ... dan {len(img_dupes)-5} lainnya")

    if lbl_dupes:
        print(f"\n[PERINGATAN] {len(lbl_dupes)} duplikat nama label setelah strip UUID:")
        for cleaned, p1, p2 in lbl_dupes[:5]:
            print(f"  '{cleaned}': {os.path.basename(p1)} vs {os.path.basename(p2)}")

    # --- Cocokkan ---
    matched_stems = sorted(set(image_map) & set(label_map))
    img_only      = sorted(set(image_map) - set(label_map))
    lbl_only      = sorted(set(label_map) - set(image_map))

    print(f"\nPasangan cocok  : {len(matched_stems)}")
    print(f"Gambar tanpa label : {len(img_only)}")
    print(f"Label tanpa gambar : {len(lbl_only)}")

    if len(matched_stems) == 0:
        print("\n[ERROR] Tidak ada pasangan yang cocok sama sekali!")
        print("  Kemungkinan penyebab:")
        print("  1. Gambar di-rename manual (000001, 000002..) tapi label masih nama asli")
        print("  2. Solusi: gunakan folder 'images' langsung dari export Label Studio")
        print("     (jangan rename gambar sebelum jalankan script ini)")
        print()
        print("  Contoh nama gambar yang ada:")
        for s in sorted(image_map)[:5]:
            print(f"    {s}")
        print("  Contoh nama label yang ada (setelah strip UUID):")
        for s in sorted(label_map)[:5]:
            print(f"    {s}")
        return

    # --- Validasi label ---
    print("\nMemvalidasi isi label...")
    invalid_labels = {}
    num_classes = len(CLASS_NAMES)
    for stem in matched_stems:
        errs = validate_label_file(label_map[stem], num_classes)
        if errs:
            invalid_labels[stem] = errs

    if invalid_labels:
        print(f"[PERINGATAN] {len(invalid_labels)} label bermasalah:")
        for stem, errs in list(invalid_labels.items())[:10]:
            print(f"  {stem}: {'; '.join(errs)}")
    else:
        print(f"  Semua {len(matched_stems)} label valid ✓")

    # --- Distribusi kelas ---
    all_label_paths = [label_map[s] for s in matched_stems]
    class_dist = analyze_class_distribution(all_label_paths)
    print("\nDistribusi kelas (jumlah bbox):")
    total_bbox = sum(class_dist.values())
    for cls_id, name in CLASS_NAMES.items():
        count = class_dist.get(cls_id, 0)
        pct   = count / total_bbox * 100 if total_bbox else 0
        bar   = "█" * int(pct / 2)
        print(f"  {cls_id} {name:8s}: {count:5d} bbox ({pct:5.1f}%) {bar}")

    if img_only:
        print(f"\nGambar TANPA label (akan diabaikan, {len(img_only)} file):")
        for s in img_only[:10]:
            print(f"  {s}")
        if len(img_only) > 10:
            print(f"  ... dan {len(img_only)-10} lainnya")

    if lbl_only:
        print(f"\nLabel TANPA gambar (akan diabaikan, {len(lbl_only)} file):")
        for s in lbl_only[:10]:
            print(f"  {s}")
        if len(lbl_only) > 10:
            print(f"  ... dan {len(lbl_only)-10} lainnya")

    # --- Split train/val ---
    random.seed(RANDOM_SEED)
    stems_shuffled = list(matched_stems)
    random.shuffle(stems_shuffled)
    n_val   = max(1, int(len(stems_shuffled) * VAL_RATIO))
    val_stems   = set(stems_shuffled[:n_val])
    train_stems = set(stems_shuffled[n_val:])

    print(f"\nSplit hasil:")
    print(f"  Train : {len(train_stems)} gambar")
    print(f"  Val   : {len(val_stems)} gambar")
    print(f"  Total : {len(matched_stems)} gambar\n")

    if dry_run:
        print("=" * 65)
        print("  DRY RUN selesai. Tidak ada file yang diubah.")
        print("  Jika hasilnya OK, jalankan tanpa --dry-run:")
        print("    python prepare_dataset_local.py")
        print("=" * 65)
        return

    # --- Buat struktur output ---
    print("Membuat folder output...")
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            Path(os.path.join(OUTPUT_DIR, split, sub)).mkdir(parents=True, exist_ok=True)

    # --- Salin file ---
    print("Menyalin file...")
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

    for i, stem in enumerate(sorted(train_stems), 1):
        copy_pair(stem, "train")
        if i % 100 == 0:
            print(f"  Train: {i}/{len(train_stems)} selesai...")

    for i, stem in enumerate(sorted(val_stems), 1):
        copy_pair(stem, "val")
        if i % 50 == 0:
            print(f"  Val: {i}/{len(val_stems)} selesai...")

    # --- Buat dataset.yaml ---
    yaml_content = f"""# Banana Detection Dataset
# Generated by prepare_dataset_local.py
# Classes: {len(CLASS_NAMES)} kelas

# PENTING: Saat upload ke Google Drive, sesuaikan path train & val
# dengan lokasi folder di Colab (biasanya /content/drive/MyDrive/...)

train: {os.path.join(OUTPUT_DIR, 'train', 'images').replace(chr(92), '/')}
val:   {os.path.join(OUTPUT_DIR, 'val', 'images').replace(chr(92), '/')}

# Jumlah kelas
nc: {len(CLASS_NAMES)}

# Nama kelas (urutan HARUS sama dengan class_id di file label)
names:
  0: mentah
  1: mengkal
  2: matang
  3: busuk
"""

    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # --- Buat classes.txt ---
    classes_path = os.path.join(OUTPUT_DIR, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for cls_id in sorted(CLASS_NAMES):
            f.write(CLASS_NAMES[cls_id] + "\n")

    # --- Buat report.txt ---
    train_dist = analyze_class_distribution([label_map[s] for s in train_stems])
    val_dist   = analyze_class_distribution([label_map[s] for s in val_stems])

    report = []
    report.append("=" * 60)
    report.append("  BANANA DATASET REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal gambar berlabel : {len(matched_stems)}")
    report.append(f"Train                 : {len(train_stems)}")
    report.append(f"Val                   : {len(val_stems)}")
    report.append(f"Gambar tanpa label    : {len(img_only)}")
    report.append(f"Label tanpa gambar    : {len(lbl_only)}")
    report.append(f"\nDistribusi kelas (train):")
    for cls_id, name in CLASS_NAMES.items():
        report.append(f"  {cls_id} {name:8s}: {train_dist.get(cls_id, 0):5d} bbox")
    report.append(f"\nDistribusi kelas (val):")
    for cls_id, name in CLASS_NAMES.items():
        report.append(f"  {cls_id} {name:8s}: {val_dist.get(cls_id, 0):5d} bbox")
    if invalid_labels:
        report.append(f"\nLabel bermasalah ({len(invalid_labels)}):")
        for stem, errs in invalid_labels.items():
            report.append(f"  {stem}: {'; '.join(errs)}")

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # --- Ringkasan akhir ---
    print()
    print("=" * 65)
    if errors == 0:
        print("  SELESAI TANPA ERROR!")
    else:
        print(f"  Selesai dengan {errors} error (cek log di atas)")
    print()
    print(f"  Folder output  : {OUTPUT_DIR}")
    print(f"  Train          : {len(train_stems)} gambar")
    print(f"  Val            : {len(val_stems)} gambar")
    print(f"  dataset.yaml   : {yaml_path}")
    print(f"  classes.txt    : {classes_path}")
    print(f"  report.txt     : {report_path}")
    print()
    print("  LANGKAH SELANJUTNYA:")
    print("  1. Buka report.txt untuk cek ringkasan dataset")
    print("  2. Upload folder OUTPUT_DIR ke Google Drive")
    print(f"     (folder: {os.path.basename(OUTPUT_DIR)})")
    print("  3. Buka banana_training_colab.ipynb di Google Colab")
    print("  4. Sesuaikan DATASET_PATH di Colab lalu Run All!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare banana dataset for YOLO training")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview saja tanpa menyalin file (default: False)"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
