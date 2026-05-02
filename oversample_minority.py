"""
==============================================================
  OVERSAMPLE MINORITY CLASSES
  Mengatasi class imbalance dengan menduplikat gambar
  yang mengandung kelas minoritas di folder TRAIN.
==============================================================

Cara kerja:
  - Hitung bbox per kelas di semua gambar train
  - Tentukan kelas dominan (paling banyak bbox)
  - Duplikat gambar yang mengandung kelas minoritas
    hingga distribusi lebih seimbang
  - File duplikat diberi nama: nama_asli_aug1.jpg, _aug2.jpg, dst

PENTING:
  - Jalankan SETELAH prepare_dataset_local.py selesai
  - Hanya mempengaruhi folder TRAIN (val tidak berubah)
  - Tidak menghapus file asli, hanya menambah salinan

Cara pakai:
  1. Setel TRAIN_DIR di bawah
  2. Jalankan: python oversample_minority.py --dry-run
     -> Preview berapa file yang akan ditambah
  3. Kalau OK, hapus --dry-run:
     python oversample_minority.py
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ================================================================
# KONFIGURASI
# ================================================================

# Folder train hasil dari prepare_dataset_local.py
TRAIN_IMAGES_DIR = r"D:\banana_dataset_colab\train\images"
TRAIN_LABELS_DIR = r"D:\banana_dataset_colab\train\labels"

# Target: semua kelas minimal sekian persen dari kelas dominan
# 0.6 = target minimal 60% dari jumlah bbox kelas terbanyak
TARGET_RATIO = 0.6

CLASS_NAMES = {0: "mentah", 1: "mengkal", 2: "matang", 3: "busuk"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
                    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

# ================================================================


def get_classes_in_label(label_path):
    """Return set of class_ids yang ada dalam satu file label."""
    classes = set()
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    except Exception:
        pass
    return classes


def count_bbox_per_class(labels_dir):
    """Hitung total bbox per class dari semua label di folder."""
    counts = defaultdict(int)
    for fname in os.listdir(labels_dir):
        if not fname.lower().endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


def build_image_label_map(images_dir, labels_dir):
    """
    Buat mapping: stem -> (image_path, label_path)
    untuk semua pasangan yang lengkap.
    """
    img_map = {}
    for fname in os.listdir(images_dir):
        ext = Path(fname).suffix
        if ext not in IMAGE_EXTENSIONS:
            continue
        stem = Path(fname).stem
        img_map[stem] = (os.path.join(images_dir, fname), ext)

    lbl_map = {}
    for fname in os.listdir(labels_dir):
        if fname.lower().endswith(".txt"):
            stem = Path(fname).stem
            lbl_map[stem] = os.path.join(labels_dir, fname)

    paired = {}
    for stem in set(img_map) & set(lbl_map):
        paired[stem] = (img_map[stem][0], img_map[stem][1], lbl_map[stem])
    return paired


def compute_oversample_plan(paired, labels_dir, target_ratio):
    """
    Hitung berapa kali setiap gambar perlu diduplikat.

    Logika:
      1. Hitung total bbox per kelas saat ini
      2. Tentukan target minimum = target_ratio * max_count
      3. Untuk setiap kelas di bawah target, hitung kekurangan
      4. Tentukan faktor duplikat untuk gambar yang mengandung kelas itu
    """
    # Bbox count sekarang
    current = count_bbox_per_class(labels_dir)
    max_count = max(current.values()) if current else 1
    target = {cls: int(max_count * target_ratio) for cls in CLASS_NAMES}

    print("\nDistribusi bbox sekarang (train):")
    for cls_id, name in CLASS_NAMES.items():
        count = current.get(cls_id, 0)
        tgt   = target[cls_id]
        diff  = tgt - count
        status = f"kurang {diff:+d}" if diff > 0 else "sudah cukup"
        bar = "=" * int(count / max_count * 30)
        print(f"  {cls_id} {name:8s}: {count:5d} bbox ({count/max_count*100:5.1f}%) [{bar}]  <- {status}")

    print(f"\nTarget minimum  : {int(max_count * target_ratio)} bbox per kelas ({int(target_ratio*100)}% dari dominan)")
    print(f"Kelas dominan   : {max_count} bbox")

    # Kumpulkan bbox per gambar per kelas
    stem_class_bbox = {}  # stem -> {cls_id: count}
    for stem, (img_path, ext, lbl_path) in paired.items():
        bbox_by_class = defaultdict(int)
        try:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        bbox_by_class[int(parts[0])] += 1
        except Exception:
            pass
        stem_class_bbox[stem] = dict(bbox_by_class)

    # Hitung kekurangan per kelas
    shortage = {}  # cls_id -> jumlah bbox yang masih kurang
    for cls_id in CLASS_NAMES:
        gap = target[cls_id] - current.get(cls_id, 0)
        if gap > 0:
            shortage[cls_id] = gap

    if not shortage:
        print("\nSemua kelas sudah memenuhi target. Tidak perlu oversampling.")
        return {}

    print(f"\nKelas yang perlu ditambah: {[CLASS_NAMES[c] for c in shortage]}")

    # Tentukan faktor duplikat untuk setiap gambar
    # Pilih gambar yang paling banyak berkontribusi ke kelas yang kurang
    oversample_plan = {}  # stem -> n_copies

    for cls_id, gap in shortage.items():
        # Urutkan gambar berdasarkan jumlah bbox kelas ini (terbanyak dulu)
        candidates = sorted(
            [(stem, bbox.get(cls_id, 0)) for stem, bbox in stem_class_bbox.items()
             if bbox.get(cls_id, 0) > 0],
            key=lambda x: -x[1]
        )

        if not candidates:
            print(f"  [PERINGATAN] Tidak ada gambar dengan kelas {CLASS_NAMES[cls_id]}!")
            continue

        avg_bbox_per_img = sum(c for _, c in candidates) / len(candidates)
        n_copies_needed  = max(1, int(gap / avg_bbox_per_img)) if avg_bbox_per_img > 0 else 1

        # Distribusikan salinan ke semua kandidat secara merata
        copies_per_img = max(1, round(n_copies_needed / len(candidates)))
        copies_per_img = min(copies_per_img, 8)  # maksimal 8 salinan per gambar

        for stem, _ in candidates:
            if stem not in oversample_plan:
                oversample_plan[stem] = 0
            oversample_plan[stem] = max(oversample_plan[stem], copies_per_img)

    return oversample_plan


def main(dry_run: bool):
    print("=" * 60)
    print("  OVERSAMPLE MINORITY CLASSES")
    print(f"  Mode: {'DRY RUN' if dry_run else 'AKTIF - file akan disalin'}")
    print("=" * 60)
    sys.stdout.flush()

    for label, path in [("TRAIN_IMAGES_DIR", TRAIN_IMAGES_DIR),
                        ("TRAIN_LABELS_DIR", TRAIN_LABELS_DIR)]:
        if not os.path.isdir(path):
            print(f"\n[ERROR] {label} tidak ditemukan: {path}")
            print("  Pastikan prepare_dataset_local.py sudah dijalankan dulu!")
            return

    paired = build_image_label_map(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    print(f"\nGambar train ditemukan : {len(paired)}")

    oversample_plan = compute_oversample_plan(paired, TRAIN_LABELS_DIR, TARGET_RATIO)

    if not oversample_plan:
        return

    total_new_files = sum(n for n in oversample_plan.values())
    print(f"\nRencana duplikat:")
    print(f"  Gambar yang akan diduplikat : {len(oversample_plan)}")
    print(f"  Total file baru             : {total_new_files} (gambar + label)")
    print(f"  Train setelah oversample    : {len(paired) + total_new_files} gambar")
    print()

    # Preview 10 gambar yang akan diduplikat
    print("Contoh gambar yang diduplikat (10 pertama):")
    for stem, n in list(oversample_plan.items())[:10]:
        print(f"  {stem} -> {n}x salinan")
    if len(oversample_plan) > 10:
        print(f"  ... dan {len(oversample_plan)-10} gambar lainnya")

    if dry_run:
        print()
        print("=" * 60)
        print("  DRY RUN selesai.")
        print("  Jalankan tanpa --dry-run untuk mulai duplikat:")
        print("    python oversample_minority.py")
        print("=" * 60)
        return

    # Eksekusi duplikat
    print("\nMulai menduplikat file...")
    errors  = 0
    created = 0

    for stem, n_copies in oversample_plan.items():
        img_src, ext, lbl_src = paired[stem]
        for i in range(1, n_copies + 1):
            new_stem  = f"{stem}_aug{i}"
            img_dst   = os.path.join(TRAIN_IMAGES_DIR, new_stem + ext)
            lbl_dst   = os.path.join(TRAIN_LABELS_DIR, new_stem + ".txt")

            # Skip kalau sudah ada (idempotent)
            if os.path.exists(img_dst):
                continue
            try:
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)
                created += 1
            except Exception as e:
                print(f"  [ERROR] {new_stem}: {e}")
                errors += 1

    # Hitung distribusi setelah oversampling
    print("\nDistribusi bbox SETELAH oversampling (train):")
    after = count_bbox_per_class(TRAIN_LABELS_DIR)
    max_after = max(after.values()) if after else 1
    for cls_id, name in CLASS_NAMES.items():
        count = after.get(cls_id, 0)
        bar   = "=" * int(count / max_after * 30)
        print(f"  {cls_id} {name:8s}: {count:5d} bbox ({count/max_after*100:5.1f}%) [{bar}]")

    print()
    print("=" * 60)
    if errors == 0:
        print(f"  SELESAI! {created} file baru ditambahkan ke folder train.")
    else:
        print(f"  Selesai dengan {errors} error.")
    print()
    print("  LANGKAH SELANJUTNYA:")
    print("  Upload ulang folder banana_dataset_colab ke Google Drive")
    print("  (timpa yang lama), lalu jalankan Colab notebook!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
