"""
Script untuk menyinkronkan nama file images dan labels dari export Label Studio (YOLO format).

Masalah:
  - Label Studio menambahkan UUID prefix pada file label: 74eaf18c_ef0a5cb9-000001_2.txt
  - Image masih nama asli dengan spasi/kurung: 000001 (2).jpg

Solusi:
  - Rename images: normalisasi nama (spasi & kurung -> underscore)
  - Rename labels: hapus UUID prefix, sisakan nama yang sudah dinormalisasi

Cara pakai:
  1. Setel IMAGES_DIR dan LABELS_DIR di bawah sesuai lokasi foldermu
  2. Jalankan: python fix_dataset_names.py
     -> Ini akan DRY RUN (preview saja, tidak mengubah file)
  3. Jika hasilnya sudah sesuai, ubah DRY_RUN = False lalu jalankan lagi
"""

import os
import re
import shutil

# ============================================================
# KONFIGURASI - sesuaikan path di bawah ini
# ============================================================
IMAGES_DIR = r"D:\banana_dataset\images"
LABELS_DIR = r"D:\banana_dataset\labels"

DRY_RUN = True  # True = preview saja | False = benar-benar rename file
# ============================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_stem(stem: str) -> str:
    """
    Normalisasi nama file (tanpa ekstensi):
      - Spasi -> underscore
      - Kurung buka/tutup dihapus: '(2)' -> '2'  tapi umumnya sudah jadi '_2'
      - Tanda hubung berlebih / underscore ganda dibersihkan
    Contoh: '000001 (2)' -> '000001_2'
    """
    # Ganti spasi dengan underscore
    stem = stem.replace(" ", "_")
    # Hapus karakter kurung
    stem = stem.replace("(", "").replace(")", "")
    # Hapus underscore berlebih
    stem = re.sub(r"_+", "_", stem)
    # Hilangkan underscore di awal/akhir
    stem = stem.strip("_")
    return stem


def strip_uuid_prefix(stem: str) -> str:
    """
    Hapus UUID prefix yang ditambahkan Label Studio.
    Pola: {hex}-{hex}-{hex}-{hex}-{hex}-{nama_asli}
          atau {hex8}_{hex8}-{nama_asli}

    Label Studio memakai format: xxxxxxxx_xxxxxxxx-nama_asli
    Contoh: '74eaf18c_ef0a5cb9-000001_2' -> '000001_2'

    Strategi: cari tanda '-' terakhir yang sebelumnya adalah karakter hex,
    lalu ambil bagian setelahnya sebagai nama asli.
    """
    # Pola UUID Label Studio: 8hex_8hex- di awal
    pattern = r"^[0-9a-f]{8}_[0-9a-f]{8}-(.+)$"
    match = re.match(pattern, stem, re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback: split di '-' pertama dan ambil sisanya
    # (untuk format UUID yang lebih panjang seperti standard UUID)
    parts = stem.split("-")
    if len(parts) > 1:
        # Cek apakah bagian pertama terlihat seperti UUID (hanya hex & underscore)
        if re.match(r"^[0-9a-f_]+$", parts[0], re.IGNORECASE):
            return "-".join(parts[1:])

    # Tidak ada prefix UUID yang dikenali, kembalikan apa adanya
    return stem


def collect_images(images_dir):
    """Kumpulkan semua file gambar beserta normalized stem-nya."""
    image_map = {}  # normalized_stem -> (original_path, ext)
    for fname in os.listdir(images_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        norm = normalize_stem(stem)
        original_path = os.path.join(images_dir, fname)
        if norm in image_map:
            print(f"  [PERINGATAN] Duplikat normalized stem '{norm}': "
                  f"'{image_map[norm][0]}' vs '{original_path}'")
        image_map[norm] = (original_path, ext)
    return image_map


def collect_labels(labels_dir):
    """Kumpulkan semua file label beserta cleaned stem-nya."""
    label_map = {}  # cleaned_stem -> original_path
    for fname in os.listdir(labels_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".txt":
            continue
        cleaned = strip_uuid_prefix(stem)
        original_path = os.path.join(labels_dir, fname)
        if cleaned in label_map:
            print(f"  [PERINGATAN] Duplikat cleaned stem '{cleaned}': "
                  f"'{label_map[cleaned]}' vs '{original_path}'")
        label_map[cleaned] = original_path
    return label_map


def main():
    print("=" * 65)
    print("  Fix Dataset Names - Label Studio YOLO Export")
    print(f"  Mode: {'DRY RUN (tidak ada perubahan)' if DRY_RUN else 'RENAME SESUNGGUHNYA'}")
    print("=" * 65)

    if not os.path.isdir(IMAGES_DIR):
        print(f"\n[ERROR] Folder images tidak ditemukan: {IMAGES_DIR}")
        return
    if not os.path.isdir(LABELS_DIR):
        print(f"\n[ERROR] Folder labels tidak ditemukan: {LABELS_DIR}")
        return

    print(f"\nImages dir : {IMAGES_DIR}")
    print(f"Labels dir : {LABELS_DIR}\n")

    image_map = collect_images(IMAGES_DIR)
    label_map = collect_labels(LABELS_DIR)

    print(f"Ditemukan {len(image_map)} gambar (setelah normalisasi)")
    print(f"Ditemukan {len(label_map)} label  (setelah strip UUID)\n")

    matched = 0
    unmatched_images = []
    unmatched_labels = []

    rename_images = []  # list of (src, dst)
    rename_labels = []

    # Cari pasangan yang cocok
    all_stems = set(image_map.keys()) | set(label_map.keys())
    for stem in sorted(all_stems):
        has_img = stem in image_map
        has_lbl = stem in label_map

        if has_img and has_lbl:
            img_path, ext = image_map[stem]
            lbl_path = label_map[stem]

            img_new = os.path.join(IMAGES_DIR, stem + ext)
            lbl_new = os.path.join(LABELS_DIR, stem + ".txt")

            img_needs_rename = img_path != img_new
            lbl_needs_rename = lbl_path != lbl_new

            if img_needs_rename:
                rename_images.append((img_path, img_new))
            if lbl_needs_rename:
                rename_labels.append((lbl_path, lbl_new))

            matched += 1

        elif has_img and not has_lbl:
            unmatched_images.append(stem)
        else:
            unmatched_labels.append(stem)

    # --- Tampilkan rencana rename ---
    print(f"Pasangan cocok (image + label): {matched}")
    print(f"Image tanpa label             : {len(unmatched_images)}")
    print(f"Label tanpa image             : {len(unmatched_labels)}\n")

    if rename_images:
        print(f"[RENAME IMAGES] {len(rename_images)} file akan direname:")
        for src, dst in rename_images[:20]:
            print(f"  {os.path.basename(src):50s}  ->  {os.path.basename(dst)}")
        if len(rename_images) > 20:
            print(f"  ... dan {len(rename_images) - 20} file lainnya")
        print()

    if rename_labels:
        print(f"[RENAME LABELS] {len(rename_labels)} file akan direname:")
        for src, dst in rename_labels[:20]:
            print(f"  {os.path.basename(src):60s}  ->  {os.path.basename(dst)}")
        if len(rename_labels) > 20:
            print(f"  ... dan {len(rename_labels) - 20} file lainnya")
        print()

    if unmatched_images:
        print(f"[PERINGATAN] {len(unmatched_images)} image tidak punya label pasangan:")
        for s in unmatched_images[:10]:
            print(f"  {s}")
        if len(unmatched_images) > 10:
            print(f"  ... dan {len(unmatched_images) - 10} lainnya")
        print()

    if unmatched_labels:
        print(f"[PERINGATAN] {len(unmatched_labels)} label tidak punya image pasangan:")
        for s in unmatched_labels[:10]:
            print(f"  {s}")
        if len(unmatched_labels) > 10:
            print(f"  ... dan {len(unmatched_labels) - 10} lainnya")
        print()

    # --- Eksekusi rename ---
    if DRY_RUN:
        print("=" * 65)
        print("  DRY RUN selesai. Tidak ada file yang diubah.")
        print("  Jika hasilnya sudah benar, ubah DRY_RUN = False")
        print("  lalu jalankan script ini lagi.")
        print("=" * 65)
        return

    # Rename images
    errors = 0
    for src, dst in rename_images:
        if os.path.exists(dst) and src != dst:
            print(f"  [SKIP] Tujuan sudah ada: {dst}")
            continue
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"  [ERROR] {src} -> {dst}: {e}")
            errors += 1

    # Rename labels
    for src, dst in rename_labels:
        if os.path.exists(dst) and src != dst:
            print(f"  [SKIP] Tujuan sudah ada: {dst}")
            continue
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"  [ERROR] {src} -> {dst}: {e}")
            errors += 1

    print("=" * 65)
    if errors == 0:
        print(f"  Selesai! {len(rename_images)} image dan {len(rename_labels)} label berhasil direname.")
    else:
        print(f"  Selesai dengan {errors} error. Cek log di atas.")
    print("=" * 65)


if __name__ == "__main__":
    main()
