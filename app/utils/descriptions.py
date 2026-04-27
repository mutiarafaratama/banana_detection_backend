"""
Static descriptions per banana quality class.
Returned alongside detection results so the mobile app can render
detail cards without hard-coding text on the client side.
"""

QUALITY_DESCRIPTIONS = {
    'Mentah': {
        'title': 'Pisang Belum Matang',
        'description': (
            'Pisang masih dalam tahap mentah. Warna hijau mendominasi '
            'dengan tekstur yang keras dan kadar pati tinggi.'
        ),
        'characteristics': [
            'Warna dominan hijau',
            'Tekstur keras dan kaku',
            'Kadar pati tinggi, gula rendah',
            'Tidak cocok dikonsumsi langsung',
        ],
        'recommendations': [
            'Tunggu 2-3 hari untuk mencapai kematangan optimal',
            'Simpan di suhu ruangan (25-30°C)',
            'Hindari paparan sinar matahari langsung',
            'Cocok untuk dimasak / digoreng / direbus',
        ],
        'ripening_time': '2-3 hari',
        'storage_tips': 'Simpan di tempat teduh, suhu ruangan',
        'health_benefits': [
            'Resistant starch baik untuk pencernaan',
            'Kadar gula rendah, cocok untuk diet',
            'Kaya vitamin B6 dan kalium',
        ],
        'color_hex': '#7CB342',
    },
    'Matang': {
        'title': 'Pisang Matang Sempurna',
        'description': (
            'Pisang dalam kondisi matang optimal, siap dikonsumsi '
            'dengan rasa manis alami dan tekstur lembut.'
        ),
        'characteristics': [
            'Warna kuning cerah dengan bintik coklat minimal',
            'Tekstur lembut dan mudah dikupas',
            'Kadar gula optimal',
            'Aroma harum khas pisang matang',
        ],
        'recommendations': [
            'Konsumsi segera untuk rasa terbaik',
            'Simpan maksimal 1-2 hari di suhu ruangan',
            'Dapat disimpan di kulkas untuk memperpanjang kesegaran',
            'Cocok untuk dikonsumsi langsung atau smoothie',
        ],
        'ripening_time': 'Siap konsumsi',
        'storage_tips': 'Konsumsi dalam 1-2 hari, atau simpan di kulkas',
        'health_benefits': [
            'Sumber energi cepat dari gula alami',
            'Tinggi kalium untuk kesehatan jantung',
            'Vitamin C dan B6 untuk sistem imun',
            'Serat alami untuk pencernaan',
        ],
        'color_hex': '#FBC02D',
    },
    'Busuk': {
        'title': 'Pisang Terlalu Matang / Busuk',
        'description': (
            'Pisang sudah melewati masa kematangan optimal. '
            'Banyak bercak coklat / hitam dan tekstur sangat lembek.'
        ),
        'characteristics': [
            'Warna coklat kehitaman dominan',
            'Tekstur sangat lembek',
            'Kadar gula sangat tinggi',
            'Mungkin mulai berair atau berbau fermentasi',
        ],
        'recommendations': [
            'Tidak direkomendasikan dikonsumsi langsung',
            'Cocok untuk banana bread atau kue',
            'Bisa untuk smoothie jika belum berbau',
            'Jika sudah berbau busuk, sebaiknya dibuang',
        ],
        'ripening_time': 'Lewat masa optimal',
        'storage_tips': 'Gunakan segera atau buang',
        'health_benefits': [
            'Masih mengandung nutrisi jika belum busuk total',
            'Kadar antioksidan cenderung meningkat',
            'Cocok untuk baking (menambah rasa manis alami)',
        ],
        'color_hex': '#6D4C41',
    },
}


def get_quality_description(quality_class):
    """
    Return description payload for a given class name.
    Falls back to a generic placeholder when class name is unknown.
    """
    if quality_class in QUALITY_DESCRIPTIONS:
        return QUALITY_DESCRIPTIONS[quality_class]

    return {
        'title': quality_class,
        'description': 'Tidak ada deskripsi tersedia untuk kelas ini.',
        'characteristics': [],
        'recommendations': [],
        'ripening_time': '-',
        'storage_tips': '-',
        'health_benefits': [],
        'color_hex': '#9E9E9E',
    }


def confidence_label(confidence):
    """Convert numeric confidence into a coarse label."""
    if confidence >= 0.85:
        return 'Sangat Yakin'
    if confidence >= 0.7:
        return 'Yakin'
    if confidence >= 0.5:
        return 'Cukup Yakin'
    return 'Kurang Yakin'


def enrich_detection(detection):
    """
    Add `description` + `confidence_label` to a detection dict.
    Mutates and returns the input for convenience.
    """
    quality = detection.get('class', '')
    detection['description'] = get_quality_description(quality)
    detection['confidence_label'] = confidence_label(detection.get('confidence', 0))
    return detection


def enrich_detections(detections):
    """Apply enrich_detection over a list."""
    return [enrich_detection(d) for d in detections]


def summarize_detections(detections):
    """
    Build a small summary block useful for live-detection HUD or list header.
    """
    by_class = {}
    confidences = []
    for d in detections:
        cls = d.get('class', 'Unknown')
        by_class[cls] = by_class.get(cls, 0) + 1
        confidences.append(d.get('confidence', 0))

    dominant_class = None
    if by_class:
        dominant_class = max(by_class.items(), key=lambda kv: kv[1])[0]

    return {
        'total': len(detections),
        'by_class': by_class,
        'dominant_class': dominant_class,
        'avg_confidence': round(sum(confidences) / len(confidences), 3) if confidences else 0,
    }
