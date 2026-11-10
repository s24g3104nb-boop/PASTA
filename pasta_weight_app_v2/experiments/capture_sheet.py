"""Reproducible vector A4 reference prototype; no photographic data is embedded."""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

SHEET_VERSION = 'pasta-a4-markers-v1'
WIDTH_MM, HEIGHT_MM = 210.0, 297.0
MARKER_MM = 20.0
MARKER_POSITIONS = {0: (10., 10.), 1: (180., 10.), 2: (180., 267.), 3: (10., 267.)}
PIXELS_PER_MM = 2.0
DICTIONARY_ID = cv2.aruco.DICT_4X4_50


def marker_bits(marker_id):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, 6)


def marker_corners_mm(marker_id):
    x, y = MARKER_POSITIONS[marker_id]
    return np.array([[x, y], [x + MARKER_MM, y],
                     [x + MARKER_MM, y + MARKER_MM], [x, y + MARKER_MM]], np.float32)


def render_sheet(pixels_per_mm=4.0):
    """Synthetic fixture. It is never evidence of real camera/printer accuracy."""
    if not np.isfinite(pixels_per_mm) or not 1 <= pixels_per_mm <= 10:
        raise ValueError('pixels_per_mm must be between 1 and 10')
    result = np.full((round(HEIGHT_MM * pixels_per_mm), round(WIDTH_MM * pixels_per_mm), 3), 255, np.uint8)
    for marker_id, (x, y) in MARKER_POSITIONS.items():
        size = round(MARKER_MM * pixels_per_mm)
        marker = cv2.resize(marker_bits(marker_id), (size, size), interpolation=cv2.INTER_NEAREST)
        px, py = round(x * pixels_per_mm), round(y * pixels_per_mm)
        result[py:py + size, px:px + size] = marker[:, :, None]
    return result


def build_svg():
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
             '<rect width="210" height="297" fill="white"/>']
    for marker_id, (x, y) in MARKER_POSITIONS.items():
        bits = marker_bits(marker_id)
        for row, column in zip(*np.where(bits == 0)):
            # Joining adjacent cells removes thin PDF/browser hairlines.
            lines.append(f'<rect x="{x + column * MARKER_MM / 6:.8f}" y="{y + row * MARKER_MM / 6:.8f}" width="{MARKER_MM / 6 + .00001:.8f}" height="{MARKER_MM / 6 + .00001:.8f}" fill="black"/>')
    lines += ['<g font-family="sans-serif" fill="#29352E" text-anchor="middle">',
              '<text x="105" y="16" font-size="5">PASTA LENS | A4 reference v1</text>',
              '<text x="105" y="23" font-size="3">RESEARCH PROTOTYPE - NO WEIGHT PREDICTION</text>',
              '<text x="105" y="36" font-size="3">Place ALL pasta ends inside the dotted zone.</text>',
              '<text x="105" y="42" font-size="3">Long strands: arrange diagonally. Do not cover markers.</text>',
              '<text x="105" y="272" font-size="3">PRINT AT 100% / ACTUAL SIZE - DO NOT FIT</text>',
              '<text x="105" y="286" font-size="3">Verify this line is exactly 100 mm with a ruler.</text>',
              '<text x="105" y="293" font-size="2.6">A4 210 x 297 mm | DICT_4X4_50 | IDs 0,1,2,3 clockwise</text>', '</g>',
              '<rect x="35" y="47" width="140" height="213" rx="2" fill="none" stroke="#99A69C" stroke-width="0.25" stroke-dasharray="2 2"/>',
              '<path d="M55 277 H155 M55 275 V279 M155 275 V279" fill="none" stroke="black" stroke-width="0.25"/>', '</svg>']
    return '\n'.join(lines)


def write_pdf(path):
    """Optional reportlab dependency is needed only to regenerate this PDF."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    doc = canvas.Canvas(str(path), pagesize=(WIDTH_MM * mm, HEIGHT_MM * mm), invariant=1)
    doc.setTitle('Pasta Lens A4 reference v1 - research prototype')
    for marker_id, (x, y) in MARKER_POSITIONS.items():
        for row, column in zip(*np.where(marker_bits(marker_id) == 0)):
            cell = MARKER_MM / 6
            doc.rect((x + column * cell) * mm, (HEIGHT_MM - y - (row + 1) * cell) * mm,
                     (cell + .00001) * mm, (cell + .00001) * mm, fill=1, stroke=0)
    labels = [(16, 5, 'PASTA LENS | A4 reference v1'),
              (23, 3, 'RESEARCH PROTOTYPE - NO WEIGHT PREDICTION'),
              (36, 3, 'Place ALL pasta ends inside the dotted zone.'),
              (42, 3, 'Long strands: arrange diagonally. Do not cover markers.'),
              (272, 3, 'PRINT AT 100% / ACTUAL SIZE - DO NOT FIT'),
              (286, 3, 'Verify this line is exactly 100 mm with a ruler.'),
              (293, 2.6, 'A4 210 x 297 mm | DICT_4X4_50 | IDs 0,1,2,3 clockwise')]
    for y, size, label in labels:
        doc.setFont('Helvetica', size * mm)
        doc.drawCentredString(105 * mm, (HEIGHT_MM - y) * mm, label)
    doc.setStrokeColorRGB(.60, .65, .61)
    doc.setLineWidth(.25 * mm)
    doc.setDash(2 * mm, 2 * mm)
    doc.roundRect(35 * mm, (HEIGHT_MM - 260) * mm, 140 * mm, 213 * mm, 2 * mm)
    doc.setDash()
    doc.setStrokeColorRGB(0, 0, 0)
    doc.line(55 * mm, 20 * mm, 155 * mm, 20 * mm)
    for x in [55, 155]:
        doc.line(x * mm, 18 * mm, x * mm, 22 * mm)
    doc.showPage()
    doc.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'Pasta_Lens_A4_reference_v1.svg').write_text(build_svg(), encoding='utf-8')
    write_pdf(args.output / 'Pasta_Lens_A4_reference_v1.pdf')
    (args.output / 'geometry.json').write_text(json.dumps(dict(version=SHEET_VERSION,
        sheet_mm=[WIDTH_MM, HEIGHT_MM], marker_mm=MARKER_MM,
        dictionary='DICT_4X4_50', marker_positions_mm=MARKER_POSITIONS,
        ruler_check_mm=100, fixture_type='synthetic_reference_design'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
