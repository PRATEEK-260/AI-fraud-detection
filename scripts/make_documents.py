"""Render synthetic KYC ID cards, then tamper half of them.

Why synthetic: there is no lawful public corpus of genuine identity documents,
and there should not be one. Rendering our own is the only responsible way to
build this — no real person's identity is involved, and nothing here can be
used as a template for a passable forgery (the cards carry an explicit
SPECIMEN mark, invented issuing authority, and no security features).

Why the signal is nevertheless real: the tampering is genuine image
manipulation, and Error Level Analysis keys on JPEG quantisation physics, not
on anything we invented. When a region is edited and the file re-saved, that
region has been through a different number of compression generations than its
surroundings, and its response to a further known-quality re-save differs
measurably. That discontinuity is a real artifact of the codec.

What this does NOT establish: that ELA catches a competent forger. A forger
who re-encodes the whole document at uniform quality, or works from a clean
re-render, erases exactly this signal. Detecting our own splices proves the
implementation is right, not that the method is sufficient. The agent reports
it in those terms.

Usage:
    .venv/bin/python scripts/make_documents.py [--n 240]
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "documents"
MANIFEST = ROOT / "data" / "processed" / "documents_manifest.csv"

SEED = 4242
W, H = 640, 400

FIRST = ["Aarav", "Diya", "Rohan", "Meera", "Kabir", "Ananya", "Vikram",
         "Ishita", "Arjun", "Priya", "Rahul", "Sneha", "Nikhil", "Tara"]
LAST = ["Sharma", "Patel", "Nair", "Reddy", "Iyer", "Gupta", "Mehta",
        "Bose", "Rao", "Chopra", "Desai", "Menon"]


def _font(size: int):
    for path in ("/usr/share/fonts/TTF/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_card(rng: random.Random, idx: int) -> tuple[Image.Image, dict]:
    """A plainly-marked specimen card. Deliberately not realistic."""
    bg = (238, 240, 246)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W - 1, 64], fill=(38, 62, 112))
    d.text((18, 20), "SPECIMEN — NOT A REAL DOCUMENT", font=_font(20),
           fill=(255, 255, 255))
    d.rectangle([0, H - 34, W - 1, H - 1], fill=(38, 62, 112))
    d.text((18, H - 27), "Synthetic test artifact · fraud-defense-system",
           font=_font(13), fill=(210, 218, 235))

    # Portrait placeholder — a gradient block, never a face.
    px, py = 26, 92
    for i in range(150):
        shade = 150 + int(60 * (i / 150))
        d.line([(px, py + i), (px + 130, py + i)], fill=(shade, shade - 8, shade - 20))
    d.rectangle([px, py, px + 130, py + 150], outline=(90, 100, 120), width=2)

    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    dob = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1960, 2004)}"
    doc_no = f"{rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"
    expiry = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2027, 2035)}"

    fields = [("NAME", name, 96), ("DATE OF BIRTH", dob, 148),
              ("DOCUMENT NO.", doc_no, 200), ("VALID UNTIL", expiry, 252)]
    for label, value, y in fields:
        d.text((186, y), label, font=_font(12), fill=(110, 118, 138))
        d.text((186, y + 16), value, font=_font(22), fill=(20, 24, 40))

    d.text((186, 306), "ISSUING AUTHORITY: OFFICE OF NOWHERE (FICTIONAL)",
           font=_font(11), fill=(120, 128, 148))

    meta = {"name": name, "dob": dob, "doc_no": doc_no, "expiry": expiry,
            "field_boxes": {f[0]: [186, f[2], 520, f[2] + 40] for f in fields}}
    return img, meta


def _simulate_capture(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make the specimen look like a PHOTO of a card, not a flat render.

    This is not decoration, it is what makes the test fair. Error Level
    Analysis reads how much high-frequency detail a region still has left to
    lose under recompression. A flat vector-style render has almost none, so
    ELA has nothing to work with and measures noise — on the first version of
    this corpus it separated the classes not at all, and the classifier's
    apparent skill came entirely from a leak. Real KYC submissions are phone
    photographs carrying sensor noise, slight blur and uneven illumination,
    which is the content ELA actually keys on. Every document gets the same
    treatment, so this adds realism, not a class giveaway.
    """
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    # Uneven illumination — a soft diagonal gradient, as from a room light.
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 1.0 - 0.10 * ((xx / w) * rng.uniform(0.4, 1.0)
                          + (yy / h) * rng.uniform(0.4, 1.0))
    arr *= shade[:, :, None]
    # Sensor noise.
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(2.2, 4.5), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))


def _pre_compress(img: Image.Image) -> Image.Image:
    """One generation of JPEG history, applied to EVERY document.

    This must happen for genuine and tampered documents alike. The first
    version of this script pre-compressed only the ones it went on to tamper,
    which gave every tampered file a second compression generation across the
    WHOLE image. A classifier then scored a perfect 1.00/1.00 by detecting
    global double-compression, while the ELA hotspot landed on the edited
    field in 0 of 43 cases — a textbook right-answer-wrong-reason leak, caught
    only by the localisation check. Both classes now share an identical
    compression pipeline, so the ONLY difference is the local splice.
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def tamper(img: Image.Image, rng: random.Random, meta: dict) -> tuple[Image.Image, dict]:
    """Edit one field the way a forger would: overwrite the value on an
    already-compressed card, so the edited region alone carries a fresher
    compression history than its surroundings."""
    field = rng.choice(["NAME", "DATE OF BIRTH", "DOCUMENT NO.", "VALID UNTIL"])
    box = meta["field_boxes"][field]

    d = ImageDraw.Draw(img)
    d.rectangle([box[0], box[1] + 14, box[2], box[3]], fill=(238, 240, 246))
    if field == "NAME":
        new = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    elif field == "DATE OF BIRTH":
        new = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1960,2004)}"
    elif field == "DOCUMENT NO.":
        new = f"{rng.randint(1000,9999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)}"
    else:
        new = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(2027,2035)}"
    d.text((box[0], box[1] + 16), new, font=_font(22), fill=(20, 24, 40))

    # A spliced region is pasted in clean: it lacks the sensor noise and the
    # compression history of the photograph around it. That contrast is the
    # artifact ELA is supposed to find.
    return img, {"tampered_field": field, "tampered_box": box, "new_value": new}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--tamper-fraction", type=float, default=0.5)
    args = ap.parse_args()

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("doc_*.jpg"):
        old.unlink()

    rows = []
    for i in range(args.n):
        img, meta = render_card(rng, i)
        # Identical capture + compression history for both classes.
        img = _pre_compress(_simulate_capture(img, rng))
        is_tampered = rng.random() < args.tamper_fraction
        extra: dict = {}
        if is_tampered:
            img, extra = tamper(img, rng, meta)
        # Genuine and tampered are saved at the SAME final quality, so overall
        # file quality is not itself the giveaway — only the internal
        # discontinuity is.
        path = OUT_DIR / f"doc_{i:04d}.jpg"
        img.save(path, "JPEG", quality=92)
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "doc_id": f"DOC{i:04d}",
            "is_tampered": int(is_tampered),
            "tampered_field": extra.get("tampered_field", ""),
            "tampered_box": json.dumps(extra.get("tampered_box", [])),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)
    print(f"Wrote {len(df)} specimen documents -> {OUT_DIR}")
    print(f"Manifest -> {MANIFEST}")
    print(df["is_tampered"].value_counts().rename(
        {0: "genuine", 1: "tampered"}).to_string())
    print("\nSYNTHETIC SPECIMENS. No real identity documents were used. The "
          "tampering\nis real image manipulation, so the ELA signal is real; "
          "the documents are not.")


if __name__ == "__main__":
    main()
