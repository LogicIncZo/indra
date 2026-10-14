"""Synthetic line-image generation for Indra Tamil OCR training.

Text pools:
  TAMIL — Tirukkural lines (public domain; train/data/kural.txt, 2652 lines,
          cleaned of unassigned codepoints — the raw dataset had 8 lines with
          U+0BA6/U+0BA7, <reserved> through Unicode 18.0).
  MULTI — English sentences (embedded; covers Latin base, digits, punctuation).
  HANZI — common-character sequences (embedded; script router needs the class,
          not recognition quality).

Rendering renders text at high resolution with random Noto fonts, then applies
scan-like degradations (blur, noise, JPEG, rotation, exposure) and finally
indra_ocr.rec_preproc.crop_line — the exact crop rule inference uses, so the
train/inference distributions match by construction.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"

TAMIL, MULTI, HANZI = "tamil", "multi", "hanzi"

ENGLISH = [
    "The report covers three years of transaction data across all districts.",
    "Section 4: Interchange fees and the Rs 5,000 crore incentive scheme.",
    "Zero-priced rails shift costs rather than remove them from the system.",
    "Figure 2 shows monthly volumes from April 2025 to March 2026.",
    "Payments infrastructure audit: method, sources, and limitations.",
    "The committee recommended that the threshold be reviewed annually.",
    "Table 3. Distribution of transaction values by merchant category.",
    "All figures are illustrative; see the monthly statistics for current data.",
    "Users do not pay directly, but the costs did not disappear anywhere.",
    "Advisory: verify any payment request against the official register.",
    "This page intentionally documents the exception handling rules.",
    "Banking outlets served 164,000 villages during the financial year.",
]

# ~200 most frequent Simplified Chinese characters; router-class data only.
HANZI_CHARS = (
    "的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会"
    "家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从"
    "无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两"
    "次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实"
    "向声车全信重三机工物气每并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表"
    "德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平"
    "报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象"
    "数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱吗语元喜曾离飞"
    "科言干流欢约各即指合反题必该论交终林请医晚制球决传画保读运及则房早院量苦火布品近坐"
    "产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留"
    "市半热送兴造谈容极随演收首根讲整式取照办强石古华拿计您装似足双妻尼转诉米称丽客南领"
    "节衣站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史谁准联妇纪基买志静阿诗独"
    "复痛消社算义竟确酒需单治卡幸兰念举仅钟怕共毛句息功官待究跟穿室易游程号居考突皮哪费"
    "倒价图具刚脑永歌响商礼细专黄块脚味灵改据般破引食仍存众注笔甚某沉血备习校默务土微娘"
    "试背料注志村妈爸爷奶哥姐弟妹"
)


def _font_pool(kind: str, rng: random.Random):
    from PIL import ImageFont

    if kind == TAMIL:
        paths = list(Path("/usr/share/fonts/truetype/noto").glob("Noto*Tamil*.ttf"))
    elif kind == MULTI:
        paths = list(Path("/usr/share/fonts/truetype/dejavu").glob("*.ttf"))
    else:
        paths = [DATA / "NotoSansCJKsc-Regular.otf"]
    if not paths:
        raise RuntimeError(f"no fonts for {kind}")
    return paths


def _augment(img, rng: random.Random):
    from PIL import Image, ImageFilter

    img = img.convert("L")
    if rng.random() < 0.6:  # gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.4)))
    if rng.random() < 0.4:  # slight rotation
        img = img.rotate(rng.uniform(-1.0, 1.0), expand=False, fillcolor=255, resample=3)
    arr = np.asarray(img, np.float32)
    arr = arr * rng.uniform(0.85, 1.05) + rng.gauss(0, rng.choice([2, 6, 12]))
    if rng.random() < 0.5:  # jpeg roundtrip
        q = rng.randrange(35, 90)
        buf = io.BytesIO()
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
        buf.seek(0)
        from PIL import Image as I

        arr = np.asarray(I.open(buf), np.float32)
    return arr.clip(0, 255)


def make_line(text: str, kind: str, rng: random.Random) -> np.ndarray:
    """Render one line; return an ink map [32, W] via the shared crop rule."""
    from PIL import Image, ImageDraw, ImageFont
    from indra_ocr.rec_preproc import crop_line

    pool = _font_pool(kind, rng)
    fpath = rng.choice(pool)
    # Big print scale: ~2 CTC frames per glyph after the 32px downscale.
    size = rng.randrange(38, 54)  # ~10-13pt text at 200 dpi page scale
    font = ImageFont.truetype(str(fpath), size)
    pad = size
    width = int(font.getlength(text)) + pad
    height = int(size * 1.35)
    img = Image.new("L", (max(width, 64), height), 255)
    d = ImageDraw.Draw(img)
    d.text((pad // 2, 0), text, font=font, fill=0, direction="ltr", language="ta" if kind == TAMIL else None)
    arr = _augment(img, rng).astype(np.uint8)
    return crop_line(arr, (0, 0, arr.shape[1], arr.shape[0]))


def load_texts() -> dict:
    kural = (DATA / "kural.txt").read_text().splitlines()
    kural += (DATA / "coverage_random.txt").read_text().splitlines()
    kural += (DATA / "coverage_grantha.txt").read_text().splitlines()
    kural = [ln.strip() for ln in kural if 12 <= len(ln.strip()) <= 120]
    hanzi = list(HANZI_CHARS)
    htexts = []
    rng = random.Random(2)
    for _ in range(400):
        n = rng.randrange(6, 22)
        htexts.append("".join(rng.choice(hanzi) for _ in range(n)))
    return {TAMIL: kural, MULTI: ENGLISH * 8, HANZI: htexts}
