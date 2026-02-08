import re
import json
import bisect

# -------------------------------------------------------------------
# 1. LOAD RAW TEXT
# -------------------------------------------------------------------
# Adjust this path if your file is elsewhere
with open("data/processed/constitution_chunks.json_pagewise.md", "r", encoding="utf-8") as f:
    full_text = f.read()

# -------------------------------------------------------------------
# 2. PATTERNS FOR PART, CHAPTER, AND ARTICLE HEADERS
# -------------------------------------------------------------------

# PART pattern examples: "PART III", "PART V", "PART IVA", "PART XIVA"
PART_RE = re.compile(
    r'\bPART\s+([IVXLC]+[A-Z]?)\b.*',
    re.IGNORECASE
)

# CHAPTER pattern examples: "CHAPTER I", "CHAPTER I.—THE EXECUTIVE"
CHAPTER_RE = re.compile(
    r'\bCHAPTER\s+([IVXLC]+)\b.*',
    re.IGNORECASE
)

# Article pattern:
# ^<digits><optional capital letter>. <title...>
# e.g. "1. Name and territory of the Union.—(1) ...", "21A. Right to education. ..."
ARTICLE_RE = re.compile(
    r'^(?P<num>\d+[A-Z]?)\.\s+(?P<title>.+)$',
    re.MULTILINE
)

# Titles that indicate repealed/omitted
REPEALED_KEYWORDS = [
    "—Omitted", "—Omitted.", "Omitted by", "omitted by",
    "repealed by", "Repealed by"
]

# -------------------------------------------------------------------
# 3. ARTICLE → PART MAPPING (CANONICAL, ROBUST)
# -------------------------------------------------------------------
def article_to_part(art_num_str: str):
    """Map article_number like '14' or '21A' to official PART."""
    m = re.match(r'(\d+)', art_num_str)
    if not m:
        return None
    n = int(m.group(1))

    # PART I: 1–4
    if 1 <= n <= 4:
        return "PART I"

    # PART II: 5–11
    if 5 <= n <= 11:
        return "PART II"

    # PART III: 12–35
    if 12 <= n <= 35:
        return "PART III"

    # PART IV: 36–51
    if 36 <= n <= 51:
        return "PART IV"

    # PART IVA: 51A
    if art_num_str == "51A":
        return "PART IVA"

    # PART V: 52–151
    if 52 <= n <= 151:
        return "PART V"

    # PART VI: 152–237
    if 152 <= n <= 237:
        return "PART VI"

    # PART VII – omitted (no current articles)

    # PART VIII: 239–241
    if 239 <= n <= 241:
        return "PART VIII"

    # PART IX: 243–243O (but not 243P onwards)
    if art_num_str.startswith("243") and not art_num_str.startswith("243P"):
        return "PART IX"

    # PART IXA: 243P–243ZG
    if art_num_str.startswith(("243P", "243Q", "243R", "243S",
                               "243T", "243U", "243V", "243W",
                               "243X", "243Y", "243Z")):
        # 243P, 243Q, ..., 243ZG
        return "PART IXA"

    # PART IXB: 243ZH–243ZT
    if art_num_str.startswith(("243ZH", "243ZI", "243ZJ", "243ZK",
                               "243ZL", "243ZM", "243ZN", "243ZO",
                               "243ZP", "243ZQ", "243ZR", "243ZS",
                               "243ZT")):
        return "PART IXB"

    # PART X: 244–244A
    if n == 244 or art_num_str == "244A":
        return "PART X"

    # PART XI: 245–263
    if 245 <= n <= 263:
        return "PART XI"

    # PART XII: 264–300A
    if 264 <= n <= 300 or art_num_str == "300A":
        return "PART XII"

    # PART XIII: 301–307
    if 301 <= n <= 307:
        return "PART XIII"

    # PART XIV: 308–323
    if 308 <= n <= 323:
        return "PART XIV"

    # PART XIVA: 323A–323B
    if art_num_str in ("323A", "323B"):
        return "PART XIVA"

    # PART XV: 324–329A
    if 324 <= n <= 329 or art_num_str == "329A":
        return "PART XV"

    # PART XVI: 330–342A
    if 330 <= n <= 342 or art_num_str == "342A":
        return "PART XVI"

    # PART XVII: 343–351
    if 343 <= n <= 351:
        return "PART XVII"

    # PART XVIII: 352–360
    if 352 <= n <= 360:
        return "PART XVIII"

    # PART XIX: 361–367
    if 361 <= n <= 367:
        return "PART XIX"

    # PART XX: 368
    if n == 368:
        return "PART XX"

    # PART XXI: 369–392
    if 369 <= n <= 392:
        return "PART XXI"

    # PART XXII: 393–395
    if 393 <= n <= 395:
        return "PART XXII"

    return None

# -------------------------------------------------------------------
# 4. PRE-SCAN: BUILD CHAPTER INDEX OVER THE WHOLE TEXT
# (Part headings are no longer used for part; only for information if needed.)
# -------------------------------------------------------------------
def build_chapter_index(text):
    chapter_positions = []
    for m in CHAPTER_RE.finditer(text):
        chap_id = m.group(1).upper()
        chapter_positions.append((m.start(), f"CHAPTER {chap_id}"))

    chapter_positions.sort(key=lambda x: x[0])
    idx_list = [c[0] for c in chapter_positions]
    labels = [c[1] for c in chapter_positions]
    return idx_list, labels

chap_idx, chap_labels = build_chapter_index(full_text)

def last_label_before(pos, idx_list, label_list):
    """Return the label whose index is <= pos (or None)."""
    i = bisect.bisect_right(idx_list, pos) - 1
    if i >= 0:
        return label_list[i]
    return None

# -------------------------------------------------------------------
# 5. MAIN EXTRACTION
# -------------------------------------------------------------------
def extract_articles_from_text(text):
    articles = []
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return []

    for idx, m in enumerate(matches):
        art_start = m.start()
        art_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[art_start:art_end]

        art_number = m.group("num")
        title_line = m.group("title").strip()

        # part: from canonical mapping, not from headings
        current_part = article_to_part(art_number)

        # chapter: optional; comment the next line and set None if you want to drop chapters
        current_chapter = last_label_before(art_start, chap_idx, chap_labels)
        # Or to disable chapter:
        # current_chapter = None

        status = "Active"
        for kw in REPEALED_KEYWORDS:
            if kw in title_line:
                status = "Repealed"
                break

        raw_text = block.strip("\n")

        article_obj = {
            "article_number": art_number,
            "article_title": title_line,
            "part": current_part,
            "chapter": current_chapter,
            "status": status,
            "raw_text": raw_text
        }
        articles.append(article_obj)

    return articles

# -------------------------------------------------------------------
# 6. RUN AND WRITE OUTPUT
# -------------------------------------------------------------------
articles = extract_articles_from_text(full_text)

with open("data/processed/constitution_structured_fixed.json", "w", encoding="utf-8") as out:
    json.dump(articles, out, ensure_ascii=False, indent=2)

print(len(articles))
# Quick sanity checks:
for probe in ["40", "14", "243A"]:
    found = next((a for a in articles if a["article_number"] == probe), None)
    if found:
        print(probe, found["part"], found["chapter"])
print(f"Extracted {len(articles)} articles.")
