import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# dependency checks
try:
    import fitz
except ImportError:
    print("Missing dependency: PyMuPDF")
    print("Install with: pip install PyMuPDF")
    sys.exit(1)

try:
    import pdfplumber
except ImportError:
    print("Missing dependency: pdfplumber")
    print("Install with: pip install pdfplumber")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing dependency: sentence-transformers")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Missing dependency: pytesseract or pillow")
    print("Install with: pip install pytesseract pillow")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    bbox: Tuple[float, float, float, float]
    page_num: int
    element_type: str


class PDFTextExtractor:
    # main extractor class for text + OCR fallback

    def _ocr_page(self, page) -> List[TextElement]:
        # fallback OCR for scanned pages
        ocr_elements = []

        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            ocr_text = pytesseract.image_to_string(img)
            lines = ocr_text.split("\n")

            for line in lines:
                text = line.strip()
                if text:
                    ocr_elements.append(
                        TextElement(
                            text=text,
                            font_name="OCR",
                            font_size=12,
                            is_bold=False,
                            is_italic=False,
                            bbox=(0, 0, 0, 0),
                            page_num=page.number + 1,
                            element_type="normal"
                        )
                    )

            logger.info(f"OCR extracted {len(ocr_elements)} lines from page {page.number+1}")

        except Exception as e:
            logger.error(f"OCR failed on page {page.number+1}: {e}")

        return ocr_elements

    def __init__(self, pdf_path: str):
        # open PDF
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.elements: List[TextElement] = []
        self.total_pages = len(self.doc)
        logger.info(f"Opened PDF: {pdf_path} ({self.total_pages} pages)")

    def extract_with_formatting(self) -> List[TextElement]:
        # extract text spans with font information
        logger.info("Extracting text with formatting...")

        for page_num in range(self.total_pages):
            page = self.doc[page_num]

            text = page.get_text()
            if len(text.strip()) < 10:
                logger.warning(f"Page {page_num + 1} has minimal text (possibly scanned)")
                logger.info("OCR triggered")
                ocr_elements = self._ocr_page(page)
                self.elements.extend(ocr_elements)
                continue

            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                element = TextElement(
                                    text=span["text"],
                                    font_name=span["font"],
                                    font_size=span["size"],
                                    is_bold="bold" in span["font"].lower(),
                                    is_italic="italic" in span["font"].lower(),
                                    bbox=tuple(span["bbox"]),
                                    page_num=page_num + 1,
                                    element_type=self._classify_element(span)
                                )
                                self.elements.append(element)

        logger.info(f"Extracted {len(self.elements)} text elements")
        return self.elements

    def _classify_element(self, span: dict) -> str:
        # simple heuristic heading detection
        font_size = span["size"]
        font_name = span["font"].lower()

        if font_size > 14 or "bold" in font_name or "heading" in font_name:
            return "heading"
        return "normal"

    def extract_tables(self) -> List[Dict]:
        # table extraction using pdfplumber
        tables_data = []

        try:
            logger.info("Extracting tables...")
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()

                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            tables_data.append({
                                'page': page_num,
                                'table_index': table_idx,
                                'data': table,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0
                            })

            logger.info(f"Extracted {len(tables_data)} tables")
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")

        return tables_data

    def get_hierarchical_structure(self) -> Dict:
        # build simple heading-based structure
        structure = {
            'title': '',
            'sections': []
        }

        if not self.elements:
            return structure

        current_section = None
        font_sizes = [e.font_size for e in self.elements]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

        for element in self.elements:
            if element.font_size > avg_font_size * 1.5 and not structure['title']:
                structure['title'] = element.text.strip()

            elif element.element_type == "heading" or element.font_size > avg_font_size * 1.2:
                current_section = {
                    'heading': element.text.strip(),
                    'page': element.page_num,
                    'font_size': element.font_size,
                    'content': []
                }
                structure['sections'].append(current_section)

            elif current_section is not None:
                current_section['content'].append({
                    'text': element.text,
                    'font': element.font_name,
                    'size': element.font_size
                })

        return structure

    def export_to_markdown(self, output_file: str):
        # export structured content to markdown
        structure = self.get_hierarchical_structure()

        with open(output_file, 'w', encoding='utf-8') as f:
            if structure['title']:
                f.write(f"# {structure['title']}\n\n")

            for section in structure['sections']:
                f.write(f"## {section['heading']}\n")
                f.write(f"*Page {section['page']}*\n\n")

                for content in section['content']:
                    text = content['text'].strip()
                    if text:
                        f.write(f"{text} ")
                f.write("\n\n")

        logger.info(f"Markdown exported to {output_file}")

    def export_detailed_json(self, output_file: str):
        # export full extraction to JSON
        data = {
            'metadata': {
                'file': self.pdf_path,
                'pages': self.total_pages,
                'elements_count': len(self.elements)
            },
            'text_elements': [
                {
                    'text': e.text,
                    'font_name': e.font_name,
                    'font_size': e.font_size,
                    'is_bold': e.is_bold,
                    'is_italic': e.is_italic,
                    'bbox': e.bbox,
                    'page': e.page_num,
                    'type': e.element_type
                }
                for e in self.elements
            ],
            'tables': self.extract_tables(),
            'structure': self.get_hierarchical_structure()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON exported to {output_file}")

    def print_summary(self):
        # print quick summary in console
        structure = self.get_hierarchical_structure()
        tables = self.extract_tables()

        print("=" * 60)
        print("PDF EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Document: {self.pdf_path}")
        print(f"Total text elements: {len(self.elements)}")
        print(f"Tables found: {len(tables)}")
        print(f"Total pages: {self.total_pages}")

        if structure['title']:
            print(f"\nDocument Title: {structure['title']}")

        print(f"\nSections found: {len(structure['sections'])}")

        if structure['sections']:
            print("\n=== First 5 Headings ===")
            for i, section in enumerate(structure['sections'][:5], 1):
                print(f"  {i}. {section['heading']} (Page {section['page']})")

        if tables:
            print("\n=== Tables ===")
            for table in tables[:3]:
                print(f"  Page {table['page']}: {table['rows']} rows × {table['columns']} columns")

        print("=" * 60)

    def close(self):
        # close PDF file
        self.doc.close()


class VectorPDFParser:
    # high-level wrapper for extraction + embeddings

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.elements = self._process_pdf(pdf_path)

        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")

    def _process_pdf(self, pdf_path: str) -> List[TextElement]:
        # run extraction pipeline
        extractor = PDFTextExtractor(pdf_path)
        elements = extractor.extract_with_formatting()
        extractor.close()
        return elements

    def detect_document_type(self) -> str:
        # simple rule-based doc type detection
        sample_text = " ".join(e.text for e in self.elements[:20]).lower()

        if any(keyword in sample_text for keyword in ["article", "constitution", "fundamental rights", "part iii", "part i"]):
            logger.info("Detected: Constitution document")
            return "constitution"

        elif any(keyword in sample_text for keyword in ["theorem", "proof", "lemma", "corollary", "calculus", "mathematics"]):
            logger.info("Detected: Mathematics document")
            return "mathematics"

        elif any(keyword in sample_text for keyword in ["meter", "consumption", "bill", "kwh", "kilolitre", "zone", "tariff"]):
            logger.info("Detected: Utility document")
            return "utility"

        else:
            logger.warning("Could not detect document type, defaulting to 'utility'")
            return "utility"

    def chunk_by_document_type(self, doc_type: str) -> List[Dict]:
        # route to appropriate chunking logic
        logger.info(f"Creating chunks for {doc_type} document...")

        if doc_type == "constitution":
            return self._chunk_constitution()
        elif doc_type == "mathematics":
            return self._chunk_mathematics()
        elif doc_type == "utility":
            return self._chunk_utility()
        else:
            return self._chunk_by_page()

    def _chunk_constitution(self) -> List[Dict]:
        # article-based chunking
        chunks = []
        current_chunk = []
        current_page = None
        chunk_id = 0

        article_pattern = re.compile(r'\b(Article|Art\.?)\s+\d+', re.IGNORECASE)

        for elem in self.elements:
            text = elem.text.strip()

            if article_pattern.search(text) and current_chunk:
                merged_text = " ".join(current_chunk).strip()
                if len(merged_text) > 20:
                    embedding = self.embedding_model.encode(merged_text, show_progress_bar=False)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": merged_text,
                        "page_nums": [current_page],
                        "embedding": embedding.tolist()
                    })
                    chunk_id += 1
                current_chunk = []

            current_page = elem.page_num
            if text:
                current_chunk.append(text)

        if current_chunk:
            merged_text = " ".join(current_chunk).strip()
            if len(merged_text) > 20:
                embedding = self.embedding_model.encode(merged_text, show_progress_bar=False)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": merged_text,
                    "page_nums": [current_page],
                    "embedding": embedding.tolist()
                })

        logger.info(f"Created {len(chunks)} article-based chunks")
        return chunks

    def _chunk_mathematics(self) -> List[Dict]:
        # theorem/section based chunking
        chunks = []
        current_chunk = []
        current_page = None
        chunk_id = 0

        boundary_pattern = re.compile(
            r'\b(Theorem|Lemma|Corollary|Definition|Example|Section|Chapter)\s+[\d.]+',
            re.IGNORECASE
        )

        for elem in self.elements:
            text = elem.text.strip()

            if boundary_pattern.search(text) and current_chunk and len(" ".join(current_chunk)) > 100:
                merged_text = " ".join(current_chunk).strip()
                embedding = self.embedding_model.encode(merged_text, show_progress_bar=False)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": merged_text,
                    "page_nums": [current_page],
                    "embedding": embedding.tolist()
                })
                chunk_id += 1
                current_chunk = []

            current_page = elem.page_num
            if text:
                current_chunk.append(text)

        if current_chunk:
            merged_text = " ".join(current_chunk).strip()
            if len(merged_text) > 20:
                embedding = self.embedding_model.encode(merged_text, show_progress_bar=False)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": merged_text,
                    "page_nums": [current_page],
                    "embedding": embedding.tolist()
                })

        logger.info(f"Created {len(chunks)} theorem-based chunks")
        return chunks

    def _chunk_utility(self) -> List[Dict]:
        # fallback utility chunking
        return self._chunk_by_page()

    def _chunk_by_page(self) -> List[Dict]:
        # default page-based chunking
        chunks = []
        page_dict = defaultdict(list)

        for elem in self.elements:
            page_dict[elem.page_num].append(elem.text)

        chunk_id = 0
        for page_num in sorted(page_dict.keys()):
            merged_text = " ".join(page_dict[page_num]).strip()

            if len(merged_text) > 20:
                embedding = self.embedding_model.encode(merged_text, show_progress_bar=False)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": merged_text,
                    "page_nums": [page_num],
                    "embedding": embedding.tolist()
                })
                chunk_id += 1

        logger.info(f"Created {len(chunks)} page-based chunks")
        return chunks

    def close(self):
        # placeholder cleanup
        pass


def process_pdf(pdf_path: str, output_prefix: str = "output") -> List[TextElement]:
    # full pipeline runner
    try:
        logger.info(f"Processing: {pdf_path}")

        extractor = PDFTextExtractor(pdf_path)
        text_elements = extractor.extract_with_formatting()

        extractor.print_summary()
        extractor.export_to_markdown(f"{output_prefix}.md")
        extractor.export_detailed_json(f"{output_prefix}.json")

        extractor.close()

        logger.info("Processing complete!")
        return text_elements

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_parser():
    # simple test runner
    print("="*60)
    print("TESTING PDF PARSER")
    print("="*60)

    test_pdf = "data/raw_pdfs/constitution.pdf"

    if not os.path.exists(test_pdf):
        print(f"Test PDF not found: {test_pdf}")
        print("Please provide a valid PDF path")
        return

    parser = VectorPDFParser(test_pdf)
    doc_type = parser.detect_document_type()
    chunks = parser.chunk_by_document_type(doc_type)

    with open("chunks_debug.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDocument Type: {doc_type}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(chunks[0]["text"][:200])
    print(f"\nEmbedding dimension: {len(chunks[0]['embedding'])}")

    parser.close()
    print("\nTest complete")


if __name__ == "__main__":
    import os
    test_parser()