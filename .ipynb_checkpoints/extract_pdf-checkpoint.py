

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Check dependencies
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ Missing dependency: PyMuPDF")
    print("Install with: pip install PyMuPDF")
    sys.exit(1)

try:
    import pdfplumber
except ImportError:
    print("❌ Missing dependency: pdfplumber")
    print("Install with: pip install pdfplumber")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Missing dependency: sentence-transformers")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("❌ Missing dependency: pytesseract or pillow")
    print("Install with: pip install pytesseract pillow")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """Represents a text element with its properties"""
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    element_type: str  # 'heading', 'normal', 'table'


class PDFTextExtractor:
    """
    Extracts text and structure from PDF documents
    Handles both text-based and scanned PDFs

    """
    def _ocr_page(self, page) -> List[TextElement]:
        """
        OCR fallback for scanned pages using Tesseract
        Converts PDF page to image and extracts text
        """

        ocr_elements = []

        try:
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # OCR
            ocr_text = pytesseract.image_to_string(img)

            # Convert OCR output into TextElements
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
                            bbox=(0,0,0,0),
                            page_num=page.number + 1,
                            element_type="normal"
                        )
                    )

            logger.info(f"OCR extracted {len(ocr_elements)} lines from page {page.number+1}")

        except Exception as e:
            logger.error(f"OCR failed on page {page.number+1}: {e}")

        return ocr_elements
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.elements: List[TextElement] = []
        self.total_pages = len(self.doc)
        logger.info(f"📄 Opened PDF: {pdf_path} ({self.total_pages} pages)")
    
    def extract_with_formatting(self) -> List[TextElement]:
        """
        Extract text with font information and structure
        
        Returns:
            List of TextElement objects
        """
        logger.info("Extracting text with formatting...")
        
        for page_num in range(self.total_pages):
            page = self.doc[page_num]
            
            # Check if page has text
            text = page.get_text()
            if len(text.strip()) < 10:
                logger.warning(f"Page {page_num + 1} has minimal text (possibly scanned)")
                logger.info("🔥 OCR TRIGGERED")
                ocr_elements = self._ocr_page(page)
                self.elements.extend(ocr_elements)
                continue
            
            # Extract text blocks with font info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():  # Only non-empty text
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
        
        logger.info(f"✅ Extracted {len(self.elements)} text elements")
        return self.elements
    
    def _classify_element(self, span: dict) -> str:
        """
        Classify text element as heading or normal based on font properties
        
        Args:
            span: Text span dictionary from PyMuPDF
            
        Returns:
            Element type: 'heading' or 'normal'
        """
        font_size = span["size"]
        font_name = span["font"].lower()
        
        # Heuristics for heading detection
        if font_size > 14 or "bold" in font_name or "heading" in font_name:
            return "heading"
        return "normal"
    
    def extract_tables(self) -> List[Dict]:
        """
        Extract tables from PDF using pdfplumber
        
        Returns:
            List of table data dictionaries
        """
        tables_data = []
        
        try:
            logger.info("Extracting tables...")
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # At least header + 1 row
                            tables_data.append({
                                'page': page_num,
                                'table_index': table_idx,
                                'data': table,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0
                            })
            
            logger.info(f"✅ Extracted {len(tables_data)} tables")
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables_data
    
    def get_hierarchical_structure(self) -> Dict:
        """
        Organize content hierarchically based on headings
        
        Returns:
            Dictionary with document structure
        """
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
            # Detect main title (largest font)
            if element.font_size > avg_font_size * 1.5 and not structure['title']:
                structure['title'] = element.text.strip()
            
            # Detect section headings
            elif element.element_type == "heading" or element.font_size > avg_font_size * 1.2:
                current_section = {
                    'heading': element.text.strip(),
                    'page': element.page_num,
                    'font_size': element.font_size,
                    'content': []
                }
                structure['sections'].append(current_section)
            
            # Regular content
            elif current_section is not None:
                current_section['content'].append({
                    'text': element.text,
                    'font': element.font_name,
                    'size': element.font_size
                })
        
        return structure
    
    def export_to_markdown(self, output_file: str):
        """Export extracted content to markdown format"""
        structure = self.get_hierarchical_structure()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write title
            if structure['title']:
                f.write(f"# {structure['title']}\n\n")
            
            # Write sections
            for section in structure['sections']:
                f.write(f"## {section['heading']}\n")
                f.write(f"*Page {section['page']}*\n\n")
                
                for content in section['content']:
                    text = content['text'].strip()
                    if text:
                        f.write(f"{text} ")
                f.write("\n\n")
        
        logger.info(f"✅ Markdown exported to {output_file}")
    
    def export_detailed_json(self, output_file: str):
        """Export complete extraction to JSON"""
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
        
        logger.info(f"✅ JSON exported to {output_file}")
    
    def print_summary(self):
        """Print a summary of extracted content"""
        structure = self.get_hierarchical_structure()
        tables = self.extract_tables()
        
        print("=" * 60)
        print("PDF EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"📄 Document: {self.pdf_path}")
        print(f"📝 Total text elements: {len(self.elements)}")
        print(f"📊 Tables found: {len(tables)}")
        print(f"📑 Total pages: {self.total_pages}")
        
        if structure['title']:
            print(f"\n📌 Document Title: {structure['title']}")
        
        print(f"\n📖 Sections found: {len(structure['sections'])}")
        
        if structure['sections']:
            print("\n=== First 5 Headings ===")
            for i, section in enumerate(structure['sections'][:5], 1):
                print(f"  {i}. {section['heading']} (Page {section['page']})")
        
        if tables:
            print("\n=== Tables ===")
            for table in tables[:3]:
                print(f"  📊 Page {table['page']}: {table['rows']} rows × {table['columns']} columns")
        
        print("=" * 60)
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()


class VectorPDFParser:
    """
    High-level PDF parser with vector embeddings
    Integrates text extraction and chunking strategies
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.elements = self._process_pdf(pdf_path)
        
        # Load embedding model
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Embedding model loaded")
    
    def _process_pdf(self, pdf_path: str) -> List[TextElement]:
        """
        Process PDF and extract text elements
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of TextElement objects
        """
        extractor = PDFTextExtractor(pdf_path)
        elements = extractor.extract_with_formatting()
        extractor.close()
        return elements
    
    def detect_document_type(self) -> str:
        """
        Auto-detect document type based on content
        
        Returns:
            Document type: 'constitution', 'mathematics', or 'utility'
        """
        # Sample first 10 elements
        sample_text = " ".join(e.text for e in self.elements[:20]).lower()
        
        # Constitution indicators
        if any(keyword in sample_text for keyword in ["article", "constitution", "fundamental rights", "part iii", "part i"]):
            logger.info("🔍 Detected: Constitution document")
            return "constitution"
        
        # Mathematics indicators
        elif any(keyword in sample_text for keyword in ["theorem", "proof", "lemma", "corollary", "calculus", "mathematics"]):
            logger.info("🔍 Detected: Mathematics document")
            return "mathematics"
        
        # Utility indicators
        elif any(keyword in sample_text for keyword in ["meter", "consumption", "bill", "kwh", "kilolitre", "zone", "tariff"]):
            logger.info("🔍 Detected: Utility document")
            return "utility"
        
        else:
            logger.warning("⚠️ Could not detect document type, defaulting to 'utility'")
            return "utility"
    
    def chunk_by_document_type(self, doc_type: str) -> List[Dict]:
        """
        Create intelligent chunks based on document type
        
        Args:
            doc_type: Type of document
            
        Returns:
            List of chunk dictionaries with embeddings
        """
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
        """
        Chunk constitution documents by articles
        Each article is a separate chunk
        """
        chunks = []
        current_chunk = []
        current_page = None
        chunk_id = 0
        
        # Pattern to detect article boundaries
        article_pattern = re.compile(r'\b(Article|Art\.?)\s+\d+', re.IGNORECASE)
        
        for elem in self.elements:
            text = elem.text.strip()
            
            # Check if this starts a new article
            if article_pattern.search(text) and current_chunk:
                # Save previous chunk
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
        
        # Add final chunk
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
        
        logger.info(f"✅ Created {len(chunks)} article-based chunks")
        return chunks
    
    def _chunk_mathematics(self) -> List[Dict]:
        """
        Chunk mathematics documents by theorems/sections
        """
        chunks = []
        current_chunk = []
        current_page = None
        chunk_id = 0
        
        # Pattern to detect theorem/section boundaries
        boundary_pattern = re.compile(
            r'\b(Theorem|Lemma|Corollary|Definition|Example|Section|Chapter)\s+[\d.]+',
            re.IGNORECASE
        )
        
        for elem in self.elements:
            text = elem.text.strip()
            
            # Check if this starts a new theorem/section
            if boundary_pattern.search(text) and current_chunk and len(" ".join(current_chunk)) > 100:
                # Save previous chunk
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
        
        # Add final chunk
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
        
        logger.info(f"✅ Created {len(chunks)} theorem-based chunks")
        return chunks
    
    def _chunk_utility(self) -> List[Dict]:
        """
        Chunk utility documents by logical sections (meters, zones, etc.)
        Falls back to page-based chunking
        """
        return self._chunk_by_page()
    
    def _chunk_by_page(self) -> List[Dict]:
        """
        Default chunking strategy: one chunk per page
        Merges all text on each page
        """
        chunks = []
        page_dict = defaultdict(list)
        
        # Group elements by page
        for elem in self.elements:
            page_dict[elem.page_num].append(elem.text)
        
        # Create chunks
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
        
        logger.info(f"✅ Created {len(chunks)} page-based chunks")
        return chunks
    
    def close(self):
        """Cleanup resources"""
        pass


# ==================== MAIN FUNCTION ====================
def process_pdf(pdf_path: str, output_prefix: str = "output") -> List[TextElement]:
    """
    Process a PDF and export results
    
    Args:
        pdf_path: Path to PDF file
        output_prefix: Prefix for output files
        
    Returns:
        List of extracted TextElement objects
    """
    try:
        logger.info(f"🔄 Processing: {pdf_path}")
        
        extractor = PDFTextExtractor(pdf_path)
        text_elements = extractor.extract_with_formatting()
        
        extractor.print_summary()
        extractor.export_to_markdown(f"{output_prefix}.md")
        extractor.export_detailed_json(f"{output_prefix}.json")
        
        extractor.close()
        
        logger.info("✅ Processing complete!")
        return text_elements
        
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==================== TESTING ====================
def test_parser():
    """Test the PDF parser"""
    print("="*60)
    print("TESTING PDF PARSER")
    print("="*60)
    
    # Create a test scenario
    test_pdf = "data/raw_pdfs/constitution.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"⚠️ Test PDF not found: {test_pdf}")
        print("Please provide a valid PDF path")
        return
    
    parser = VectorPDFParser(test_pdf)
    doc_type = parser.detect_document_type()
    chunks = parser.chunk_by_document_type(doc_type)
    
    print(f"\nDocument Type: {doc_type}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(chunks[0]["text"][:200])
    print(f"\nEmbedding dimension: {len(chunks[0]['embedding'])}")
    
    parser.close()
    print("\n✅ Test complete")


if __name__ == "__main__":
    import os
    test_parser()