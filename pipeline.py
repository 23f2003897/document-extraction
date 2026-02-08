import logging
import json
import time
import os
import re
from pathlib import Path

from extract_pdf import VectorPDFParser
from run_slm import ExtractionSLM
from load_db import DocumentDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractionPipeline:

    def __init__(self, db_path="data/documents.db", ollama_url="http://localhost:11434"):

        self.db = DocumentDatabase(db_path)
        self.slm = None
        self.ollama_url = ollama_url

        logger.info("✅ Pipeline initialized")

    # ================= LOAD MODEL =================
    def _ensure_slm_loaded(self):

        if self.slm is None:
            logger.info("Loading SLM (Ollama Phi3)...")
            self.slm = ExtractionSLM(
                ollama_url=self.ollama_url,
                model="phi3"
            )

    # ================= CLEAN TEXT =================
    def _clean_text(self, text):
        text = re.sub(r'\b[A-Z](?:\s+[A-Z]){2,}\b', '', text)
        return text

    # ================= STRUCTURE DETECTION =================
    def _detect_structure(self, chunks, doc_type):

        full_text = "\n".join([self._clean_text(c.get("text","")) for c in chunks])

        if doc_type == "constitution":
            pattern = r'Article\s+\d+[A-Za-z0-9()\-]*'
        elif doc_type == "mathematics":
            pattern = r'(Example\s+\d+|Theorem\s+\d+)'
        elif doc_type == "utility":
            pattern = r'(Meter\s*ID[:\-]?\s*\w+|Pipeline\s*ID[:\-]?\s*\w+)'
        else:
            return chunks

        matches = list(dict.fromkeys(re.findall(pattern, full_text)))

        structured_chunks = []

        for m in matches:

            idx = full_text.find(m)
            context = full_text[max(0, idx-800):idx+1200]

            if any(x in context.lower() for x in ["contents","preface","abbreviations"]):
                continue

            structured_chunks.append({
                "text": context.strip(),
                "page_nums":[None]
            })

        logger.info(f"🔥 Structured chunks: {len(structured_chunks)}")

        return structured_chunks if structured_chunks else chunks

    # ================= HIERARCHY TRACKING =================
    def _update_hierarchy(self, text, state):

        part_match = re.search(r'(PART\s+[IVXLC]+)', text, re.IGNORECASE)
        if part_match:
            state["part"] = part_match.group(1).upper()

        chapter_match = re.search(r'(CHAPTER\s+[IVXLC]+)', text, re.IGNORECASE)
        if chapter_match:
            state["chapter"] = chapter_match.group(1).upper()

        return state

    def _apply_hierarchy(self, records):

        state = {"part": None, "chapter": None}
        updated = []

        for r in records:

            text = r.get("raw_text","")

            state = self._update_hierarchy(text, state)

            if not r.get("part"):
                r["part"] = state["part"]

            if not r.get("chapter"):
                r["chapter"] = state["chapter"]

            updated.append(r)

        return updated

    # ================= CONSTITUTION CLEANUP =================
    def _clean_results(self, records):

        cleaned = {}

        for r in records:

            art = r.get("article_number")

            if not art:
                continue

            if " and " in art.lower():
                continue

            raw = (r.get("raw_text") or "").lower()

            if any(x in raw for x in ["preface","abbreviations","contents","appendix"]):
                continue

            if art not in cleaned or len(raw) > len(cleaned[art]["raw_text"] or ""):
                cleaned[art] = r

        return list(cleaned.values())

    # ================= MATHEMATICS CLEANUP =================
    def _clean_math_results(self, records):

        cleaned = []

        for r in records:

            text = (r.get("raw_text") or "").lower()

            if not text or len(text) < 50:
                continue

            if any(x in text for x in [
                "second edition",
                "textbook of engineering mathematics",
                "copyright",
                "answers to objective type questions"
            ]):
                continue

            if re.search(r'\d+\.\d+\s+[a-z]', text):
                continue

            if re.search(r'\b[A-Z](?:\s+[A-Z]){3,}\b', r.get("raw_text","")):
                continue

            if "example" in text and not r.get("theorem_number"):
                r["theorem_number"] = "Example"

            cleaned.append(r)

        return cleaned

    # ================= UTILITY CLEANUP (NEW) =================
    def _clean_utility_results(self, records):

        cleaned = {}

        for r in records:

            entity = r.get("entity_id")
            if not entity:
                continue

            # Normalize numeric values
            value = r.get("value")
            try:
                if value is not None:
                    r["value"] = float(value)
            except:
                pass

            # Keep best version (prefer record with page_number)
            if entity not in cleaned:
                cleaned[entity] = r
            else:
                if cleaned[entity].get("page_number") is None and r.get("page_number") is not None:
                    cleaned[entity] = r

        return list(cleaned.values())

    # ================= MAIN PROCESS =================
    def process_pdf(self, pdf_path, doc_type=None, max_chunks=None, batch_size=3):

        parser = VectorPDFParser(pdf_path)

        if doc_type is None:
            doc_type = parser.detect_document_type()

        chunks = parser.chunk_by_document_type(doc_type)

        chunks = self._detect_structure(chunks, doc_type)

        if max_chunks:
            chunks = chunks[:max_chunks]

        parser.close()

        os.makedirs("data/processed", exist_ok=True)
        output_file = f"data/processed/{doc_type}_output.json"

        self._ensure_slm_loaded()

        all_records = []

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i+batch_size]
            extracted = self.slm.batch_extract(batch, doc_type)
            all_records.extend(extracted)

        # hierarchy tracking
        all_records = self._apply_hierarchy(all_records)

        # domain cleanup
        if doc_type == "constitution":
            all_records = self._clean_results(all_records)

        if doc_type == "mathematics":
            all_records = self._clean_math_results(all_records)

        if doc_type == "utility":
            all_records = self._clean_utility_results(all_records)

        with open(output_file,"w",encoding="utf-8") as f:
            json.dump(all_records,f,indent=2,ensure_ascii=False)

        self.db.bulk_insert(all_records, doc_type)
        self.db.bulk_insert_chunks(chunks, doc_type)

        logger.info("✅ PROCESSING COMPLETE")

        return all_records

    def close(self):
        self.db.close()


# ================= MAIN =================

def main():

    pipeline = ExtractionPipeline()

    pdfs = [
        ("data/raw_pdfs/constitution.pdf","constitution"),
        ("data/raw_pdfs/mathematics_m1.pdf","mathematics"),
        ("data/raw_pdfs/water_utility_report.pdf","utility"),
    ]

    for path, dtype in pdfs:
        pipeline.process_pdf(path, dtype, max_chunks=100)

    pipeline.close()


if __name__=="__main__":
    main()
