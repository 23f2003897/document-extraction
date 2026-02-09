import json
import logging
import re
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionPrompt:
    # holds prompt config for each document type
    system_prompt: str
    user_template: str
    expected_fields: List[str]
    example_output: Dict[str, str]


class OllamaClient:
    # client wrapper for ollama api calls
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3"):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self._verify_connection()
    
    def _verify_connection(self):
        # check ollama server + model availability
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
                logger.warning(f"Run: ollama pull {self.model}")
            else:
                logger.info(f"Connected to Ollama - Model: {self.model}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            logger.error(f"Error: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise ConnectionError("Ollama service not available")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # generate response from model
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 512,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Generation failed: {e}")
            return ""


class ExtractionSLM:
    # main structured extraction logic using ollama
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "phi3"):
        logger.info("Initializing Extraction SLM with Ollama Phi3...")
        
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.prompts = self._initialize_prompts()
        
        logger.info("Extraction SLM ready!")
    
    def _initialize_prompts(self) -> Dict[str, ExtractionPrompt]:
        # initialize prompts for all document types
        
        constitution_prompt = ExtractionPrompt(
            system_prompt="""You are an expert at extracting structured information from Indian Constitution documents.
Your task is to identify articles, parts, and chapters with high precision.
Always output valid JSON with exact field names as specified.""",
            
            user_template="""Extract information from this Indian Constitution text.

TEXT:
{text}

INSTRUCTIONS:

- article_number = text matching "Article X"
- article_title = text immediately after article number
- part = any line like "PART III"
- chapter = any line like "CHAPTER I"
- if missing return null
# 1. Find the Article number (e.g., "Article 14", "Article 21", "Art. 1")
# 2. Extract the article title/heading
# 3. Identify the PART (e.g., "PART III", "PART I") - may be mentioned earlier in document
# 4. Identify the CHAPTER if mentioned (e.g., "CHAPTER I", "CHAPTER II")
# 5. If information is not present, use null

OUTPUT FORMAT (valid JSON only):
{{
    "article_number": "Article X or null",
    "article_title": "Title text or null",
    "part": "PART X or null",
    "chapter": "CHAPTER X or null"
}}

Return ONLY the JSON object, no other text.""",
            
            expected_fields=["article_number", "article_title", "part", "chapter"],
            example_output={
                "article_number": "Article 14",
                "article_title": "Equality before law",
                "part": "PART III",
                "chapter": "CHAPTER I"
            }
        )
        
        mathematics_prompt = ExtractionPrompt(
            system_prompt="""You are an expert at extracting structured information from engineering mathematics textbooks.
Your task is to identify chapters, sections, theorems, and their properties.
Always output valid JSON with exact field names as specified.""",
            
            user_template="""Extract information from this engineering mathematics textbook text.

TEXT:
{text}

INSTRUCTIONS:
1. Find the Chapter name/heading (e.g., "Chapter 3: Differential Calculus", "Unit II")
2. Find the Section name if present (e.g., "3.1 Limits", "Section A")
3. Find Theorem number (e.g., "Theorem 3.1", "Theorem 2.5.1", "Thm 1")
4. Find Theorem title/name (e.g., "Squeeze Theorem", "Mean Value Theorem") is usually inside parentheses after theorem number.
5. If information is not present, use null

OUTPUT FORMAT (valid JSON only):
{{
    "chapter_name": "Chapter heading or null",
    "section_name": "Section heading or null",
    "theorem_number": "Theorem identifier or null",
    "theorem_title": "Theorem name or null"
}}

Return ONLY the JSON object, no other text.""",
            
            expected_fields=["chapter_name", "section_name", "theorem_number", "theorem_title"],
            example_output={
                "chapter_name": "Chapter 3: Differential Calculus",
                "section_name": "3.1 Limits and Continuity",
                "theorem_number": "Theorem 3.1",
                "theorem_title": "Squeeze Theorem"
            }
        )
        
        utility_prompt = ExtractionPrompt(
            system_prompt="""You are an expert at extracting structured information from utility documents (water, electricity, logistics bills/reports).
Your task is to identify entities, locations, dates, and consumption values.
Always output valid JSON with exact field names as specified.""",
            
            user_template="""Extract information from this utility document text (water/electricity/logistics).

TEXT:
{text}

INSTRUCTIONS:
1. Find entity_id: Meter ID, Consumer ID, Account Number, Zone ID, Vehicle ID, etc
2. Find location: Address, Zone, Area, Region
3. Find date: Bill date, Reading date, Report date (format as YYYY-MM-DD if possible)
4. Find value: Consumption value, usage amount (numeric only)
5. Find unit: kWh, KL, litres, cubic meters, tons, etc.
6. If information is not present, use null

OUTPUT FORMAT (valid JSON only):
{{
    "entity_id": "ID or null",
    "location": "Location text or null",
    "date": "Date (YYYY-MM-DD) or null",
    "value": "numeric value or null",
    "unit": "measurement unit or null"
}}

Return ONLY the JSON object, no other text.""",
            
            expected_fields=["entity_id", "location", "date", "value", "unit"],
            example_output={
                "entity_id": "METER-12345",
                "location": "Zone-4, Sector B",
                "date": "2024-01-15",
                "value": "450.5",
                "unit": "kWh"
            }
        )
        
        return {
            "constitution": constitution_prompt,
            "mathematics": mathematics_prompt,
            "utility": utility_prompt
        }
    
    def _clean_text_for_extraction(self, text: str, max_length: int = 2000) -> str:
        # basic preprocessing before sending to model
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def _parse_json_response(self, response: str, expected_fields: List[str]) -> Dict[str, Optional[str]]:
        # parse json output from model safely
        json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                result = {}
                for field in expected_fields:
                    value = data.get(field)
                    if value in ["", "null", "None", "N/A", "n/a"]:
                        value = None
                    result[field] = value
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                logger.debug(f"Attempted to parse: {json_str[:200]}")
        
        logger.info("Using fallback pattern extraction")
        return self._fallback_extraction(response, expected_fields)
    
    def _fallback_extraction(self, text: str, expected_fields: List[str]) -> Dict[str, Optional[str]]:
        # fallback regex parsing
        result = {field: None for field in expected_fields}
        
        for field in expected_fields:
            pattern = rf'"{field}"\s*:\s*"([^"]*)"'
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if value and value.lower() not in ["null", "none", "n/a", ""]:
                    result[field] = value
        
        return result
    
    def extract_structured_data(self, chunk: Dict, doc_type: str) -> Dict:
        # extract structured fields from single chunk
        if doc_type not in self.prompts:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        prompt_config = self.prompts[doc_type]
        
        text = chunk.get("text", "")
        cleaned_text = self._clean_text_for_extraction(text)
        
        if len(cleaned_text) < 10:
            logger.warning("Text too short, skipping extraction")
            return self._create_empty_result(prompt_config.expected_fields, chunk)
        
        user_prompt = prompt_config.user_template.format(text=cleaned_text)
        
        response = self.client.generate(
            prompt=user_prompt,
            system_prompt=prompt_config.system_prompt
        )
        
        if not response:
            logger.warning("Empty response from model")
            return self._create_empty_result(prompt_config.expected_fields, chunk)
        
        extracted = self._parse_json_response(response, prompt_config.expected_fields)
        
        extracted["page_number"] = chunk.get("page_nums", [None])[0]
        extracted["raw_text"] = text[:500]
        
        return extracted
    
    def _create_empty_result(self, fields: List[str], chunk: Dict) -> Dict:
        # create empty output template
        result = {field: None for field in fields}
        result["page_number"] = chunk.get("page_nums", [None])[0]
        result["raw_text"] = chunk.get("text", "")[:500]
        return result
    
    def batch_extract(self, chunks: List[Dict], doc_type: str) -> List[Dict]:
        # batch extraction loop
        from tqdm import tqdm
        
        results = []
        
        logger.info(f"Starting batch extraction for {len(chunks)} chunks...")
        
        for chunk in tqdm(chunks, desc=f"Extracting {doc_type}"):
            try:
                result = self.extract_structured_data(chunk, doc_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Extraction failed for chunk: {e}")
                results.append(self._create_empty_result(
                    self.prompts[doc_type].expected_fields, 
                    chunk
                ))
        
        non_empty = sum(1 for r in results if any(v is not None for k, v in r.items() if k not in ["page_number", "raw_text"]))
        logger.info(f"Extracted {non_empty}/{len(results)} non-empty records")
        
        return results


def test_extraction():
    # quick functional test
    print("="*60)
    print("TESTING OLLAMA PHI3 EXTRACTION")
    print("="*60)
    
    try:
        slm = ExtractionSLM()
        
        test_chunk_const = {
            "text": """PART III
            FUNDAMENTAL RIGHTS
            
            Article 14 Equality before law
            The State shall not deny to any person equality before the law 
            or the equal protection of the laws within the territory of India.""",
            "page_nums": [12]
        }
        
        result = slm.extract_structured_data(test_chunk_const, "constitution")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_extraction()