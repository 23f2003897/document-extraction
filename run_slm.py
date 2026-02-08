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
    """Container for document-specific prompts and field definitions"""
    system_prompt: str
    user_template: str
    expected_fields: List[str]
    example_output: Dict[str, str]


class OllamaClient:
    """
    Client for interacting with Ollama API
    Handles connection, generation, and error handling
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3"):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
                logger.warning(f"Run: ollama pull {self.model}")
            else:
                logger.info(f"✅ Connected to Ollama - Model: {self.model}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Ollama at {self.base_url}")
            logger.error(f"Error: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise ConnectionError("Ollama service not available")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate response from Ollama
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system context
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 512,  # Max tokens to generate
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=300  # 300 second timeout
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
    """
    Main extraction system using Ollama Phi3
    Handles all document types with improved prompting
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "phi3"):
        logger.info("🚀 Initializing Extraction SLM with Ollama Phi3...")
        
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.prompts = self._initialize_prompts()
        
        logger.info("✅ Extraction SLM ready!")
    
    def _initialize_prompts(self) -> Dict[str, ExtractionPrompt]:
        """
        Initialize document-specific prompts with clear instructions
        Each prompt is optimized for accuracy with examples
        """
        
        # ==================== CONSTITUTION PROMPTS ====================
        constitution_prompt = ExtractionPrompt(
            system_prompt="""You are an expert at extracting structured information from Indian Constitution documents.
Your task is to identify articles, parts, and chapters with high precision.
Always output valid JSON with exact field names as specified.""",
            
            user_template="""Extract information from this Indian Constitution text.

TEXT:
{text}

INSTRUCTIONS:
1. Find the Article number (e.g., "Article 14", "Article 21", "Art. 1")
2. Extract the article title/heading
3. Identify the PART (e.g., "PART III", "PART I") - may be mentioned earlier in document
4. Identify the CHAPTER if mentioned (e.g., "CHAPTER I", "CHAPTER II")
5. If information is not present, use null

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
        
        # ==================== MATHEMATICS PROMPTS ====================
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
4. Find Theorem title/name (e.g., "Squeeze Theorem", "Mean Value Theorem")
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
        
        # ==================== UTILITY PROMPTS ====================
        utility_prompt = ExtractionPrompt(
            system_prompt="""You are an expert at extracting structured information from utility documents (water, electricity, logistics bills/reports).
Your task is to identify entities, locations, dates, and consumption values.
Always output valid JSON with exact field names as specified.""",
            
            user_template="""Extract information from this utility document text (water/electricity/logistics).

TEXT:
{text}

INSTRUCTIONS:
1. Find entity_id: Meter ID, Consumer ID, Account Number, Zone ID, Vehicle ID, etc.
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
        """
        Clean and prepare text for extraction
        
        Args:
            text: Raw text from PDF
            max_length: Maximum characters to process
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might confuse the model
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def _parse_json_response(self, response: str, expected_fields: List[str]) -> Dict[str, Optional[str]]:
        """
        Robust JSON parsing with fallback extraction
        
        Args:
            response: Raw model response
            expected_fields: List of expected field names
            
        Returns:
            Dictionary with extracted fields
        """
        # Try to find JSON block in response
        json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Ensure all expected fields are present
                result = {}
                for field in expected_fields:
                    value = data.get(field)
                    # Convert empty strings and "null" string to None
                    if value in ["", "null", "None", "N/A", "n/a"]:
                        value = None
                    result[field] = value
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                logger.debug(f"Attempted to parse: {json_str[:200]}")
        
        # Fallback: pattern-based extraction
        logger.info("Using fallback pattern extraction")
        return self._fallback_extraction(response, expected_fields)
    
    def _fallback_extraction(self, text: str, expected_fields: List[str]) -> Dict[str, Optional[str]]:
        """
        Fallback extraction using regex patterns when JSON parsing fails
        
        Args:
            text: Response text
            expected_fields: List of field names to extract
            
        Returns:
            Dictionary with extracted values (may be None)
        """
        result = {field: None for field in expected_fields}
        
        # Try to extract field: "value" patterns
        for field in expected_fields:
            # Pattern: "field_name": "value"
            pattern = rf'"{field}"\s*:\s*"([^"]*)"'
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if value and value.lower() not in ["null", "none", "n/a", ""]:
                    result[field] = value
        
        return result
    
    def extract_structured_data(self, chunk: Dict, doc_type: str) -> Dict:
        """
        Extract structured data from a single chunk
        
        Args:
            chunk: Dictionary containing 'text' and 'page_nums'
            doc_type: Type of document ('constitution', 'mathematics', 'utility')
            
        Returns:
            Dictionary with extracted structured data
        """
        if doc_type not in self.prompts:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        prompt_config = self.prompts[doc_type]
        
        # Clean text
        text = chunk.get("text", "")
        cleaned_text = self._clean_text_for_extraction(text)
        
        if len(cleaned_text) < 10:
            logger.warning("Text too short, skipping extraction")
            return self._create_empty_result(prompt_config.expected_fields, chunk)
        
        # Build prompt
        user_prompt = prompt_config.user_template.format(text=cleaned_text)
        
        # Generate response
        logger.debug(f"Generating extraction for {doc_type}...")
        response = self.client.generate(
            prompt=user_prompt,
            system_prompt=prompt_config.system_prompt
        )
        
        if not response:
            logger.warning("Empty response from model")
            return self._create_empty_result(prompt_config.expected_fields, chunk)
        
        logger.debug(f"Model response: {response[:200]}...")
        
        # Parse response
        extracted = self._parse_json_response(response, prompt_config.expected_fields)
        
        # Add metadata
        extracted["page_number"] = chunk.get("page_nums", [None])[0]
        extracted["raw_text"] = text[:500]  # Store first 500 chars of original text
        
        return extracted
    
    def _create_empty_result(self, fields: List[str], chunk: Dict) -> Dict:
        """Create empty result when extraction fails"""
        result = {field: None for field in fields}
        result["page_number"] = chunk.get("page_nums", [None])[0]
        result["raw_text"] = chunk.get("text", "")[:500]
        return result
    
    def batch_extract(self, chunks: List[Dict], doc_type: str) -> List[Dict]:
        """
        Extract structured data from multiple chunks
        
        Args:
            chunks: List of chunk dictionaries
            doc_type: Type of document
            
        Returns:
            List of extracted records
        """
        from tqdm import tqdm
        
        results = []
        
        logger.info(f"Starting batch extraction for {len(chunks)} chunks...")
        
        for chunk in tqdm(chunks, desc=f"⚡ Extracting {doc_type}"):
            try:
                result = self.extract_structured_data(chunk, doc_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Extraction failed for chunk: {e}")
                # Add empty result to maintain alignment
                results.append(self._create_empty_result(
                    self.prompts[doc_type].expected_fields, 
                    chunk
                ))
        
        # Log extraction statistics
        non_empty = sum(1 for r in results if any(v is not None for k, v in r.items() if k not in ["page_number", "raw_text"]))
        logger.info(f"✅ Extracted {non_empty}/{len(results)} non-empty records")
        
        return results


# ==================== TESTING FUNCTION ====================
def test_extraction():
    """Test the extraction system with sample data"""
    
    print("="*60)
    print("TESTING OLLAMA PHI3 EXTRACTION")
    print("="*60)
    
    try:
        slm = ExtractionSLM()
        
        # Test 1: Constitution
        print("\n[TEST 1] Constitution Document")
        test_chunk_const = {
            "text": """PART III
            FUNDAMENTAL RIGHTS
            
            Article 14 Equality before law
            The State shall not deny to any person equality before the law 
            or the equal protection of the laws within the territory of India.""",
            "page_nums": [12]
        }
        
        result = slm.extract_structured_data(test_chunk_const, "constitution")
        print("Result:")
        print(json.dumps(result, indent=2))
        
        # Test 2: Mathematics
        print("\n[TEST 2] Mathematics Textbook")
        test_chunk_math = {
            "text": """Chapter 3: Differential Calculus
            
            3.1 Limits and Continuity
            
            Theorem 3.1 (Squeeze Theorem)
            If f(x) ≤ g(x) ≤ h(x) for all x in some interval...""",
            "page_nums": [45]
        }
        
        result = slm.extract_structured_data(test_chunk_math, "mathematics")
        print("Result:")
        print(json.dumps(result, indent=2))
        
        # Test 3: Utility
        print("\n[TEST 3] Utility Document")
        test_chunk_utility = {
            "text": """Water Consumption Report
            
            Meter ID: METER-12345
            Location: Zone-4, Sector B
            Bill Date: 15-Jan-2024
            Consumption: 450.5 KL
            Previous Reading: 2100.3 KL""",
            "page_nums": [3]
        }
        
        result = slm.extract_structured_data(test_chunk_utility, "utility")
        print("Result:")
        print(json.dumps(result, indent=2))
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_extraction()