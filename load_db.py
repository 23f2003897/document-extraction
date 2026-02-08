import sqlite3
import json
import logging
from typing import List, Dict
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DocumentDatabase:
    """
    SQLite database for storing extracted document data.
    Separate tables for each document type.
    """
    
    def __init__(self, db_path: str = "data/documents.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to database: {self.db_path}")
    
    def _create_tables(self):
        """Create tables for each document type"""
        
        # Constitution table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS constitution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_number TEXT,
            article_title TEXT,
            part TEXT,
            chapter TEXT,
            page_number INTEGER,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index for faster searches
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_constitution_article 
        ON constitution(article_number)
        """)
        
        # Mathematics table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS mathematics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_name TEXT,
            section_name TEXT,
            theorem_number TEXT,
            theorem_title TEXT,
            page_number INTEGER,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_mathematics_chapter 
        ON mathematics(chapter_name)
        """)
        
        # Utility table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS utility (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT,
            location TEXT,
            date TEXT,
            value REAL,
            unit TEXT,
            page_number INTEGER,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_utility_location 
        ON utility(location)
        """)
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_utility_date 
        ON utility(date)
        """)
        
        # Chunks with embeddings table (for semantic search)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            doc_type TEXT,
            page_number INTEGER,
            text TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_doctype 
        ON document_chunks(doc_type)
        """)
        
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_page 
        ON document_chunks(page_number)
        """)
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def insert_constitution_record(self, record: Dict):
        """Insert a single Constitution record"""
        self.cursor.execute("""
        INSERT INTO constitution (article_number, article_title, part, chapter, page_number, raw_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record.get("article_number"),
            record.get("article_title"),
            record.get("part"),
            record.get("chapter"),
            record.get("page_number"),
            record.get("raw_text")
        ))
    
    def insert_mathematics_record(self, record: Dict):
        """Insert a single Mathematics record"""
        self.cursor.execute("""
        INSERT INTO mathematics (chapter_name, section_name, theorem_number, theorem_title, page_number, raw_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record.get("chapter_name"),
            record.get("section_name"),
            record.get("theorem_number"),
            record.get("theorem_title"),
            record.get("page_number"),
            record.get("raw_text")
        ))
    
    def insert_utility_record(self, record: Dict):
        """Insert a single Utility record"""
        self.cursor.execute("""
        INSERT INTO utility (entity_id, location, date, value, unit, page_number, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("entity_id"),
            record.get("location"),
            record.get("date"),
            record.get("value"),
            record.get("unit"),
            record.get("page_number"),
            record.get("raw_text")
        ))
    
    def bulk_insert(self, records: List[Dict], doc_type: str):
        """
        Insert multiple records efficiently.
        """
        logger.info(f"Inserting {len(records)} records for {doc_type}")
        
        insert_func = {
            "constitution": self.insert_constitution_record,
            "mathematics": self.insert_mathematics_record,
            "utility": self.insert_utility_record
        }
        
        if doc_type not in insert_func:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        for record in records:
            try:
                insert_func[doc_type](record)
            except Exception as e:
                logger.error(f"Failed to insert record: {e}")
                logger.error(f"Record: {record}")
        
        self.conn.commit()
        logger.info(f"Inserted {len(records)} records successfully")
    
    def insert_chunk_with_embedding(self, chunk: Dict, doc_type: str):
        """Insert a chunk with its embedding vector"""
        import pickle
        
        # Serialize embedding (list of floats) to binary
        embedding_blob = pickle.dumps(chunk.get("embedding", []))
        
        self.cursor.execute("""
        INSERT INTO document_chunks (chunk_id, doc_type, page_number, text, embedding)
        VALUES (?, ?, ?, ?, ?)
        """, (
            chunk.get("chunk_id"),
            doc_type,
            chunk.get("page_nums", [None])[0],
            chunk.get("text"),
            embedding_blob
        ))
    
    def bulk_insert_chunks(self, chunks: List[Dict], doc_type: str):
        """Insert multiple chunks with embeddings"""
        logger.info(f"Inserting {len(chunks)} chunks with embeddings for {doc_type}")
        
        for chunk in chunks:
            try:
                self.insert_chunk_with_embedding(chunk, doc_type)
            except Exception as e:
                logger.error(f"Failed to insert chunk: {e}")
        
        self.conn.commit()
        logger.info(f"✅ Inserted {len(chunks)} chunks successfully")
    
    def get_all_chunks(self, doc_type: str = None) -> List[Dict]:
        """Retrieve all chunks, optionally filtered by document type"""
        import pickle
        
        if doc_type:
            self.cursor.execute("""
            SELECT * FROM document_chunks WHERE doc_type = ?
            """, (doc_type,))
        else:
            self.cursor.execute("SELECT * FROM document_chunks")
        
        rows = self.cursor.fetchall()
        chunks = []
        
        for row in rows:
            chunk = dict(row)
            # Deserialize embedding
            chunk["embedding"] = pickle.loads(chunk["embedding"])
            chunks.append(chunk)
        
        return chunks
    
    # Query methods
    def search_constitution_by_article(self, article_number: str) -> List[Dict]:
        """Search Constitution by article number"""
        self.cursor.execute("""
        SELECT * FROM constitution 
        WHERE article_number = ?
        """, (article_number,))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_constitution_by_part(self, part: str) -> List[Dict]:
        """Search Constitution by part"""
        self.cursor.execute("""
        SELECT * FROM constitution 
        WHERE part LIKE ?
        ORDER BY article_number
        """, (f"%{part}%",))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_mathematics_by_chapter(self, chapter: str) -> List[Dict]:
        """Search Mathematics by chapter"""
        self.cursor.execute("""
        SELECT * FROM mathematics 
        WHERE chapter_name LIKE ?
        ORDER BY theorem_number
        """, (f"%{chapter}%",))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_mathematics_by_theorem(self, theorem: str) -> List[Dict]:
        """Search Mathematics by theorem number/title"""
        self.cursor.execute("""
        SELECT * FROM mathematics 
        WHERE theorem_number LIKE ? OR theorem_title LIKE ?
        """, (f"%{theorem}%", f"%{theorem}%"))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_utility_by_location(self, location: str) -> List[Dict]:
        """Search Utility by location"""
        self.cursor.execute("""
        SELECT * FROM utility 
        WHERE location LIKE ?
        ORDER BY date DESC
        """, (f"%{location}%",))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_utility_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Search Utility by date range"""
        self.cursor.execute("""
        SELECT * FROM utility 
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC
        """, (start_date, end_date))
        return [dict(row) for row in self.cursor.fetchall()]
    
    # Wrappers for notebook Search UI (Document_Extraction_Search_UI.ipynb)
    def search_article(self, query: str) -> List[Dict]:
        """Search constitution by article number or part. Alias for notebook."""
        out = self.search_constitution_by_article(query)
        if not out:
            out = self.search_constitution_by_part(query)
        return out
    
    def get_theorems(self, query: str) -> List[Dict]:
        """Search mathematics by chapter or theorem. Alias for notebook."""
        by_ch = self.search_mathematics_by_chapter(query)
        by_th = self.search_mathematics_by_theorem(query)
        seen = set()
        out = []
        for r in by_ch + by_th:
            k = r.get("id")
            if k not in seen:
                seen.add(k)
                out.append(r)
        return out
    
    def filter_location(self, query: str) -> List[Dict]:
        """Search utility by location. Alias for notebook."""
        return self.search_utility_by_location(query)
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        for table in ["constitution", "mathematics", "utility", "document_chunks"]:
            self.cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = self.cursor.fetchone()["count"]
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")



# Testing function
def test_database():
    """Test database operations"""
    
    # Initialize database
    db = DocumentDatabase("data/test.db")
    
    # Test Constitution insert
    test_record_const = {
        "article_number": "21",
        "article_title": "Right to Life",
        "part": "PART III",
        "chapter": None,
        "page_number": 12,
        "raw_text": "Sample text..."
    }
    db.insert_constitution_record(test_record_const)
    
    # Test Mathematics insert
    test_record_math = {
        "chapter_name": "Differential Calculus",
        "section_name": "Limits",
        "theorem_number": "3.1",
        "theorem_title": "Squeeze Theorem",
        "page_number": 45,
        "raw_text": "Sample text..."
    }
    db.insert_mathematics_record(test_record_math)
    
    # Test Utility insert
    test_record_utility = {
        "entity_id": "METER-12345",
        "location": "Zone-4",
        "date": "2024-01-15",
        "value": 450.5,
        "unit": "kWh",
        "page_number": 3,
        "raw_text": "Sample text..."
    }
    db.insert_utility_record(test_record_utility)
    
    # Test chunk with embedding insert
    test_chunk = {
        "chunk_id": 0,
        "text": "Test chunk text",
        "page_nums": [1],
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    db.insert_chunk_with_embedding(test_chunk, "constitution")
    
    db.conn.commit()
    
    # Test queries
    print("\n=== Testing Queries ===")
    
    result = db.search_constitution_by_article("21")
    print(f"\nArticle 21 search: {len(result)} results")
    print(json.dumps(result[0], indent=2, default=str))
    
    result = db.search_mathematics_by_theorem("3.1")
    print(f"\nTheorem 3.1 search: {len(result)} results")
    print(json.dumps(result[0], indent=2, default=str))
    
    result = db.search_utility_by_location("Zone-4")
    print(f"\nZone-4 search: {len(result)} results")
    print(json.dumps(result[0], indent=2, default=str))
    
    # Test chunk retrieval
    chunks = db.get_all_chunks("constitution")
    print(f"\nChunks for constitution: {len(chunks)}")
    if chunks:
        print(f"First chunk has embedding of length: {len(chunks[0]['embedding'])}")
    
    # Statistics
    stats = db.get_statistics()
    print(f"\n=== Database Statistics ===")
    print(json.dumps(stats, indent=2))
    
    db.close()



if __name__ == "__main__":
    test_database()
