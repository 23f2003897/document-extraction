import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="Document Extraction MVP",
    page_icon="📄",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_database_connection():
    """Get database connection"""
    db_path = "data/documents.db"
    if not Path(db_path).exists():
        st.error(f"Database not found at {db_path}. Please run the pipeline first.")
        return None
    return sqlite3.connect(db_path, check_same_thread=False)

def get_database_stats(conn):
    """Get database statistics"""
    cursor = conn.cursor()
    stats = {}
    
    for table in ["constitution", "mathematics", "utility"]:
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()[0]
        except:
            stats[table] = 0
    
    return stats

def search_constitution(conn, search_type, search_value):
    """Search Constitution database"""
    cursor = conn.cursor()
    
    if search_type == "Article Number":
        query = "SELECT * FROM constitution WHERE article_number = ?"
        cursor.execute(query, (search_value,))
    elif search_type == "Part":
        query = "SELECT * FROM constitution WHERE part LIKE ? ORDER BY article_number"
        cursor.execute(query, (f"%{search_value}%",))
    elif search_type == "All Records":
        query = "SELECT * FROM constitution ORDER BY article_number LIMIT 100"
        cursor.execute(query)
    
    columns = [description[0] for description in cursor.description]
    results = cursor.fetchall()
    
    return pd.DataFrame(results, columns=columns) if results else pd.DataFrame()

def search_mathematics(conn, search_type, search_value):
    """Search Mathematics database"""
    cursor = conn.cursor()
    
    if search_type == "Chapter":
        query = "SELECT * FROM mathematics WHERE chapter_name LIKE ? ORDER BY theorem_number"
        cursor.execute(query, (f"%{search_value}%",))
    elif search_type == "Theorem":
        query = "SELECT * FROM mathematics WHERE theorem_number LIKE ? OR theorem_title LIKE ?"
        cursor.execute(query, (f"%{search_value}%", f"%{search_value}%"))
    elif search_type == "All Records":
        query = "SELECT * FROM mathematics ORDER BY chapter_name, theorem_number LIMIT 100"
        cursor.execute(query)
    
    columns = [description[0] for description in cursor.description]
    results = cursor.fetchall()
    
    return pd.DataFrame(results, columns=columns) if results else pd.DataFrame()

def search_utility(conn, location=None, date_from=None, date_to=None):
    """Search Utility database"""
    cursor = conn.cursor()
    
    query = "SELECT * FROM utility WHERE 1=1"
    params = []
    
    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    
    query += " ORDER BY date DESC LIMIT 100"
    
    cursor.execute(query, params)
    columns = [description[0] for description in cursor.description]
    results = cursor.fetchall()
    
    return pd.DataFrame(results, columns=columns) if results else pd.DataFrame()

# Main App
def main():
    st.title("📄 Document Extraction MVP")
    st.markdown("*Search and explore extracted data from PDFs - Water Utility Domain*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database connection
        conn = get_database_connection()
        
        if conn:
            st.success("✅ Database Connected")
            
            # Show statistics
            stats = get_database_stats(conn)
            st.markdown("### 📊 Database Statistics")
            st.metric("Constitution Records", stats.get("constitution", 0))
            st.metric("Mathematics Records", stats.get("mathematics", 0))
            st.metric("Utility Records", stats.get("utility", 0))
        else:
            st.error("❌ Database Not Found")
            st.info("Run `python pipeline.py` to process PDFs")
            return
        
        st.markdown("---")
        st.markdown("### 🔍 Document Type")
        doc_type = st.radio(
            "Select document type:",
            ["Constitution", "Mathematics", "Utility"]
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("Document Extraction MVP v1.0")
        st.caption("Powered by PyMuPDF + Regex Extraction")
    
    # Main content area
    if doc_type == "Constitution":
        st.header("📜 Indian Constitution Search")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            search_type = st.selectbox(
                "Search by:",
                ["Article Number", "Part", "All Records"]
            )
        
        with col2:
            if search_type != "All Records":
                search_value = st.text_input(
                    "Enter search value:",
                    placeholder="e.g., 21 or PART III"
                )
            else:
                search_value = None
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching..."):
                df = search_constitution(conn, search_type, search_value)
                
                if not df.empty:
                    st.success(f"Found {len(df)} result(s)")
                    
                    # Display results
                    for idx, row in df.iterrows():
                        with st.expander(
                            f"📄 Article {row['article_number']} - {row.get('article_title', 'N/A')[:50]}"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Article Number:** {row['article_number']}")
                                st.markdown(f"**Title:** {row.get('article_title', 'N/A')}")
                            with col2:
                                st.markdown(f"**Part:** {row.get('part', 'N/A')}")
                                st.markdown(f"**Chapter:** {row.get('chapter', 'N/A')}")
                                st.markdown(f"**Page:** {row.get('page_number', 'N/A')}")
                            
                            st.markdown("**Raw Text:**")
                            st.text_area(
                                "Content",
                                row.get('raw_text', 'N/A'),
                                height=150,
                                key=f"const_{idx}",
                                label_visibility="collapsed"
                            )
                else:
                    st.warning("No results found")
    
    elif doc_type == "Mathematics":
        st.header("📐 Engineering Mathematics Search")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            search_type = st.selectbox(
                "Search by:",
                ["Chapter", "Theorem", "All Records"]
            )
        
        with col2:
            if search_type != "All Records":
                search_value = st.text_input(
                    "Enter search value:",
                    placeholder="e.g., Calculus or 2.1"
                )
            else:
                search_value = None
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching..."):
                df = search_mathematics(conn, search_type, search_value)
                
                if not df.empty:
                    st.success(f"Found {len(df)} result(s)")
                    
                    # Display results
                    for idx, row in df.iterrows():
                        with st.expander(
                            f"📐 {row.get('chapter_name', 'N/A')} - Theorem {row.get('theorem_number', 'N/A')}"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Chapter:** {row.get('chapter_name', 'N/A')}")
                                st.markdown(f"**Section:** {row.get('section_name', 'N/A')}")
                            with col2:
                                st.markdown(f"**Theorem Number:** {row.get('theorem_number', 'N/A')}")
                                st.markdown(f"**Theorem Title:** {row.get('theorem_title', 'N/A')}")
                                st.markdown(f"**Page:** {row.get('page_number', 'N/A')}")
                            
                            st.markdown("**Raw Text:**")
                            st.text_area(
                                "Content",
                                row.get('raw_text', 'N/A'),
                                height=150,
                                key=f"math_{idx}",
                                label_visibility="collapsed"
                            )
                else:
                    st.warning("No results found")
    
    else:  # Utility
        st.header("💧 Water Utility Search")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input(
                "Location:",
                placeholder="e.g., Zone-4, Delhi"
            )
        
        with col2:
            date_from = st.date_input(
                "From Date:",
                value=datetime.now() - timedelta(days=30)
            )
        
        with col3:
            date_to = st.date_input(
                "To Date:",
                value=datetime.now()
            )
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner("Searching..."):
                df = search_utility(
                    conn,
                    location if location else None,
                    date_from.strftime("%Y-%m-%d") if date_from else None,
                    date_to.strftime("%Y-%m-%d") if date_to else None
                )
                
                if not df.empty:
                    st.success(f"Found {len(df)} result(s)")
                    
                    # Display as table first
                    st.dataframe(
                        df[['entity_id', 'location', 'date', 'value', 'unit', 'page_number']],
                        use_container_width=True
                    )
                    
                    # Detailed view
                    st.markdown("### 📋 Detailed Records")
                    for idx, row in df.iterrows():
                        with st.expander(
                            f"💧 {row.get('entity_id', 'N/A')} - {row.get('location', 'N/A')} ({row.get('date', 'N/A')})"
                        ):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Entity ID:** {row.get('entity_id', 'N/A')}")
                                st.markdown(f"**Location:** {row.get('location', 'N/A')}")
                            with col2:
                                st.markdown(f"**Date:** {row.get('date', 'N/A')}")
                                st.markdown(f"**Page:** {row.get('page_number', 'N/A')}")
                            with col3:
                                st.markdown(f"**Value:** {row.get('value', 'N/A')}")
                                st.markdown(f"**Unit:** {row.get('unit', 'N/A')}")
                            
                            st.markdown("**Raw Text:**")
                            st.text_area(
                                "Content",
                                row.get('raw_text', 'N/A'),
                                height=100,
                                key=f"util_{idx}",
                                label_visibility="collapsed"
                            )
                else:
                    st.warning("No results found. Try different search criteria.")

if __name__ == "__main__":
    main()