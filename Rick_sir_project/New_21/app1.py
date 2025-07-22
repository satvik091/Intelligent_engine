# Advanced Multi-Source Search Engine with LangChain and Vector Database
# Requirements: pip install streamlit langchain langchain-community faiss-cpu sentence-transformers groq arxiv wikipedia

import streamlit as st
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os

# Core imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Custom imports for internet search
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings using HuggingFace sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model"""
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.model.encode([text], convert_to_tensor=False)[0]

class VectorDatabase:
    """FAISS-based vector database for storing and retrieving embeddings"""
    
    def __init__(self, embedding_dim: int):
        """Initialize FAISS index"""
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.documents = []
        self.metadata = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """Add documents with their embeddings and metadata"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'index_size': self.index.ntotal
        }

class CustomInternetTool:
    """Custom tool for internet search using web scraping"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the internet and return results"""
        try:
            # Use DuckDuckGo for search (no API key required)
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Parse search results
            for result_div in soup.find_all('div', class_='web-result')[:num_results]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True)
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'Internet Search'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Internet search error: {str(e)}")
            return []

class MultiSourceSearchEngine:
    """Main search engine class integrating all components"""
    
    def __init__(self, groq_api_key: str):
        """Initialize the search engine"""
        self.embedding_manager = EmbeddingManager()
        self.vector_db = VectorDatabase(self.embedding_manager.embedding_dim)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0.1,
            max_tokens=2048
        )
        
        # Initialize tools
        self._init_tools()
        
        # Initialize agent
        self._init_agent()
    
    def _init_tools(self):
        """Initialize search tools"""
        # Arxiv tool
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
        self.arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        
        # Wikipedia tool
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
        self.wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        
        # Custom internet tool
        self.internet_tool = CustomInternetTool()
        
        # Create LangChain tools
        self.tools = [
            Tool(
                name="ArxivSearch",
                description="Search academic papers on Arxiv. Use this for scientific research questions.",
                func=self.arxiv_tool.run
            ),
            Tool(
                name="WikipediaSearch", 
                description="Search Wikipedia for general knowledge and factual information.",
                func=self.wiki_tool.run
            )
        ]
    
    def _init_agent(self):
        """Initialize the ReAct agent"""
        template = """
        You are a research assistant that can search multiple sources to answer questions.
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Question: {input}
        Thought: {agent_scratchpad}
        """
        
        prompt = PromptTemplate.from_template(template)
        
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3
        )
    
    def search_all_sources(self, query: str) -> Dict[str, Any]:
        """Search all sources and return combined results"""
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'arxiv_results': [],
            'wikipedia_results': [],
            'internet_results': [],
            'vector_results': []
        }
        
        try:
            # Search Arxiv
            st.info("🔬 Searching Arxiv for academic papers...")
            arxiv_result = self.arxiv_tool.run(query)
            if arxiv_result:
                results['arxiv_results'] = self._parse_arxiv_results(arxiv_result)
            
            # Search Wikipedia
            st.info("📚 Searching Wikipedia...")
            wiki_result = self.wiki_tool.run(query)
            if wiki_result:
                results['wikipedia_results'] = self._parse_wiki_results(wiki_result)
            
            # Search Internet
            st.info("🌐 Searching the internet...")
            internet_results = self.internet_tool.search(query, num_results=5)
            results['internet_results'] = internet_results
            
            # Add to vector database and search
            self._add_to_vector_db(results)
            
            # Search vector database
            if self.vector_db.get_stats()['total_documents'] > 0:
                query_embedding = self.embedding_manager.encode_single(query)
                vector_results = self.vector_db.search(query_embedding, top_k=5)
                results['vector_results'] = vector_results
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            logger.error(f"Search error: {str(e)}")
        
        return results
    
    def _parse_arxiv_results(self, arxiv_text: str) -> List[Dict]:
        """Parse Arxiv results"""
        results = []
        if arxiv_text and "Published:" in arxiv_text:
            sections = arxiv_text.split("Published:")
            for i, section in enumerate(sections[1:], 1):
                lines = section.strip().split('\n')
                if len(lines) >= 3:
                    results.append({
                        'title': lines[1].replace('Title:', '').strip(),
                        'summary': lines[2].replace('Summary:', '').strip()[:500],
                        'source': 'Arxiv',
                        'rank': i
                    })
        return results
    
    def _parse_wiki_results(self, wiki_text: str) -> List[Dict]:
        """Parse Wikipedia results"""
        results = []
        if wiki_text:
            # Split by page titles or use the entire text
            chunks = self.text_splitter.split_text(wiki_text)
            for i, chunk in enumerate(chunks[:3]):  # Limit to 3 chunks
                results.append({
                    'content': chunk[:500],
                    'source': 'Wikipedia',
                    'rank': i + 1
                })
        return results
    
    def _add_to_vector_db(self, results: Dict[str, Any]):
        """Add search results to vector database"""
        documents = []
        metadata = []
        
        # Add Arxiv results
        for result in results.get('arxiv_results', []):
            doc_text = f"{result.get('title', '')} {result.get('summary', '')}"
            documents.append(doc_text)
            metadata.append({
                'source': 'Arxiv',
                'title': result.get('title', ''),
                'type': 'academic_paper'
            })
        
        # Add Wikipedia results
        for result in results.get('wikipedia_results', []):
            documents.append(result.get('content', ''))
            metadata.append({
                'source': 'Wikipedia', 
                'type': 'encyclopedia'
            })
        
        # Add Internet results
        for result in results.get('internet_results', []):
            doc_text = f"{result.get('title', '')} {result.get('snippet', '')}"
            documents.append(doc_text)
            metadata.append({
                'source': 'Internet',
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'type': 'web_content'
            })
        
        if documents:
            embeddings = self.embedding_manager.encode(documents)
            self.vector_db.add_documents(documents, embeddings, metadata)
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate LLM-powered summary of search results"""
        try:
            # Prepare context from all sources
            context_parts = []
            
            # Add Arxiv results
            if results['arxiv_results']:
                arxiv_context = "Academic Papers (Arxiv):\n"
                for i, result in enumerate(results['arxiv_results'][:3]):
                    arxiv_context += f"{i+1}. {result.get('title', '')}: {result.get('summary', '')}\n"
                context_parts.append(arxiv_context)
            
            # Add Wikipedia results
            if results['wikipedia_results']:
                wiki_context = "Encyclopedia (Wikipedia):\n"
                for i, result in enumerate(results['wikipedia_results'][:2]):
                    wiki_context += f"{i+1}. {result.get('content', '')}\n"
                context_parts.append(wiki_context)
            
            # Add Internet results
            if results['internet_results']:
                internet_context = "Web Search Results:\n"
                for i, result in enumerate(results['internet_results'][:3]):
                    internet_context += f"{i+1}. {result.get('title', '')}: {result.get('snippet', '')}\n"
                context_parts.append(internet_context)
            
            full_context = "\n\n".join(context_parts)
            
            # Generate summary using LLM
            prompt = f"""
            Based on the following search results from multiple sources, provide a comprehensive summary that answers the query: "{results['query']}"
            
            Search Results:
            {full_context}
            
            Please provide:
            1. A clear, comprehensive answer to the query
            2. Key insights from academic sources (if available)
            3. Supporting information from other sources
            4. Any important caveats or limitations
            
            Make sure to synthesize information from all available sources and provide proper attribution.
            """
            
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return f"Error generating summary: {str(e)}"

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Advanced Multi-Source Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔍 Advanced Multi-Source Search Engine")
    st.markdown("*Powered by LangChain, Vector Database, and Multiple AI Sources*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    groq_api_key = st.sidebar.text_input(
        "Groq API Key", 
        type="password",
        help="Enter your Groq API key for LLM access"
    )
    
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
        st.info("You can get a free API key from: https://console.groq.com/")
        return
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        with st.spinner("Initializing search engine..."):
            try:
                st.session_state.search_engine = MultiSourceSearchEngine(groq_api_key)
                st.success("✅ Search engine initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize search engine: {str(e)}")
                return
    
    search_engine = st.session_state.search_engine
    
    # Main search interface
    st.header("🎯 Search Interface")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'quantum computing applications in machine learning'",
        help="Ask any question - the system will search academic papers, Wikipedia, and the web"
    )
    
    # Search button
    if st.button("🚀 Search All Sources", type="primary"):
        if query:
            with st.spinner("Searching multiple sources..."):
                # Perform search
                results = search_engine.search_all_sources(query)
                
                # Store results in session state
                st.session_state.last_results = results
                
                # Display results
                display_results(results, search_engine)
        else:
            st.warning("Please enter a search query.")
    
    # Display previous results if available
    # if 'last_results' in st.session_state:
    #     st.divider()
    #     st.header("📊 Previous Search Results")
    #     display_results(st.session_state.last_results, search_engine)
    
    # # Sidebar stats
    # if hasattr(search_engine, 'vector_db'):
    #     st.sidebar.header("📈 Database Statistics")
    #     stats = search_engine.vector_db.get_stats()
    #     st.sidebar.metric("Total Documents", stats['total_documents'])
    #     st.sidebar.metric("Embedding Dimension", stats['embedding_dimension'])

def display_results(results: Dict[str, Any], search_engine):
    """Display search results in the UI"""
    
    # Generate and display summary
    st.subheader("🎯 AI-Generated Summary")
    with st.spinner("Generating comprehensive summary..."):
        summary = search_engine.generate_summary(results)
        st.write(summary)
    
    # Create tabs for different sources
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Academic (Arxiv)", "🌐 Wikipedia", "🔗 Internet", "🧠 Vector Search"])
    
    with tab1:
        st.subheader("Academic Papers from Arxiv")
        arxiv_results = results.get('arxiv_results', [])
        if arxiv_results:
            for i, result in enumerate(arxiv_results):
                with st.expander(f"Paper {i+1}: {result.get('title', 'Untitled')[:100]}..."):
                    st.write(f"**Summary:** {result.get('summary', 'No summary available')}")
                    st.caption(f"Source: {result.get('source', 'Arxiv')}")
        else:
            st.info("No academic papers found for this query.")
    
    with tab2:
        st.subheader("Wikipedia Results")
        wiki_results = results.get('wikipedia_results', [])
        if wiki_results:
            for i, result in enumerate(wiki_results):
                with st.expander(f"Wikipedia Entry {i+1}"):
                    st.write(result.get('content', 'No content available'))
                    st.caption(f"Source: {result.get('source', 'Wikipedia')}")
        else:
            st.info("No Wikipedia articles found for this query.")
    
    with tab3:
        st.subheader("Internet Search Results")
        internet_results = results.get('internet_results', [])
        if internet_results:
            for i, result in enumerate(internet_results):
                with st.expander(f"Web Result {i+1}: {result.get('title', 'Untitled')[:80]}..."):
                    st.write(f"**Title:** {result.get('title', 'No title')}")
                    st.write(f"**Snippet:** {result.get('snippet', 'No snippet available')}")
                    if result.get('url'):
                        st.write(f"**URL:** {result.get('url')}")
                    st.caption(f"Source: {result.get('source', 'Internet')}")
        else:
            st.info("No internet results found for this query.")
    
    with tab4:
        st.subheader("Vector Database Search")
        vector_results = results.get('vector_results', [])
        if vector_results:
            for result in vector_results:
                with st.expander(f"Match {result['rank']} (Similarity: {result['similarity_score']:.3f})"):
                    st.write(f"**Content:** {result['document'][:500]}...")
                    st.write(f"**Source:** {result['metadata'].get('source', 'Unknown')}")
                    st.write(f"**Type:** {result['metadata'].get('type', 'Unknown')}")
                    if 'title' in result['metadata']:
                        st.write(f"**Title:** {result['metadata']['title']}")
        else:
            st.info("No vector database results available. Perform a search first to populate the database.")

if __name__ == "__main__":
    main()