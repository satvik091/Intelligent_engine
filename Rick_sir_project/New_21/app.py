import streamlit as st
import os
from langchain_community.tools import Tool, DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Intelligent Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .search-box {
        margin: 2rem 0;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    .source-container {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .tool-info {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize all components with caching for better performance"""
    try:
        # Set USER_AGENT for Wikipedia requests
        os.environ["USER_AGENT"] = "IntelligentSearchEngine/1.0"
        
        # Initialize tools
        search = DuckDuckGoSearchRun()
        wikipedia_api = WikipediaAPIWrapper()
        wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api)
        arxiv = ArxivQueryRun()
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Ollama LLM
        llm = OllamaLLM(
            model="hf.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF:Q4_K_M",  # Change this to your preferred Ollama model
            temperature=0.5,
            num_ctx=2048
        )
        
        # Create tools with detailed descriptions
        tools = [
            Tool(
                name="Internet_Search",
                func=search.run,
                description=(
                    "Useful for searching current information, news, and general web content. "
                    "Input should be a search query. Use this for topics that require up-to-date information."
                )
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description=(
                    "Useful for factual information about people, places, companies, historical events, "
                    "and scientific concepts. Input should be a precise search term."
                )
            ),
            Tool(
                name="Arxiv",
                func=arxiv.run,
                description=(
                    "Useful for searching academic papers, scientific research, and technical publications. "
                    "Input should be keywords related to academic topics."
                )
            )
        ]
        
        # Create agent
        agent_prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        # Text processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        return {
            'agent_executor': agent_executor,
            'embeddings': embeddings,
            'text_splitter': text_splitter,
            'llm': llm
        }
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None

def create_summarization_prompt():
    """Create a prompt template for AI-based summarization"""
    return PromptTemplate(
        input_variables=["context", "query"],
        template="""
        You are an expert research assistant and summarizer. Your task is to provide a comprehensive, 
        well-structured summary based on the information gathered from multiple sources.

        Context from research:
        {context}

        Original Query: {query}

        Instructions:
        1. Provide a clear, concise summary that directly answers the query
        2. Include key facts, statistics, and important details
        3. Organize information logically with proper structure
        4. Mention which sources provided specific information
        5. If there are conflicting viewpoints, present them objectively
        6. Keep the summary informative but accessible

        Summary:
        """
    )

def extract_sources_from_response(response_text, tools_used):
    """Extract and format sources from agent response"""
    sources = []
    documents = []
    
    # Parse the response to extract information from different tools
    if "Internet_Search" in response_text or "search" in str(tools_used).lower():
        sources.append(("Web Search", "Information from web search results"))
        documents.append(Document(
            page_content=response_text, 
            metadata={"source": "Web Search"}
        ))
    
    if "Wikipedia" in response_text or "wikipedia" in str(tools_used).lower():
        sources.append(("Wikipedia", "Information from Wikipedia"))
        documents.append(Document(
            page_content=response_text, 
            metadata={"source": "Wikipedia"}
        ))
        
    if "Arxiv" in response_text or "arxiv" in str(tools_used).lower():
        sources.append(("Arxiv", "Information from academic papers"))
        documents.append(Document(
            page_content=response_text, 
            metadata={"source": "Arxiv"}
        ))
    
    return sources, documents

def perform_intelligent_search(query, components):
    """Perform intelligent search using the agent and return summarized results"""
    try:
        agent_executor = components['agent_executor']
        embeddings = components['embeddings']
        text_splitter = components['text_splitter']
        llm = components['llm']
        
        # Execute agent to gather information
        with st.spinner("🔍 Searching across multiple sources..."):
            agent_response = agent_executor.invoke({"input": query})
            
        raw_output = agent_response.get("output", "")
        
        # Extract sources and create documents
        sources, documents = extract_sources_from_response(
            raw_output, 
            agent_response.get("intermediate_steps", [])
        )
        
        # Create enhanced context for summarization
        context = raw_output
        
        # If we have documents, create vector store for better context
        if documents:
            with st.spinner("📊 Processing and analyzing information..."):
                chunks = text_splitter.split_documents(documents)
                if chunks:
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    relevant_docs = retriever.get_relevant_documents(query)
                    
                    context = "\n\n".join([
                        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                        f"Content: {doc.page_content[:1000]}..."
                        for doc in relevant_docs
                    ])
        
        # Generate AI-based summary
        with st.spinner("🤖 Generating intelligent summary..."):
            summarization_prompt = create_summarization_prompt()
            summary_input = summarization_prompt.format(context=context, query=query)
            summary = llm.invoke(summary_input)
        
        return {
            "summary": summary,
            "raw_context": context,
            "sources": sources,
            "agent_response": raw_output
        }
        
    except Exception as e:
        logger.error(f"Error in intelligent search: {str(e)}")
        return {
            "summary": f"An error occurred while processing your request: {str(e)}",
            "raw_context": "",
            "sources": [],
            "agent_response": ""
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Intelligent Search Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d;">Powered by Ollama AI with Multi-Source Search & Advanced Summarization</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # st.header("🛠️ Configuration")
        
        # Model selection
        # model_name = st.selectbox(
        #     "Select Ollama Model",
        #     ["llama3.2:latest", "mistral:latest", "codellama:latest", "llama2:latest"],
        #     help="Choose your preferred Ollama model"
        # )
        
        # # Search options
        # st.header("🔍 Search Options")
        # max_iterations = st.slider("Max Search Iterations", 1, 10, 5)
        # temperature = st.slider("AI Temperature", 0.0, 1.0, 0.5, 0.1)
        
        # Tool information
        st.header("🧰 Available Tools")
        with st.expander("View Tool Details"):
            st.markdown("""
            **Internet Search** 📰
            - Current news and web content
            - Real-time information
            
            **Wikipedia** 📚
            - Factual information
            - Historical events & biographies
            
            **Arxiv** 🔬
            - Academic papers
            - Scientific research
            """)
    
    # Initialize components
    if 'components' not in st.session_state:
        with st.spinner("🚀 Initializing AI components..."):
            st.session_state.components = initialize_components()
    
    if not st.session_state.components:
        st.error("Failed to initialize components. Please check your setup.")
        return
    
    # Search interface
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'Latest developments in artificial intelligence' or 'Quantum computing research papers'",
        help="Ask anything! The AI will search across web, Wikipedia, and academic papers."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search execution
    if search_button and query:
        start_time = time.time()
        
        # Perform search
        results = perform_intelligent_search(query, st.session_state.components)
        
        search_time = time.time() - start_time
        
        # Display results
        st.markdown("---")
        
        # Search metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Search Time", f"{search_time:.2f}s")
        
        with col2:
            st.metric("Status", "✅ Complete")
        
        # AI Summary
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("🤖 AI-Generated Summary")
        st.markdown(results['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Source Information
        # if results['sources']:
        #     st.subheader("📋 Sources Used")
        #     for i, (source_type, source_desc) in enumerate(results['sources'], 1):
        #         with st.expander(f"📌 {source_type}"):
        #             st.write(source_desc)
        
        # Raw Context (expandable)
        with st.expander("🔍 View Raw Research Data"):
            st.text_area("Raw Context", results['raw_context'], height=300)
        
        # Agent Response (for debugging)
        with st.expander("🤖 View Agent Response"):
            st.code(results['agent_response'], language="text")
    
    elif search_button and not query:
        st.warning("⚠️ Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6c757d;">Built with Streamlit, LangChain, and Ollama | '
        'Multi-source intelligent search with AI summarization</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
