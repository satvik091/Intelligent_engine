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
#----
# import streamlit as st
# import os
# from langchain_community.tools import Tool, DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain import hub
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.prompts import PromptTemplate
# import time
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Set page config
# st.set_page_config(
#     page_title="Intelligent Search Engine",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         color: #2E86AB;
#         margin-bottom: 2rem;
#     }
#     .search-box {
#         margin: 2rem 0;
#     }
#     .result-container {
#         background-color: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#         border-left: 4px solid #2E86AB;
#     }
#     .source-container {
#         background-color: #e9ecef;
#         padding: 1rem;
#         border-radius: 5px;
#         margin: 0.5rem 0;
#     }
#     .tool-info {
#         background-color: #d4edda;
#         padding: 1rem;
#         border-radius: 5px;
#         margin: 1rem 0;
#         border-left: 4px solid #28a745;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def initialize_components():
#     """Initialize all components with caching for better performance"""
#     try:
#         # Set USER_AGENT for Wikipedia requests
#         os.environ["USER_AGENT"] = "IntelligentSearchEngine/1.0"
        
#         # Initialize tools
#         search = DuckDuckGoSearchRun()
#         wikipedia_api = WikipediaAPIWrapper()
#         wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api)
#         arxiv = ArxivQueryRun()
        
#         # Initialize embeddings
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )
        
#         # Initialize Ollama LLM
#         llm = OllamaLLM(
#             model="mistral:latest "
#         )
        
#         # Create tools with detailed descriptions
#         tools = [
#             Tool(
#                 name="Internet_Search",
#                 func=search.run,
#                 description=(
#                     "Useful for searching current information, news, and general web content. "
#                     "Input should be a search query. Use this for topics that require up-to-date information."
#                 )
#             ),
#             Tool(
#                 name="Wikipedia",
#                 func=wikipedia.run,
#                 description=(
#                     "Useful for factual information about people, places, companies, historical events, "
#                     "and scientific concepts. Input should be a precise search term."
#                 )
#             ),
#             Tool(
#                 name="Arxiv",
#                 func=arxiv.run,
#                 description=(
#                     "Useful for searching academic papers, scientific research, and technical publications. "
#                     "Input should be keywords related to academic topics."
#                 )
#             )
#         ]
        
#         # Create agent
#         agent_prompt = hub.pull("hwchase17/react")
#         agent = create_react_agent(llm, tools, agent_prompt)
#         agent_executor = AgentExecutor(
#             agent=agent,
#             tools=tools,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_iterations=5
#         )
        
#         # Text processing
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, 
#             chunk_overlap=200
#         )
        
#         return {
#             'agent_executor': agent_executor,
#             'embeddings': embeddings,
#             'text_splitter': text_splitter,
#             'llm': llm
#         }
        
#     except Exception as e:
#         st.error(f"Error initializing components: {str(e)}")
#         return None

# def create_summarization_prompt():
#     """Create a prompt template for AI-based summarization"""
#     return PromptTemplate(
#         input_variables=["context", "query"],
#         template="""
#         You are an expert research assistant and summarizer. Your task is to provide a comprehensive, 
#         well-structured summary based on the information gathered from multiple sources.

#         Context from research:
#         {context}

#         Original Query: {query}

#         Instructions:
#         1. Provide a clear, concise summary that directly answers the query
#         2. Include key facts, statistics, and important details
#         3. Organize information logically with proper structure
#         4. Mention which sources provided specific information
#         5. If there are conflicting viewpoints, present them objectively
#         6. Keep the summary informative but accessible

#         Summary:
#         """
#     )

# def create_text_summarization_prompt():
#     """Create a prompt template for direct text summarization"""
#     return PromptTemplate(
#         input_variables=["text", "summary_type", "length"],
#         template="""
#         You are an expert AI summarization assistant. Your task is to create high-quality summaries of the provided text.

#         Text to summarize:
#         {text}

#         Summary Type: {summary_type}
#         Target Length: {length}

#         Instructions based on summary type:
        
#         **If Extractive Summary:**
#         - Extract the most important sentences and key points directly from the text
#         - Maintain original wording and structure
#         - Focus on factual information and main arguments
        
#         **If Abstractive Summary:**
#         - Rewrite and rephrase the content in your own words
#         - Capture the essence and meaning rather than exact wording
#         - Create a flowing, coherent narrative
        
#         **If Bullet Point Summary:**
#         - Break down into clear, concise bullet points
#         - Each point should represent a key idea or fact
#         - Use parallel structure and consistent formatting
        
#         **If Executive Summary:**
#         - Focus on key decisions, recommendations, and outcomes
#         - Include critical data points and conclusions
#         - Write for business/professional audience

#         Additional Guidelines:
#         1. Maintain accuracy and don't add information not in the original text
#         2. Preserve important context and nuance
#         3. Use clear, professional language
#         4. Structure the summary logically
#         5. Include key statistics, dates, and figures when relevant

#         Summary:
#         """
#     )

# def summarize_text(text, summary_type, length, llm):
#     """AI-based text summarization function"""
#     try:
#         # Create summarization prompt
#         summarization_prompt = create_text_summarization_prompt()
        
#         # Format the prompt
#         prompt_input = summarization_prompt.format(
#             text=text,
#             summary_type=summary_type,
#             length=length
#         )
        
#         # Generate summary
#         summary = llm.invoke(prompt_input)
        
#         # Calculate basic metrics
#         original_words = len(text.split())
#         summary_words = len(summary.split())
#         compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0
        
#         return {
#             "summary": summary,
#             "original_words": original_words,
#             "summary_words": summary_words,
#             "compression_ratio": compression_ratio
#         }
        
#     except Exception as e:
#         logger.error(f"Error in text summarization: {str(e)}")
#         return {
#             "summary": f"Error occurred during summarization: {str(e)}",
#             "original_words": 0,
#             "summary_words": 0,
#             "compression_ratio": 0
#         }

# def extract_text_from_uploaded_file(uploaded_file):
#     """Extract text from uploaded files"""
#     try:
#         if uploaded_file.type == "text/plain":
#             return str(uploaded_file.read(), "utf-8")
#         elif uploaded_file.type == "application/pdf":
#             # Note: For PDF support, you'd need to install PyPDF2 or similar
#             st.warning("PDF support requires additional libraries. Please copy-paste text for now.")
#             return None
#         else:
#             st.warning("Unsupported file format. Please use .txt files or copy-paste text.")
#             return None
#     except Exception as e:
#         st.error(f"Error reading file: {str(e)}")
#         return None

# def extract_sources_from_response(response_text, tools_used):
#     """Extract and format sources from agent response"""
#     sources = []
#     documents = []
    
#     # Parse the response to extract information from different tools
#     if "Internet_Search" in response_text or "search" in str(tools_used).lower():
#         sources.append(("Web Search", "Information from web search results"))
#         documents.append(Document(
#             page_content=response_text, 
#             metadata={"source": "Web Search"}
#         ))
    
#     if "Wikipedia" in response_text or "wikipedia" in str(tools_used).lower():
#         sources.append(("Wikipedia", "Information from Wikipedia"))
#         documents.append(Document(
#             page_content=response_text, 
#             metadata={"source": "Wikipedia"}
#         ))
        
#     if "Arxiv" in response_text or "arxiv" in str(tools_used).lower():
#         sources.append(("Arxiv", "Information from academic papers"))
#         documents.append(Document(
#             page_content=response_text, 
#             metadata={"source": "Arxiv"}
#         ))
    
#     return sources, documents
#     """Extract and format sources from agent response"""
#     sources = []
#     documents = []
    
#     # Parse the response to extract information from different tools
#     if "Internet_Search" in response_text or "search" in str(tools_used).lower():
#         sources.append(("Web Search", "Information from web search results"))
#         documents.append(Document(
#             page_content=response_text, 
#             metadata={"source": "Web Search"}
#         ))
    
#     if "Wikipedia" in response_text or "wikipedia" in str(tools_used).lower():
#         sources.append(("Wikipedia", "Information from Wikipedia"))
#         documents.append(Document(
#             page_content=response_text, 
#             metadata={"source": "Wikipedia"}
#         ))
        
#     if "Arxiv" in response_text or "arxiv" in str(tools_used).lower():
#         sources.append(("Arxiv", "Information from academic papers"))
#         documents.append(Document(
#             page_content=response_text, 
#             metadata={"source": "Arxiv"}
#         ))
    
#     return sources, documents

# def perform_intelligent_search(query, components):
#     """Perform intelligent search using the agent and return summarized results"""
#     try:
#         agent_executor = components['agent_executor']
#         embeddings = components['embeddings']
#         text_splitter = components['text_splitter']
#         llm = components['llm']
        
#         # Execute agent to gather information
#         with st.spinner("🔍 Searching across multiple sources..."):
#             agent_response = agent_executor.invoke({"input": query})
            
#         raw_output = agent_response.get("output", "")
        
#         # Extract sources and create documents
#         sources, documents = extract_sources_from_response(
#             raw_output, 
#             agent_response.get("intermediate_steps", [])
#         )
        
#         # Create enhanced context for summarization
#         context = raw_output
        
#         # If we have documents, create vector store for better context
#         if documents:
#             with st.spinner("📊 Processing and analyzing information..."):
#                 chunks = text_splitter.split_documents(documents)
#                 if chunks:
#                     vectorstore = FAISS.from_documents(chunks, embeddings)
#                     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#                     relevant_docs = retriever.get_relevant_documents(query)
                    
#                     context = "\n\n".join([
#                         f"Source: {doc.metadata.get('source', 'Unknown')}\n"
#                         f"Content: {doc.page_content[:1000]}..."
#                         for doc in relevant_docs
#                     ])
        
#         # Generate AI-based summary
#         with st.spinner("🤖 Generating intelligent summary..."):
#             summarization_prompt = create_summarization_prompt()
#             summary_input = summarization_prompt.format(context=context, query=query)
#             summary = llm.invoke(summary_input)
        
#         return {
#             "summary": summary,
#             "raw_context": context,
#             "sources": sources,
#             "agent_response": raw_output
#         }
        
#     except Exception as e:
#         logger.error(f"Error in intelligent search: {str(e)}")
#         return {
#             "summary": f"An error occurred while processing your request: {str(e)}",
#             "raw_context": "",
#             "sources": [],
#             "agent_response": ""
#         }

# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🔍 Intelligent Search Engine</h1>', unsafe_allow_html=True)
#     st.markdown('<p style="text-align: center; color: #6c757d;">Powered by Ollama AI with Multi-Source Search & Advanced Summarization</p>', unsafe_allow_html=True)
    
#     # Main Navigation Tabs
#     tab1, tab2 = st.tabs(["🔍 Intelligent Search", "📝 AI Text Summarizer"])
    
#     # Sidebar
#     with st.sidebar:
#         # st.header("🛠️ Configuration")
        
#         # # Model selection
#         # model_name = st.selectbox(
#         #     "Select Ollama Model",
#         #     ["llama3.2:latest", "mistral:latest", "codellama:latest", "llama2:latest"],
#         #     help="Choose your preferred Ollama model"
#         # )
        
#         # # Search options
#         # st.header("🔍 Search Options")
#         # max_iterations = st.slider("Max Search Iterations", 1, 10, 5)
#         # temperature = st.slider("AI Temperature", 0.0, 1.0, 0.5, 0.1)
        
#         # Tool information
#         st.header("🧰 Available Tools")
#         with st.expander("View Tool Details"):
#             st.markdown("""
#             **Internet Search** 📰
#             - Current news and web content
#             - Real-time information
            
#             **Wikipedia** 📚
#             - Factual information
#             - Historical events & biographies
            
#             **Arxiv** 🔬
#             - Academic papers
#             - Scientific research
            
#             **AI Summarizer** 🤖
#             - Text summarization
#             - Multiple summary types
#             - Document processing
#             """)
    
#     # Initialize components
#     if 'components' not in st.session_state:
#         with st.spinner("🚀 Initializing AI components..."):
#             st.session_state.components = initialize_components()
    
#     if not st.session_state.components:
#         st.error("Failed to initialize components. Please check your setup.")
#         return
    
#     # Tab 1: Intelligent Search
#     with tab1:
#         st.header("🔍 Multi-Source Intelligent Search")
        
#         # Search interface
#         st.markdown('<div class="search-box">', unsafe_allow_html=True)
#         query = st.text_input(
#             "Enter your search query:",
#             placeholder="e.g., 'Latest developments in artificial intelligence' or 'Quantum computing research papers'",
#             help="Ask anything! The AI will search across web, Wikipedia, and academic papers.",
#             key="search_query"
#         )
        
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Search execution
#         if search_button and query:
#             start_time = time.time()
            
#             # Perform search
#             results = perform_intelligent_search(query, st.session_state.components)
            
#             search_time = time.time() - start_time
            
#             # Display results
#             st.markdown("---")
            
#             # Search metrics
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Search Time", f"{search_time:.2f}s")
#             with col2:
#                 st.metric("Sources Found", len(results['sources']))
#             with col3:
#                 st.metric("Status", "✅ Complete")
            
#             # AI Summary
#             st.markdown('<div class="result-container">', unsafe_allow_html=True)
#             st.subheader("🤖 AI-Generated Summary")
#             st.markdown(results['summary'])
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             # Source Information
#             if results['sources']:
#                 st.subheader("📋 Sources Used")
#                 for i, (source_type, source_desc) in enumerate(results['sources'], 1):
#                     with st.expander(f"📌 {source_type}"):
#                         st.write(source_desc)
            
#             # Raw Context (expandable)
#             with st.expander("🔍 View Raw Research Data"):
#                 st.text_area("Raw Context", results['raw_context'], height=300, key="raw_context")
            
#             # Agent Response (for debugging)
#             with st.expander("🤖 View Agent Response"):
#                 st.code(results['agent_response'], language="text")
        
#         elif search_button and not query:
#             st.warning("⚠️ Please enter a search query.")
    
#     # Tab 2: AI Text Summarizer
#     with tab2:
#         st.header("📝 AI-Powered Text Summarization")
        
#         # Summarization options
#         col1, col2 = st.columns(2)
        
#         with col1:
#             summary_type = st.selectbox(
#                 "Summary Type",
#                 ["Abstractive Summary", "Extractive Summary", "Bullet Points", "Executive Summary"],
#                 help="Choose the type of summary you want"
#             )
        
#         with col2:
#             summary_length = st.selectbox(
#                 "Summary Length",
#                 ["Short (1-2 paragraphs)", "Medium (3-4 paragraphs)", "Long (5+ paragraphs)"],
#                 index=1,
#                 help="Select desired summary length"
#             )
        
#         # Input methods
#         st.subheader("📄 Input Your Text")
        
#         input_method = st.radio(
#             "Choose input method:",
#             ["Type/Paste Text", "Upload File"],
#             horizontal=True
#         )
        
#         text_to_summarize = ""
        
#         if input_method == "Type/Paste Text":
#             text_to_summarize = st.text_area(
#                 "Enter text to summarize:",
#                 height=200,
#                 placeholder="Paste your text here...",
#                 help="Paste any text you want to summarize",
#                 key="text_input"
#             )
#         else:
#             uploaded_file = st.file_uploader(
#                 "Upload a text file",
#                 type=['txt'],
#                 help="Upload a .txt file to summarize"
#             )
            
#             if uploaded_file is not None:
#                 text_to_summarize = extract_text_from_uploaded_file(uploaded_file)
#                 if text_to_summarize:
#                     with st.expander("📄 Preview Uploaded Text"):
#                         st.text_area("File Content", text_to_summarize[:1000] + "...", height=150, disabled=True)
        
#         # Summarize button
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             summarize_button = st.button("📝 Generate Summary", type="primary", use_container_width=True)
        
#         # Summarization execution
#         if summarize_button and text_to_summarize:
#             if len(text_to_summarize.strip()) < 50:
#                 st.warning("⚠️ Please provide more text (at least 50 characters) for meaningful summarization.")
#             else:
#                 start_time = time.time()
                
#                 with st.spinner("🤖 Generating AI summary..."):
#                     summary_results = summarize_text(
#                         text_to_summarize, 
#                         summary_type, 
#                         summary_length, 
#                         st.session_state.components['llm']
#                     )
                
#                 summarization_time = time.time() - start_time
                
#                 # Display results
#                 st.markdown("---")
                
#                 # Summary metrics
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Processing Time", f"{summarization_time:.2f}s")
#                 with col2:
#                     st.metric("Original Words", summary_results['original_words'])
#                 with col3:
#                     st.metric("Summary Words", summary_results['summary_words'])
#                 with col4:
#                     st.metric("Compression", f"{summary_results['compression_ratio']}%")
                
#                 # Generated Summary
#                 st.markdown('<div class="result-container">', unsafe_allow_html=True)
#                 st.subheader(f"📋 {summary_type}")
#                 st.markdown(summary_results['summary'])
#                 st.markdown('</div>', unsafe_allow_html=True)
                
#                 # Download option
#                 st.download_button(
#                     label="💾 Download Summary",
#                     data=summary_results['summary'],
#                     file_name=f"summary_{int(time.time())}.txt",
#                     mime="text/plain"
#                 )
                
#                 # Original text preview (expandable)
#                 with st.expander("📄 View Original Text"):
#                     st.text_area("Original Text", text_to_summarize, height=300, disabled=True, key="original_text")
        
#         elif summarize_button and not text_to_summarize:
#             st.warning("⚠️ Please provide text to summarize.")
        
#         # Example texts for demonstration
#         st.markdown("---")
#         st.subheader("🎯 Try Sample Texts")
        
#         sample_texts = {
#             "Scientific Article": "Artificial intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, revolutionizing industries from healthcare to transportation. Machine learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed for every task. Deep learning, which uses neural networks with multiple layers, has achieved remarkable breakthroughs in image recognition, natural language processing, and game-playing. The applications of AI are vast and growing rapidly. In healthcare, AI systems can analyze medical images to detect diseases earlier than human doctors. In autonomous vehicles, AI processes sensor data to navigate safely through complex traffic scenarios. Natural language processing allows AI to understand and generate human-like text, enabling chatbots and virtual assistants. However, the rise of AI also presents significant challenges. Concerns about job displacement, privacy, algorithmic bias, and the potential for misuse have sparked important debates about AI governance and ethics. As AI becomes more sophisticated and ubiquitous, society must grapple with questions about accountability, transparency, and the appropriate limits of artificial intelligence.",
            
#             "Business Report": "The quarterly financial results show significant growth across all major business segments. Revenue increased by 23% compared to the same period last year, reaching $4.2 billion. The technology division led growth with a 35% increase, driven by strong demand for cloud services and artificial intelligence solutions. Operating margins improved to 18.5%, up from 15.2% in the previous quarter, demonstrating successful cost optimization initiatives. Customer acquisition increased by 28%, with particular strength in the enterprise segment. International markets contributed 42% of total revenue, showing the company's successful global expansion strategy. Looking ahead, management expects continued growth momentum, with projected revenue growth of 20-25% for the full year. Key investment areas include research and development, strategic acquisitions, and expansion into emerging markets. Risk factors include increased competition, regulatory changes, and potential economic uncertainties that could impact customer spending patterns.",
            
#             "News Article": "Breaking developments in climate change research reveal accelerating ice sheet melting in Antarctica and Greenland. Scientists from multiple international institutions report that ice loss has tripled over the past decade, contributing to faster sea level rise than previously projected. The study, published in Nature Climate Change, analyzed satellite data spanning 30 years and found that warming ocean temperatures are causing ice sheets to melt from below at unprecedented rates. Coastal communities worldwide face increasing risks from flooding and storm surges. The research indicates that even if global warming is limited to 1.5 degrees Celsius above pre-industrial levels, significant sea level rise is now unavoidable. Policymakers are calling for immediate action to strengthen coastal defenses and implement more aggressive carbon reduction strategies. The findings underscore the urgency of international climate commitments and the need for rapid deployment of renewable energy technologies."
#         }
        
#         selected_sample = st.selectbox("Choose a sample text:", ["Select a sample..."] + list(sample_texts.keys()))
        
#         if selected_sample != "Select a sample...":
#             if st.button(f"📋 Use {selected_sample} Sample"):
#                 st.session_state.sample_text = sample_texts[selected_sample]
#                 st.rerun()
        
#         if 'sample_text' in st.session_state:
#             st.text_area("Sample Text Loaded:", st.session_state.sample_text, height=150, key="sample_display")
#             if st.button("🔄 Clear Sample"):
#                 del st.session_state.sample_text
#                 st.rerun()
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         '<p style="text-align: center; color: #6c757d;">Built with Streamlit, LangChain, and Ollama | '
#         'Multi-source intelligent search with AI summarization</p>',
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()