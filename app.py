"""
Streamlit Hierarchical RAG Application for FDA Regulatory Documents
Main UI and orchestration logic for the hierarchical RAG system.
"""

import streamlit as st
import os
import uuid
from pathlib import Path
import tempfile
from typing import Optional

from utils.extract_text import extract_text_from_file
from utils.router import DomainRouter
from utils.document_classifier import DocumentHierarchyClassifier, HIERARCHY_LEVELS
from utils.summarizer import DocumentSummarizer
from utils.ingest import process_document
from utils.vectorstore import HierarchicalVectorStore
from utils.rag import HierarchicalRAG
from utils.api_validator import APIValidator


# Page configuration
st.set_page_config(
    page_title="Hierarchical RAG - FDA Regulatory Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'router' not in st.session_state:
    st.session_state.router = None
if 'hierarchy_classifier' not in st.session_state:
    st.session_state.hierarchy_classifier = None
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = []
if 'document_summaries' not in st.session_state:
    st.session_state.document_summaries = {}  # file_id -> summary


def initialize_components():
    """Initialize vector store, router, and RAG pipeline."""
    if st.session_state.vector_store is None:
        # Get configuration from session state or use defaults
        embedding_type = st.session_state.get('embedding_type', 'local')
        llm_type = st.session_state.get('llm_type', 'openai')
        
        openai_key = st.session_state.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        
        try:
            # Initialize router
            st.session_state.router = DomainRouter()
            
            # Initialize hierarchy classifier
            st.session_state.hierarchy_classifier = DocumentHierarchyClassifier()
            
            # Initialize summarizer
            st.session_state.summarizer = DocumentSummarizer(
                llm_type=llm_type,
                openai_api_key=openai_key,
                ollama_model=st.session_state.get('ollama_model', 'llama2')
            )
            
            # Initialize vector store
            st.session_state.vector_store = HierarchicalVectorStore(
                embedding_type=embedding_type,
                openai_api_key=openai_key,
                model_name=st.session_state.get('local_model_name', 'all-MiniLM-L6-v2')
            )
            
            # Initialize RAG pipeline
            st.session_state.rag_pipeline = HierarchicalRAG(
                vector_store=st.session_state.vector_store,
                router=st.session_state.router,
                llm_type=llm_type,
                openai_api_key=openai_key,
                ollama_model=st.session_state.get('ollama_model', 'llama2')
            )
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return False
    
    return True


# Sidebar for document upload and configuration
with st.sidebar:
    st.header("📚 Document Management")
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=False):
        embedding_type = st.selectbox(
            "Embedding Model",
            ["local", "openai"],
            index=0,
            help="Choose between local sentence-transformers or OpenAI embeddings"
        )
        st.session_state.embedding_type = embedding_type
        
        if embedding_type == "openai":
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get('openai_api_key', ''),
                help="Enter your OpenAI API key"
            )
            st.session_state.openai_api_key = openai_key
            
            # API validation button
            if st.button("🔍 Validate API Key", help="Test if your API key is valid and has quota"):
                if openai_key:
                    with st.spinner("Validating API key..."):
                        is_valid, error_msg = APIValidator.validate_openai_key(openai_key)
                        if is_valid:
                            st.success("✅ API key is valid and working!")
                        else:
                            st.error(f"❌ {error_msg}")
                            if "quota" in error_msg.lower():
                                st.warning("💡 Tip: Switch to local embeddings (sentence-transformers) to avoid quota issues.")
                else:
                    st.warning("Please enter an API key first.")
        else:
            local_model = st.text_input(
                "Local Model Name",
                value=st.session_state.get('local_model_name', 'all-MiniLM-L6-v2'),
                help="Sentence-transformers model name"
            )
            st.session_state.local_model_name = local_model
        
        llm_type = st.selectbox(
            "LLM Provider",
            ["openai", "ollama"],
            index=0,
            help="Choose between OpenAI GPT or local Ollama"
        )
        st.session_state.llm_type = llm_type
        
        if llm_type == "openai":
            llm_openai_key = st.text_input(
                "OpenAI API Key (for LLM)",
                type="password",
                value=st.session_state.get('openai_api_key', ''),
                help="Enter your OpenAI API key for LLM (can be same as embedding key)"
            )
            if llm_openai_key:
                st.session_state.openai_api_key = llm_openai_key
            
            # API validation button for LLM
            if st.button("🔍 Validate LLM API Key", help="Test if your LLM API key is valid and has quota"):
                if llm_openai_key:
                    with st.spinner("Validating API key..."):
                        is_valid, error_msg = APIValidator.validate_openai_key(llm_openai_key)
                        if is_valid:
                            st.success("✅ LLM API key is valid and working!")
                        else:
                            st.error(f"❌ {error_msg}")
                            if "quota" in error_msg.lower():
                                st.warning("💡 Tip: Switch to Ollama (local LLM) to avoid quota issues.")
                else:
                    st.warning("Please enter an API key first.")
        
        if llm_type == "ollama":
            ollama_model = st.text_input(
                "Ollama Model Name",
                value=st.session_state.get('ollama_model', 'llama2'),
                help="Ollama model name (e.g., llama2, mistral)"
            )
            st.session_state.ollama_model = ollama_model
        
        if st.button("🔄 Reinitialize Components"):
            st.session_state.vector_store = None
            st.session_state.rag_pipeline = None
            st.session_state.router = None
            st.rerun()
    
    st.divider()
    
    # Document upload - Multiple files
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files (multiple files supported)",
        type=['pdf', 'docx', 'doc', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more PDF, DOCX, or TXT files to add to the knowledge base"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        # Show file info
        st.info(f"📄 **{len(uploaded_files)} file(s) selected**")
        for file in uploaded_files:
            st.write(f"  • {file.name} ({file.size / 1024:.2f} KB)")
        
        # Classify & Ingest button
        if st.button("🏷️ Classify & Ingest All", type="primary"):
            if not initialize_components():
                st.error("Failed to initialize components. Check your configuration.")
            else:
                # First pass: classify all files and collect those needing user input
                files_needing_input = {}
                files_ready = {}
                
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract text
                        extracted_text = extract_text_from_file(tmp_path)
                        
                        if not extracted_text:
                            st.warning(f"⚠️ Could not extract text from '{uploaded_file.name}'")
                            os.unlink(tmp_path)
                            continue
                        
                        # Classify hierarchy level
                        hierarchy_result = st.session_state.hierarchy_classifier.classify_document(
                            extracted_text, uploaded_file.name
                        )
                        
                        hierarchy_level = hierarchy_result.get("level")
                        needs_user_input = hierarchy_result.get("needs_user_input", False) or hierarchy_level is None
                        
                        if needs_user_input:
                            files_needing_input[uploaded_file.name] = {
                                "file": uploaded_file,
                                "text": extracted_text,
                                "tmp_path": tmp_path,
                                "classification": hierarchy_result
                            }
                        else:
                            files_ready[uploaded_file.name] = {
                                "file": uploaded_file,
                                "text": extracted_text,
                                "tmp_path": tmp_path,
                                "hierarchy_level": hierarchy_level
                            }
                    
                    except Exception as e:
                        st.error(f"Error processing '{uploaded_file.name}': {str(e)}")
                
                # Handle files needing user input
                if files_needing_input:
                    st.subheader("⚠️ Manual Classification Required")
                    st.write("The following files could not be automatically classified. Please select their hierarchy levels:")
                    
                    user_selections = {}
                    for file_name, file_data in files_needing_input.items():
                        st.write(f"**{file_name}**")
                        
                        # Show classification scores
                        if file_data["classification"].get("scores"):
                            st.write("**Classification Scores:**")
                            for level, scores in sorted(
                                file_data["classification"]["scores"].items(),
                                key=lambda x: x[1]["weighted"],
                                reverse=True
                            )[:3]:
                                st.write(f"  • {level}: {scores['weighted']} points")
                        
                        # User selection
                        selected_level = st.selectbox(
                            f"Select hierarchy level:",
                            HIERARCHY_LEVELS,
                            key=f"hierarchy_select_{file_name}"
                        )
                        user_selections[file_name] = selected_level
                        st.divider()
                    
                    # Update files_ready with user selections
                    for file_name, selected_level in user_selections.items():
                        file_data = files_needing_input[file_name]
                        files_ready[file_name] = {
                            "file": file_data["file"],
                            "text": file_data["text"],
                            "tmp_path": file_data["tmp_path"],
                            "hierarchy_level": selected_level
                        }
                
                # Process all ready files
                if files_ready:
                    processing_results = []
                    
                    for file_name, file_data in files_ready.items():
                        with st.spinner(f"Processing {file_name}..."):
                            try:
                                uploaded_file = file_data["file"]
                                extracted_text = file_data["text"]
                                hierarchy_level = file_data["hierarchy_level"]
                                tmp_path = file_data["tmp_path"]
                                
                                # Generate summary (with error handling)
                                try:
                                    summary = st.session_state.summarizer.summarize(extracted_text, max_length=200)
                                except Exception as e:
                                    error_msg = APIValidator.handle_openai_error(e, "Summarization")
                                    st.warning(f"⚠️ {error_msg}")
                                    # Use extractive summary as fallback
                                    summary = st.session_state.summarizer.extractive_summary(extracted_text, 10)
                                
                                # Process document with hierarchy level
                                file_id = str(uuid.uuid4())
                                result = process_document(
                                    text=extracted_text,
                                    file_name=uploaded_file.name,
                                    file_id=file_id,
                                    router=st.session_state.router,
                                    hierarchy_classifier=st.session_state.hierarchy_classifier,
                                    hierarchy_level=hierarchy_level,
                                    chunk_size=800,
                                    overlap=100
                                )
                                
                                processed_chunks = result["chunks"]
                                
                                # Extract texts, metadatas, and ids
                                texts = [chunk["text"] for chunk in processed_chunks]
                                metadatas = [chunk["metadata"] for chunk in processed_chunks]
                                ids = [chunk["id"] for chunk in processed_chunks]
                                
                                # Add to vector store (with error handling)
                                try:
                                    st.session_state.vector_store.add_documents(
                                        texts=texts,
                                        metadatas=metadatas,
                                        ids=ids
                                    )
                                except Exception as e:
                                    error_msg = APIValidator.handle_openai_error(e, "Embedding generation")
                                    st.error(f"❌ Failed to add documents to vector store: {error_msg}")
                                    if APIValidator.is_quota_error(e):
                                        st.warning("💡 Tip: Switch to local embeddings in the configuration to avoid quota issues.")
                                    raise  # Re-raise to stop processing
                                
                                # Store summary
                                st.session_state.document_summaries[file_id] = {
                                    "summary": summary,
                                    "file_name": uploaded_file.name,
                                    "hierarchy_level": hierarchy_level
                                }
                                
                                # Track processed documents
                                if uploaded_file.name not in st.session_state.documents_processed:
                                    st.session_state.documents_processed.append(uploaded_file.name)
                                
                                processing_results.append({
                                    "file_name": uploaded_file.name,
                                    "file_id": file_id,
                                    "chunks": len(processed_chunks),
                                    "hierarchy_level": hierarchy_level,
                                    "summary": summary
                                })
                                
                                # Clean up temp file
                                os.unlink(tmp_path)
                            
                            except Exception as e:
                                st.error(f"Error processing '{file_name}': {str(e)}")
                    
                    # Show results
                    if processing_results:
                        st.success(f"✅ Successfully processed {len(processing_results)} file(s)!")
                        for result in processing_results:
                            st.write(f"**{result['file_name']}**")
                            st.write(f"  • Hierarchy Level: {result['hierarchy_level']}")
                            st.write(f"  • Chunks: {result['chunks']}")
                            with st.expander(f"View Summary - {result['file_name']}"):
                                st.write(result['summary'])
                        
                        st.rerun()
    
    st.divider()
    
    # Knowledge base stats
    st.subheader("📊 Knowledge Base Stats")
    if st.session_state.vector_store:
        doc_count = st.session_state.vector_store.get_collection_count()
        st.metric("Total Chunks", doc_count)
        st.metric("Documents", len(st.session_state.documents_processed))
        
        # Hierarchy statistics
        hierarchy_stats = st.session_state.vector_store.get_hierarchy_stats()
        if hierarchy_stats:
            st.write("**Chunks by Hierarchy Level:**")
            for level, count in sorted(hierarchy_stats.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  • {level}: {count}")
        
        # Domain statistics
        domain_stats = st.session_state.vector_store.get_domain_stats()
        if domain_stats:
            st.write("**Chunks by Domain:**")
            for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  • {domain}: {count}")
        
        if st.session_state.documents_processed:
            st.write("**Processed Documents:**")
            for doc in st.session_state.documents_processed:
                st.write(f"  • {doc}")
    else:
        st.info("Initialize components to see stats")
    
    # Clear database button
    if st.button("🗑️ Clear Knowledge Base", type="secondary"):
        if st.session_state.vector_store:
            st.session_state.vector_store.clear_collection()
            st.session_state.documents_processed = []
            st.session_state.document_summaries = {}
            st.success("Knowledge base cleared!")
            st.rerun()
        else:
            st.warning("No knowledge base to clear.")


# Main page
st.title("🏥 Hierarchical RAG - FDA Regulatory Assistant")
st.markdown("**Two-Step Hierarchical Retrieval: Route → Retrieve → Generate**")

# Initialize components if not already done
if not initialize_components():
    st.warning("⚠️ Please configure your settings in the sidebar and initialize components.")
else:
    # Documents organized by hierarchy level
    if st.session_state.vector_store and st.session_state.vector_store.get_collection_count() > 0:
        st.header("📚 Documents by Hierarchy Level")
        
        hierarchy_stats = st.session_state.vector_store.get_hierarchy_stats()
        
        if hierarchy_stats:
            # Create tabs for each hierarchy level
            hierarchy_levels_with_docs = [level for level in HIERARCHY_LEVELS if level in hierarchy_stats]
            
            if hierarchy_levels_with_docs:
                tabs = st.tabs([f"{level} ({hierarchy_stats.get(level, 0)} chunks)" for level in hierarchy_levels_with_docs])
                
                for tab, hierarchy_level in zip(tabs, hierarchy_levels_with_docs):
                    with tab:
                        documents = st.session_state.vector_store.get_documents_by_hierarchy(hierarchy_level)
                        
                        if documents:
                            st.write(f"**{len(documents)} document(s) in this hierarchy level**")
                            
                            for doc_info in documents:
                                file_id = doc_info['file_id']
                                file_name = doc_info['file_name']
                                chunk_count = doc_info['chunk_count']
                                
                                with st.expander(f"📄 {file_name} ({chunk_count} chunks)", expanded=False):
                                    # Show summary if available
                                    if file_id in st.session_state.document_summaries:
                                        summary_info = st.session_state.document_summaries[file_id]
                                        st.write("**Summary:**")
                                        st.write(summary_info['summary'])
                                        st.divider()
                                    
                                    # Show document info
                                    st.write(f"**File ID:** {file_id}")
                                    st.write(f"**Hierarchy Level:** {hierarchy_level}")
                                    st.write(f"**Chunks:** {chunk_count}")
                        else:
                            st.info("No documents found in this hierarchy level.")
            else:
                st.info("No documents have been classified yet.")
        else:
            st.info("No documents in the knowledge base yet.")
        
        st.divider()
    
    # Query interface
    st.header("💬 Ask a Question")
    
    user_question = st.text_area(
        "Enter your regulatory question:",
        height=100,
        placeholder="e.g., What are the requirements for clinical trial protocols?",
        help="Type your question and press Submit to search the knowledge base"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔍 Submit", type="primary")
    
    if submit_button and user_question:
        with st.spinner("Routing query, retrieving context, and generating response..."):
            try:
                result = st.session_state.rag_pipeline.query(user_question, top_k=5)
                
                # Display routing result
                st.header("🎯 Routing Result")
                routing = result['routing_result']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Domain", routing['domain'])
                with col2:
                    st.metric("Confidence", f"{routing['confidence']:.2%}")
                
                # Show all domain scores
                with st.expander("View all domain scores", expanded=False):
                    if 'scores' in routing:
                        for domain, scores in routing['scores'].items():
                            st.write(f"**{domain}**: {scores['count']} matches, weighted score: {scores['weighted']}")
                
                # Display answer
                st.header("📝 Answer")
                st.write(result['answer'])
                
                # Display retrieved context
                if result['retrieved_chunks']:
                    st.header("📚 Retrieved Context")
                    st.caption(f"Retrieved {len(result['retrieved_chunks'])} chunks from domain: **{result['domain']}**")
                    
                    with st.expander("View retrieved chunks", expanded=False):
                        for i, chunk in enumerate(result['retrieved_chunks'], 1):
                            metadata = chunk.get('metadata', {})
                            st.markdown(f"**Chunk {i}**")
                            st.caption(
                                f"Source: {metadata.get('file_name', 'Unknown')} | "
                                f"Domain: {metadata.get('domain', 'Unknown')} | "
                                f"Chunk Index: {metadata.get('chunk_index', '?')}"
                            )
                            if chunk.get('distance') is not None:
                                st.caption(f"Similarity distance: {chunk['distance']:.4f}")
                            st.text_area(
                                f"Content {i}",
                                chunk.get('text', ''),
                                height=150,
                                disabled=True,
                                key=f"chunk_{i}"
                            )
                            st.divider()
                else:
                    st.info("No relevant context was retrieved from the knowledge base.")
            
            except Exception as e:
                error_str = str(e)
                # Check if it's an API error
                if "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
                    error_msg = APIValidator.handle_openai_error(e, "Query processing")
                    st.error(error_msg)
                    st.warning("💡 **Solution:** Switch to local models (Ollama) or local embeddings in the configuration sidebar.")
                elif "API" in error_str or "api" in error_str:
                    error_msg = APIValidator.handle_openai_error(e, "Query processing")
                    st.error(error_msg)
                else:
                    st.error(f"Error processing query: {error_str}")
    
    elif submit_button:
        st.warning("Please enter a question first.")
    
    # Instructions
    with st.expander("ℹ️ How Hierarchical RAG Works", expanded=False):
        st.markdown("""
        ### Two-Step Hierarchical Retrieval Process
        
        1. **Routing Step**: The router analyzes your question and predicts which domain it belongs to
           (e.g., Clinical, Nonclinical, CMC, Compliance, etc.)
        
        2. **Retrieval Step**: The system retrieves relevant chunks ONLY from the predicted domain,
           ensuring domain-specific context
        
        3. **Generation Step**: The LLM generates an answer using the domain-filtered context
        
        ### Benefits
        
        - **Focused Retrieval**: Only searches within relevant domain, reducing noise
        - **Domain Awareness**: System understands regulatory document structure
        - **Scalability**: Can handle large document collections efficiently
        
        ### Getting Started
        
        1. **Configure Settings** (Sidebar):
           - Choose embedding model (local or OpenAI)
           - Select LLM provider (OpenAI or Ollama)
           - Enter API keys if needed
        
        2. **Add Documents**:
           - Upload PDF or DOCX files
           - Click "Classify & Ingest" to process
           - System automatically assigns domains to chunks
        
        3. **Ask Questions**:
           - Type your regulatory question
           - System routes to domain, retrieves, and answers
           - View routing result and retrieved context
        
        ### Domain Hierarchy
        
        The system uses predefined domains (placeholders - can be modified):
        - Clinical
        - Nonclinical
        - CMC (Chemistry, Manufacturing, Controls)
        - Compliance
        - Quality Systems
        - Submission Logistics
        - Labeling
        
        Each domain can have subdomains for finer granularity.
        """)

