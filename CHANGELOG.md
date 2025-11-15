# Changelog

All notable changes to the Hierarchical RAG for FDA Regulatory Documents project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-XX

### Added

#### Core Architecture
- **Initial Hierarchical RAG Implementation**
  - Created two-step retrieval system: route queries to domains, then retrieve only from that domain
  - Implemented domain router using keyword-based heuristics (placeholder for future classifier)
  - Built ChromaDB vector store with domain-based metadata filtering
  - Created RAG pipeline with retrieval and generation logic

#### Document Processing
- **Text Extraction Support**
  - Added PDF text extraction using PyMuPDF (`fitz`)
  - Added DOCX text extraction using python-docx
  - Added TXT file support with multiple encoding detection (UTF-8, latin-1, cp1252, iso-8859-1)
  - Implemented automatic encoding fallback for text files

- **Document Chunking**
  - Created intelligent text chunking (~500-1000 token segments)
  - Implemented overlap between chunks (100 tokens default)
  - Added sentence boundary detection for better chunk quality

- **Domain Classification**
  - Implemented 7 regulatory domains: Clinical, Nonclinical, CMC, Compliance, Quality Systems, Submission Logistics, Labeling
  - Added domain assignment to each chunk during ingestion
  - Created domain statistics tracking

#### Document Hierarchy Classification
- **7-Level Hierarchy System**
  - Implemented hierarchy classification with 7 levels:
    1. Acts and Laws
    2. Administrative Law / Regulations / Directives / Code of Federal Regulations (CFR)
    3. Pharmacopoeia Standards
    4. Authority Manual / Procedures
    5. Guidances / Guidelines / Q&As
    6. Position Paper
    7. White Papers
  - Created keyword-based classifier with pattern matching
  - Added confidence scoring for hierarchy classification
  - Implemented user input workflow for unclear classifications

#### Document Summarization
- **Multi-Method Summarization**
  - Added OpenAI GPT-based summarization (default)
  - Added Ollama local LLM summarization support
  - Implemented extractive summarization as fallback
  - Automatic fallback to extractive method on API failures
  - Configurable summary length (default: 200 words)

#### User Interface
- **Streamlit Application**
  - Created main application with sidebar and main content area
  - Added document upload interface with multiple file support
  - Implemented "Classify & Ingest" workflow
  - Created query interface with routing visualization
  - Added document organization by hierarchy level with tabs
  - Implemented expandable document cards with summaries

- **Configuration Panel**
  - Embedding model selection (local vs OpenAI)
  - LLM provider selection (OpenAI vs Ollama)
  - API key input fields with password masking
  - Component reinitialization button

- **Statistics Dashboard**
  - Total chunks counter
  - Document count tracker
  - Hierarchy level distribution
  - Domain distribution
  - Processed documents list

#### Multiple File Upload
- **Batch Processing**
  - Added support for uploading multiple files simultaneously
  - Implemented batch classification workflow
  - Created user input collection for unclear classifications
  - Added processing results summary with expandable summaries

#### API Error Handling & Validation
- **API Validator Module** (`utils/api_validator.py`)
  - Created `APIValidator` class for API key validation
  - Implemented `validate_openai_key()` method with test API call
  - Added `handle_openai_error()` for user-friendly error messages
  - Created error type detection (quota, rate limit, authentication, etc.)
  - Added helper methods: `is_quota_error()`, `is_rate_limit_error()`

- **Error Handling Throughout Application**
  - Wrapped all OpenAI API calls in try-except blocks
  - Added specific error handling for:
    - Quota exceeded errors (429 insufficient_quota)
    - Rate limit errors (429 rate_limit_exceeded)
    - Authentication errors (invalid_api_key)
    - Connection errors
    - Timeout errors
  - Implemented graceful fallbacks:
    - Summarization: Falls back to extractive method
    - RAG generation: Returns helpful error message
    - Embeddings: Raises clear error with guidance

- **UI Error Handling**
  - Added "Validate API Key" button in configuration
  - Added "Validate LLM API Key" button for LLM provider
  - Implemented clear error messages with emoji indicators
  - Added helpful tips for resolving errors (switch to local models)
  - Prevented infinite loops by catching errors immediately

#### Vector Store Enhancements
- **Hierarchy Support**
  - Added `get_hierarchy_stats()` method for hierarchy-level statistics
  - Created `get_documents_by_hierarchy()` for filtering by hierarchy level
  - Stored hierarchy level in chunk metadata
  - Enabled hierarchy-based document organization

- **Domain Filtering**
  - Implemented `search_by_domain()` for domain-specific retrieval
  - Added `search_all()` as fallback when domain filtering returns no results
  - Created domain statistics tracking

#### Session State Management
- **State Persistence**
  - Document summaries stored in session state
  - Processed documents list tracking
  - Component initialization state management
  - Configuration persistence across interactions

### Changed

#### Document Processing Workflow
- **Enhanced `process_document()` Function**
  - Modified to return dictionary with chunks, hierarchy level, and classification result
  - Added hierarchy classifier parameter
  - Added optional pre-assigned hierarchy level parameter
  - Updated metadata to include hierarchy_level field

#### Error Messages
- **Improved User Feedback**
  - Replaced technical error messages with user-friendly explanations
  - Added actionable guidance (e.g., "Switch to local models")
  - Included emoji indicators for quick error type identification
  - Added context about what operation failed

### Technical Details

#### File Structure
```
.
├── app.py                      # Main Streamlit UI (606 lines)
├── utils/
│   ├── extract_text.py        # PDF/DOCX/TXT extraction
│   ├── router.py              # Domain router (keyword-based)
│   ├── document_classifier.py # Hierarchy level classifier
│   ├── summarizer.py          # Document summarization
│   ├── ingest.py              # Chunking + domain/hierarchy assignment
│   ├── vectorstore.py         # ChromaDB with domain filtering
│   ├── rag.py                 # Hierarchical RAG pipeline
│   └── api_validator.py       # API validation & error handling
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

#### Dependencies
- `streamlit>=1.28.0` - Web application framework
- `chromadb>=0.4.15` - Vector database
- `pymupdf>=1.23.0` - PDF text extraction
- `python-docx>=1.1.0` - DOCX text extraction
- `sentence-transformers>=2.2.2` - Local embeddings
- `openai>=0.28.0` - OpenAI API client
- `requests>=2.31.0` - HTTP requests for Ollama
- `numpy>=1.24.0` - Numerical operations

#### Configuration Options
- **Embedding Models:**
  - Local: `all-MiniLM-L6-v2` (default, no API key needed)
  - OpenAI: `text-embedding-ada-002` (requires API key)

- **LLM Providers:**
  - OpenAI: GPT-3.5-turbo (requires API key)
  - Ollama: Local models like llama2, mistral (requires Ollama running)

- **Storage:**
  - ChromaDB: Persistent local storage in `./chroma_db/` directory

### Security & Best Practices

- API keys stored in session state (not persisted to disk)
- Password-masked input fields for API keys
- Error messages don't expose sensitive information
- Graceful degradation when APIs fail
- No infinite retry loops

### Known Limitations

- Router uses keyword-based heuristics (can be replaced with trained classifier)
- Hierarchy classifier uses simple keyword matching (can be enhanced)
- Extractive summarization is basic (first/last sentences)
- Session state summaries cleared on page refresh
- No persistent storage for document summaries (only in session)

### Future Enhancements

- Fine-tuned domain router classifier
- Enhanced hierarchy classification with ML models
- Hybrid retrieval (BM25 + embeddings)
- Subdomain-level routing
- Multi-domain queries
- Citation tracking
- Document versioning
- Advanced metadata filtering
- Persistent summary storage
- Batch API call optimization
- Rate limiting and retry logic with exponential backoff

## [Unreleased]

### Planned
- Integration with fine-tuned classification models
- Enhanced summarization with abstractive methods
- Document similarity search across hierarchy levels
- Export/import functionality for knowledge base
- User authentication and multi-user support
- Advanced analytics and reporting

---

## Version History

- **v1.0.0** - Initial release with full hierarchical RAG functionality
  - Core RAG pipeline
  - Document hierarchy classification
  - Multiple file upload
  - API error handling
  - Document summarization
  - UI with organization by hierarchy level

---

## Notes

- All changes maintain backward compatibility with existing ChromaDB databases
- Error handling is designed to prevent data loss
- Fallback mechanisms ensure system continues functioning even with API failures
- The system is designed to be easily extensible for future enhancements

