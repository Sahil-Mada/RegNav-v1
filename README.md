# Hierarchical RAG for FDA Regulatory Documents

A Hierarchical RAG (H-RAG) system for FDA regulatory documents that routes queries to specific domains and retrieves context only from those domains. Documents are automatically classified into 7 regulatory hierarchy levels and summarized for easy review.

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys (optional, for OpenAI features):**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or enter it in the app's configuration sidebar.

3. **Set up Ollama (optional, for local LLM):**
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull a model:
   ollama pull llama2
   ```

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **The app will open in your browser** (usually at `http://localhost:8501`)

## Step-by-Step Usage Guide

### Step 1: Configure Settings (Sidebar)

1. Open the **Configuration** section in the sidebar
2. **Choose Embedding Model:**
   - **Local**: Uses sentence-transformers (default: `all-MiniLM-L6-v2`) - no API key needed
   - **OpenAI**: Uses `text-embedding-ada-002` - requires OpenAI API key
3. **Select LLM Provider:**
   - **OpenAI**: Uses GPT-3.5-turbo - requires API key
   - **Ollama**: Uses local Ollama models - requires Ollama running locally
4. Enter API keys if using OpenAI features
5. Click **"🔄 Reinitialize Components"** if you change settings

### Step 2: Upload Documents

1. In the sidebar, under **"Upload Documents"**
2. Click **"Choose PDF, DOCX, or TXT files"**
3. **Select one or more files** (multiple files supported)
4. Files will be listed with their sizes

### Step 3: Classify & Ingest Documents

1. Click **"🏷️ Classify & Ingest All"** button
2. **The system will:**
   - Extract text from each file
   - Classify each document into one of 7 hierarchy levels
   - Generate a summary for each document
   - Chunk the text and assign domain labels
   - Store everything in the vector database

3. **If classification is unclear:**
   - You'll see a warning message
   - Classification scores will be shown
   - Select the correct hierarchy level from the dropdown for each file
   - The system will continue processing

4. **After processing:**
   - You'll see a success message
   - Each file's hierarchy level and chunk count
   - Expandable summaries for each document

### Step 4: View Documents by Hierarchy Level

1. On the main page, scroll to **"📚 Documents by Hierarchy Level"**
2. **Documents are organized in tabs** by their hierarchy level:
   - Acts and Laws
   - Administrative Law / Regulations / Directives / Code of Federal Regulations (CFR)
   - Pharmacopoeia Standards
   - Authority Manual / Procedures
   - Guidances / Guidelines / Q&As
   - Position Paper
   - White Papers
3. **Click on each tab** to see documents in that level
4. **Expand document cards** to view:
   - Document summary
   - File ID
   - Hierarchy level
   - Number of chunks

### Step 5: Ask Questions

1. In the main page, find **"💬 Ask a Question"** section
2. **Type your regulatory question** in the text box
3. Click **"🔍 Submit"**
4. **The system will:**
   - Route your query to the appropriate domain
   - Retrieve relevant chunks only from that domain
   - Generate an answer using the retrieved context
5. **View results:**
   - **Routing Result**: Shows predicted domain and confidence
   - **Answer**: Generated response based on retrieved context
   - **Retrieved Context**: Expandable view of source chunks

### Step 6: Monitor Knowledge Base (Sidebar)

1. Check **"📊 Knowledge Base Stats"** in the sidebar:
   - Total chunks in database
   - Number of processed documents
   - Chunks organized by hierarchy level
   - Chunks organized by domain
   - List of all processed documents
2. **Clear Knowledge Base** if needed (removes all data)

## How It Works

### Two-Step Hierarchical Retrieval

1. **Routing Step**: Analyzes your question and predicts which domain it belongs to
   - Domains: Clinical, Nonclinical, CMC, Compliance, Quality Systems, Submission Logistics, Labeling
2. **Retrieval Step**: Searches only within the predicted domain using ChromaDB metadata filtering
   - This ensures domain-specific, focused results

### Document Hierarchy Classification

Documents are automatically classified into 7 regulatory hierarchy levels:
- **Acts and Laws**: Congressional legislation, statutes
- **Administrative Law / Regulations / Directives / CFR**: Federal regulations, CFR sections
- **Pharmacopoeia Standards**: USP, NF, compendial standards
- **Authority Manual / Procedures**: SOPs, policy manuals, procedures
- **Guidances / Guidelines / Q&As**: FDA guidance documents, recommendations
- **Position Paper**: Position statements, viewpoints
- **White Papers**: Technical papers, research documents

### Document Processing Pipeline

1. **Text Extraction**: Extracts text from PDF, DOCX, or TXT files
2. **Hierarchy Classification**: Classifies document into hierarchy level (with user input if unclear)
3. **Summarization**: Generates ~200-word summary using LLM or extractive methods
4. **Chunking**: Splits text into ~500-1000 token segments with overlap
5. **Domain Assignment**: Assigns each chunk to a regulatory domain
6. **Embedding**: Creates vector embeddings for each chunk
7. **Storage**: Stores in ChromaDB with all metadata

## Project Structure

```
.
├── app.py                      # Main Streamlit UI and workflow
├── utils/
│   ├── extract_text.py         # PDF/DOCX/TXT text extraction
│   ├── router.py              # Domain router (keyword-based)
│   ├── document_classifier.py  # Hierarchy level classifier
│   ├── summarizer.py          # Document summarization
│   ├── ingest.py              # Chunking + domain/hierarchy assignment
│   ├── vectorstore.py         # ChromaDB with domain filtering
│   └── rag.py                 # Hierarchical RAG pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Customization

### Modify Domain Hierarchy

Edit `DOMAIN_HIERARCHY` in `utils/router.py`:
- Add/remove domains
- Update keywords for each domain
- Adjust subdomains

### Modify Document Hierarchy Levels

Edit `HIERARCHY_LEVELS` and `HIERARCHY_KEYWORDS` in `utils/document_classifier.py`:
- Add/remove hierarchy levels
- Update keywords and patterns for classification
- Adjust confidence thresholds

### Replace Router with Trained Classifier

Edit `utils/router.py` → `DomainRouter.route_query()` method:
- Replace keyword matching with fine-tuned model
- Use zero-shot LLM classifier
- Implement more sophisticated NLP methods

### Replace Hierarchy Classifier

Edit `utils/document_classifier.py` → `DocumentHierarchyClassifier.classify_document()`:
- Use fine-tuned classification model
- Implement LLM-based classification
- Add multi-class classification logic

### Adjust Summarization

Edit `utils/summarizer.py`:
- Change summary length (default: 200 words)
- Switch between LLM and extractive methods
- Customize summarization prompts

## Configuration Options

- **Embeddings:**
  - Local: `all-MiniLM-L6-v2` (default, no API key)
  - OpenAI: `text-embedding-ada-002` (requires API key)

- **LLM:**
  - OpenAI: GPT-3.5-turbo (requires API key)
  - Ollama: Local models like llama2, mistral (requires Ollama running)

- **Storage:**
  - ChromaDB: Persistent local storage in `./chroma_db/` directory

## Troubleshooting

**"Could not extract text from file":**
- Ensure file is not corrupted
- Check file format (PDF, DOCX, DOC, or TXT)
- Try re-saving the file

**"Failed to initialize components":**
- Check API keys if using OpenAI
- Ensure Ollama is running if using Ollama
- Verify all dependencies are installed

**"No relevant context found":**
- Upload more documents to the knowledge base
- Try rephrasing your question
- Check if documents cover the topic

**Classification unclear:**
- Review the classification scores shown
- Manually select the correct hierarchy level
- The system will remember your selection

## Notes

- ChromaDB data persists in `./chroma_db/` directory
- Local models download automatically on first use
- Document summaries are stored in session state (cleared on refresh)
- Domain filtering uses ChromaDB `where` clause for efficient retrieval
- Multiple files can be processed simultaneously
