# Architecture

## High-Level Pipeline

Files -> Parser -> Chunks -> Embeddings -> Vector Database -> Retriever -> Local LLM -> User Interface

## Components

### 1. Ingestion
Responsible for discovering files and recording metadata.

### 2. Parser
Responsible for converting supported files into text.

### 3. Chunker
Splits large text into smaller searchable pieces.

### 4. Embedding Model
Converts chunks into numerical vectors.

### 5. Vector Database
Stores vectors and metadata for similarity search.

### 6. Retriever
Finds relevant chunks for a user query.

### 7. Local LLM
Generates responses using retrieved context.

### 8. Interface
Allows the user to interact with the system through browser or API.

## Design Rule

Every component should be replaceable.

