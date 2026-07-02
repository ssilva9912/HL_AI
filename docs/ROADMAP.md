# Roadmap

## Phase 1: Laptop MVP
- Install Ollama
- Install Open WebUI
- Run a small local model
- Create a test document folder
- Ask questions through the local interface

## Phase 2: File Search
- Scan folders
- Extract text from documents
- Chunk text
- Generate embeddings
- Store vectors locally
- Retrieve relevant chunks

## Phase 3: RAG
- Connect retrieval to local LLM
- Add source paths
- Force answers to cite local files
- Add "I don't know" behavior when context is missing

## Phase 4: Homelab Integration
- Move services to Docker
- Add network shares
- Add local DNS name such as ai.home
- Add authentication

## Phase 5: Security
- Separate trusted devices, servers, IoT, and guests
- Add firewall rules
- Add backups
- Avoid exposing AI services to the public internet