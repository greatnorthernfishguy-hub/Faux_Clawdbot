"""
Recursive Context Manager for Clawdbot

CHANGELOG [2025-01-28 - Josh]
Implements MIT's Recursive Language Model technique for unlimited context.

REFERENCE: https://www.youtube.com/watch?v=huszaaJPjU8
"MIT basically solved unlimited context windows"

APPROACH:
Instead of cramming everything into context (hits limits) or summarizing 
(lossy compression), we:

1. Store entire codebase in searchable environment
2. Give model TOOLS to query what it needs
3. Model recursively retrieves relevant pieces
4. No summarization loss - full fidelity access

This is like RAG, but IN-ENVIRONMENT with the model actively deciding
what context it needs rather than us guessing upfront.

EXAMPLE FLOW:
User: "How does Genesis handle surprise?"
Model: search_code("Genesis surprise detection")
    → Finds: genesis/substrate.py, genesis/attention.py
Model: read_file("genesis/substrate.py", lines 145-167)
    → Gets actual implementation
Model: search_testament("surprise detection rationale")
    → Gets design decision
Model: Synthesizes answer from retrieved pieces

NO CONTEXT WINDOW LIMIT - just selective retrieval.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import hashlib


class RecursiveContextManager:
    """
    Manages unlimited context via recursive retrieval.
    
    The model has TOOLS to search and read the codebase selectively,
    rather than loading everything upfront.
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize context manager for a repository.
        
        Args:
            repo_path: Path to the code repository
        """
        self.repo_path = Path(repo_path)
        
        # Initialize ChromaDB for semantic search
        # Using persistent storage so we don't re-index every restart
        self.chroma_client = chromadb.PersistentClient(
            path="/workspace/chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get CODEBASE collection
        collection_name = self._get_collection_name()
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"📚 Loaded existing index: {self.collection.count()} files")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "E-T Systems codebase"}
            )
            print(f"🆕 Created new collection: {collection_name}")
            self._index_codebase()
        
        # Create or get CONVERSATION collection for persistence
        # CHANGELOG [2025-01-30 - Josh]: Added conversation persistence
        # Implements full MIT recursive technique - chat history is searchable context
        conversations_name = f"conversations_{self._get_collection_name().split('_')[1]}"
        try:
            self.conversations = self.chroma_client.get_collection(conversations_name)
            print(f"💬 Loaded conversation history: {self.conversations.count()} exchanges")
        except:
            self.conversations = self.chroma_client.create_collection(
                name=conversations_name,
                metadata={"description": "Clawdbot conversation history"}
            )
            print(f"🆕 Created conversation collection: {conversations_name}")
    
    def _get_collection_name(self) -> str:
        """Generate unique collection name based on repo path."""
        path_hash = hashlib.md5(str(self.repo_path).encode()).hexdigest()[:8]
        return f"codebase_{path_hash}"
    
    def _index_codebase(self):
        """
        Index all code files for semantic search.
        
        This creates the "environment" that the model can search through.
        We index with metadata so search results include file paths.
        """
        print(f"📂 Indexing codebase at {self.repo_path}...")
        
        # File types to index
        code_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.txt', '.json', '.yaml', '.yml'}
        
        # Skip these directories
        skip_dirs = {'node_modules', '.git', '__pycache__', 'venv', 'env', '.venv', 'dist', 'build'}
        
        documents = []
        metadatas = []
        ids = []
        
        for file_path in self.repo_path.rglob('*'):
            # Skip directories and non-code files
            if file_path.is_dir():
                continue
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            if file_path.suffix not in code_extensions:
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Don't index empty files or massive files
                if not content.strip() or len(content) > 100000:
                    continue
                
                relative_path = str(file_path.relative_to(self.repo_path))
                
                documents.append(content)
                metadatas.append({
                    "path": relative_path,
                    "type": file_path.suffix[1:],  # Remove leading dot
                    "size": len(content)
                })
                ids.append(relative_path)
                
            except Exception as e:
                print(f"⚠️  Skipping {file_path.name}: {e}")
                continue
        
        if documents:
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            
            print(f"✅ Indexed {len(documents)} files")
        else:
            print("⚠️  No files found to index")
    
    def search_code(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search codebase semantically.
        
        This is a TOOL available to the model for recursive retrieval.
        Model can search for concepts without knowing exact file names.
        
        Args:
            query: What to search for (e.g. "surprise detection", "vector embedding")
            n_results: How many results to return
        
        Returns:
            List of dicts with {file, snippet, relevance}
        """
        if self.collection.count() == 0:
            return [{"error": "No files indexed yet"}]
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        # Format results for the model
        formatted = []
        for i in range(len(results['documents'][0])):
            # Truncate document to first 500 chars for search results
            # Model can read_file() if it wants the full content
            snippet = results['documents'][0][i][:500]
            if len(results['documents'][0][i]) > 500:
                snippet += "... [truncated, use read_file to see more]"
            
            formatted.append({
                "file": results['metadatas'][0][i]['path'],
                "snippet": snippet,
                "relevance": round(1 - results['distances'][0][i], 3),
                "type": results['metadatas'][0][i]['type']
            })
        
        return formatted
    
    def read_file(self, path: str, lines: Optional[Tuple[int, int]] = None) -> str:
        """
        Read a specific file or line range.
        
        This is a TOOL available to the model.
        After searching, model can read full files as needed.
        
        Args:
            path: Relative path to file
            lines: Optional (start, end) line numbers (1-indexed, inclusive)
        
        Returns:
            File content or specified lines
        """
        full_path = self.repo_path / path
        
        if not full_path.exists():
            return f"Error: File not found: {path}"
        
        if not full_path.is_relative_to(self.repo_path):
            return "Error: Path outside repository"
        
        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            if lines:
                start, end = lines
                content_lines = content.split('\n')
                # Adjust for 1-indexed
                selected_lines = content_lines[start-1:end]
                return '\n'.join(selected_lines)
            
            return content
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def search_testament(self, query: str) -> str:
        """
        Search architectural decisions in Testament.
        
        This is a TOOL available to the model.
        Helps model understand design rationale.
        
        Args:
            query: What decision to look for
        
        Returns:
            Relevant Testament sections
        """
        testament_path = self.repo_path / "TESTAMENT.md"
        
        if not testament_path.exists():
            return "Testament not found. No architectural decisions recorded yet."
        
        try:
            content = testament_path.read_text(encoding='utf-8')
            
            # Split into sections (marked by ## headers)
            sections = content.split('\n## ')
            
            # Simple relevance: sections that contain query terms
            query_lower = query.lower()
            relevant = []
            
            for section in sections:
                if query_lower in section.lower():
                    # Include section with header
                    if not section.startswith('#'):
                        section = '## ' + section
                    relevant.append(section)
            
            if relevant:
                return '\n\n'.join(relevant)
            else:
                return f"No Testament entries found matching '{query}'"
        
        except Exception as e:
            return f"Error searching Testament: {str(e)}"
    
    def list_files(self, directory: str = ".") -> List[str]:
        """
        List files in a directory.
        
        This is a TOOL available to the model.
        Helps model explore repository structure.
        
        Args:
            directory: Directory to list (relative path)
        
        Returns:
            List of file/directory names
        """
        dir_path = self.repo_path / directory
        
        if not dir_path.exists():
            return [f"Error: Directory not found: {directory}"]
        
        if not dir_path.is_relative_to(self.repo_path):
            return ["Error: Path outside repository"]
        
        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                # Skip hidden and system directories
                if item.name.startswith('.'):
                    continue
                if item.name in {'node_modules', '__pycache__', 'venv'}:
                    continue
                
                # Mark directories with /
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    items.append(item.name)
            
            return items
            
        except Exception as e:
            return [f"Error listing directory: {str(e)}"]
    
    def save_conversation_turn(self, user_message: str, assistant_message: str, turn_id: int):
        """
        Save a conversation turn to persistent storage.
        
        CHANGELOG [2025-01-30 - Josh]
        Implements MIT recursive technique for conversations.
        Chat history becomes searchable context that persists across sessions.
        
        Args:
            user_message: What the user said
            assistant_message: What Clawdbot responded
            turn_id: Unique ID for this turn (timestamp-based)
        """
        import time
        
        # Create a combined document for semantic search
        combined = f"USER: {user_message}\n\nASSISTANT: {assistant_message}"
        
        # Save with metadata
        self.conversations.add(
            documents=[combined],
            metadatas=[{
                "user": user_message[:500],  # Truncate for metadata
                "assistant": assistant_message[:500],
                "timestamp": int(time.time()),
                "turn": turn_id
            }],
            ids=[f"turn_{turn_id}"]
        )
        
        print(f"💾 Saved conversation turn {turn_id}")
    
    def search_conversations(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search past conversations for relevant context.
        
        This enables TRUE unlimited context - Clawdbot can remember
        everything ever discussed by searching its own conversation history.
        
        Args:
            query: What to search for in past conversations
            n_results: How many results to return
            
        Returns:
            List of past conversation turns with user/assistant messages
        """
        if self.conversations.count() == 0:
            return []
        
        results = self.conversations.query(
            query_texts=[query],
            n_results=min(n_results, self.conversations.count())
        )
        
        formatted = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted.append({
                "turn": metadata.get("turn", "unknown"),
                "user": metadata.get("user", ""),
                "assistant": metadata.get("assistant", ""),
                "full_text": doc,
                "relevance": i + 1  # Lower is more relevant
            })
        
        return formatted
    
    def get_conversation_count(self) -> int:
        """Get total number of saved conversation turns."""
        return self.conversations.count()
    
    def get_stats(self) -> Dict:
        """
        Get statistics about indexed codebase.
        
        Returns:
            Dict with file counts, sizes, etc.
        """
        return {
            "total_files": self.collection.count(),
            "repo_path": str(self.repo_path),
            "collection_name": self.collection.name
        }
