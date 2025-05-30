export interface Document {
  id: string;
  filename: string;
  content: string;
  uploadDate: Date;
  processed: boolean;
  chunkCount: number;
  fileType: string;
  size: number;
}

export interface ChatMessage {
  id: string;
  text: string;
  user: boolean;
  timestamp: Date;
  sources?: DocumentSource[];
}

export interface DocumentSource {
  filename: string;
  chunkId: string;
  relevanceScore: number;
  snippet: string;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  metadata: {
    filename: string;
    chunk_id: number;
    text: string;
    tags: string[];
  };
}

export interface ProcessingStatus {
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  message?: string;
  error?: string;
}

export interface AppState {
  documents: Document[];
  chatHistory: ChatMessage[];
  isProcessing: boolean;
  currentDocument?: Document;
}