import RNFS from 'react-native-fs';
import { Document, ProcessingStatus } from '../types';
import { OpenAIService } from './OpenAIService';
import { VectorService } from './VectorService';
import AsyncStorage from '@react-native-async-storage/async-storage';

export class DocumentService {
  private openAIService: OpenAIService;
  private vectorService: VectorService;
  private documentsStorageKey = 'stored_documents';

  constructor() {
    this.openAIService = new OpenAIService();
    this.vectorService = new VectorService();
  }

  async initialize(): Promise<boolean> {
    const openAIReady = await this.openAIService.initialize();
    const vectorReady = await this.vectorService.initialize();
    return openAIReady && vectorReady;
  }

  async extractTextFromFile(filePath: string, fileType: string): Promise<string> {
    try {
      if (fileType === 'txt' || fileType === 'text') {
        return await RNFS.readFile(filePath, 'utf8');
      }
      
      // For PDFs and other complex formats, we'll need native modules
      // For now, return a placeholder that indicates the file needs processing
      throw new Error(`File type ${fileType} requires external processing`);
    } catch (error) {
      console.error('Text extraction failed:', error);
      throw error;
    }
  }

  chunkText(text: string, maxTokens: number = 500): string[] {
    // Simple chunking by sentences and paragraphs
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const chunks: string[] = [];
    let currentChunk = '';
    let currentTokens = 0;

    for (const sentence of sentences) {
      const sentenceTokens = this.estimateTokens(sentence);
      
      if (currentTokens + sentenceTokens > maxTokens && currentChunk) {
        chunks.push(currentChunk.trim());
        currentChunk = sentence;
        currentTokens = sentenceTokens;
      } else {
        currentChunk += (currentChunk ? '. ' : '') + sentence;
        currentTokens += sentenceTokens;
      }
    }

    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }

    return chunks.length > 0 ? chunks : [text];
  }

  private estimateTokens(text: string): number {
    // Rough estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  async processDocument(
    filePath: string,
    filename: string,
    onProgress?: (status: ProcessingStatus) => void
  ): Promise<Document> {
    const fileType = filename.split('.').pop()?.toLowerCase() || '';
    const stats = await RNFS.stat(filePath);
    
    const document: Document = {
      id: `${Date.now()}_${filename}`,
      filename,
      content: '',
      uploadDate: new Date(),
      processed: false,
      chunkCount: 0,
      fileType,
      size: stats.size,
    };

    try {
      onProgress?.({
        filename,
        status: 'processing',
        progress: 10,
        message: 'Extracting text...',
      });

      // Extract text
      const text = await this.extractTextFromFile(filePath, fileType);
      document.content = text;

      onProgress?.({
        filename,
        status: 'processing',
        progress: 30,
        message: 'Creating chunks...',
      });

      // Chunk the text
      const chunks = this.chunkText(text);
      document.chunkCount = chunks.length;

      onProgress?.({
        filename,
        status: 'processing',
        progress: 50,
        message: 'Generating embeddings...',
      });

      // Process each chunk
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const chunkId = `${document.id}_chunk_${i}`;

        try {
          // Generate embedding
          const embedding = await this.openAIService.generateEmbeddings(chunk);
          
          // Generate tags
          const tags = await this.openAIService.generateTags(chunk, filename);

          // Store in vector database
          const metadata = {
            filename,
            chunk_id: i,
            text: chunk,
            tags,
            document_id: document.id,
            file_type: fileType,
          };

          await this.vectorService.upsertVector(chunkId, embedding, metadata);

          const progress = 50 + ((i + 1) / chunks.length) * 40;
          onProgress?.({
            filename,
            status: 'processing',
            progress,
            message: `Processing chunk ${i + 1} of ${chunks.length}...`,
          });
        } catch (error) {
          console.error(`Failed to process chunk ${i}:`, error);
        }
      }

      document.processed = true;
      
      // Save document to local storage
      await this.saveDocument(document);

      onProgress?.({
        filename,
        status: 'completed',
        progress: 100,
        message: 'Processing complete!',
      });

      return document;
    } catch (error) {
      onProgress?.({
        filename,
        status: 'error',
        progress: 0,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  async saveDocument(document: Document): Promise<void> {
    try {
      const existingDocs = await this.getStoredDocuments();
      const updatedDocs = existingDocs.filter(d => d.id !== document.id);
      updatedDocs.push(document);
      
      await AsyncStorage.setItem(
        this.documentsStorageKey,
        JSON.stringify(updatedDocs)
      );
    } catch (error) {
      console.error('Failed to save document:', error);
      throw error;
    }
  }

  async getStoredDocuments(): Promise<Document[]> {
    try {
      const stored = await AsyncStorage.getItem(this.documentsStorageKey);
      if (stored) {
        const docs = JSON.parse(stored);
        // Convert date strings back to Date objects
        return docs.map((doc: any) => ({
          ...doc,
          uploadDate: new Date(doc.uploadDate),
        }));
      }
      return [];
    } catch (error) {
      console.error('Failed to load documents:', error);
      return [];
    }
  }

  async deleteDocument(documentId: string): Promise<void> {
    try {
      const docs = await this.getStoredDocuments();
      const document = docs.find(d => d.id === documentId);
      
      if (document) {
        // Delete from vector database
        const chunkIds = Array.from(
          { length: document.chunkCount },
          (_, i) => `${documentId}_chunk_${i}`
        );
        await this.vectorService.deleteVectors(chunkIds);
        
        // Remove from local storage
        const updatedDocs = docs.filter(d => d.id !== documentId);
        await AsyncStorage.setItem(
          this.documentsStorageKey,
          JSON.stringify(updatedDocs)
        );
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
      throw error;
    }
  }

  async searchDocuments(query: string, topK: number = 5): Promise<any[]> {
    try {
      // Generate embedding for the query
      const queryEmbedding = await this.openAIService.generateEmbeddings(query);
      
      // Search in vector database
      const results = await this.vectorService.queryVectors(queryEmbedding, topK);
      
      return results;
    } catch (error) {
      console.error('Document search failed:', error);
      throw error;
    }
  }
}