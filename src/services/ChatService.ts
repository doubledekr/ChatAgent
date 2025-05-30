import { ChatMessage, DocumentSource } from '../types';
import { OpenAIService } from './OpenAIService';
import { DocumentService } from './DocumentService';
import AsyncStorage from '@react-native-async-storage/async-storage';

export class ChatService {
  private openAIService: OpenAIService;
  private documentService: DocumentService;
  private chatHistoryKey = 'chat_history';

  constructor() {
    this.openAIService = new OpenAIService();
    this.documentService = new DocumentService();
  }

  async initialize(): Promise<boolean> {
    const openAIReady = await this.openAIService.initialize();
    const documentReady = await this.documentService.initialize();
    return openAIReady && documentReady;
  }

  async sendMessage(
    message: string,
    onStreamChunk?: (chunk: string) => void
  ): Promise<ChatMessage> {
    try {
      // Search for relevant document chunks
      const searchResults = await this.documentService.searchDocuments(message, 5);
      
      // Extract context and sources
      const context = searchResults
        .map(result => result.metadata?.text || '')
        .join('\n\n');
      
      const sources: DocumentSource[] = searchResults.map(result => ({
        filename: result.metadata?.filename || 'Unknown',
        chunkId: result.metadata?.chunk_id || '0',
        relevanceScore: result.score || 0,
        snippet: (result.metadata?.text || '').substring(0, 200) + '...',
      }));

      // Generate AI response
      const response = await this.openAIService.generateChatResponse(
        message,
        context,
        onStreamChunk
      );

      // Create chat message
      const chatMessage: ChatMessage = {
        id: `${Date.now()}_response`,
        text: response,
        user: false,
        timestamp: new Date(),
        sources: sources.length > 0 ? sources : undefined,
      };

      // Save to chat history
      await this.saveChatMessage(chatMessage);
      
      return chatMessage;
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  async saveChatMessage(message: ChatMessage): Promise<void> {
    try {
      const history = await this.getChatHistory();
      history.push(message);
      
      // Keep only last 100 messages
      const trimmedHistory = history.slice(-100);
      
      await AsyncStorage.setItem(
        this.chatHistoryKey,
        JSON.stringify(trimmedHistory)
      );
    } catch (error) {
      console.error('Failed to save chat message:', error);
    }
  }

  async getChatHistory(): Promise<ChatMessage[]> {
    try {
      const stored = await AsyncStorage.getItem(this.chatHistoryKey);
      if (stored) {
        const messages = JSON.parse(stored);
        // Convert date strings back to Date objects
        return messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }));
      }
      return [];
    } catch (error) {
      console.error('Failed to load chat history:', error);
      return [];
    }
  }

  async clearChatHistory(): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.chatHistoryKey);
    } catch (error) {
      console.error('Failed to clear chat history:', error);
    }
  }

  async saveUserMessage(text: string): Promise<ChatMessage> {
    const userMessage: ChatMessage = {
      id: `${Date.now()}_user`,
      text,
      user: true,
      timestamp: new Date(),
    };

    await this.saveChatMessage(userMessage);
    return userMessage;
  }
}