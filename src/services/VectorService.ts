import AsyncStorage from '@react-native-async-storage/async-storage';
import { VectorSearchResult } from '../types';

export class VectorService {
  private apiKey: string | null = null;
  private environment: string | null = null;
  private indexName: string = 'docai';
  private baseUrl: string = '';

  async initialize(): Promise<boolean> {
    try {
      this.apiKey = await AsyncStorage.getItem('PINECONE_API_KEY');
      this.environment = await AsyncStorage.getItem('PINECONE_ENVIRONMENT');
      
      if (this.apiKey && this.environment) {
        this.baseUrl = `https://${this.indexName}-${this.environment}.svc.pinecone.io`;
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to load Pinecone configuration:', error);
      return false;
    }
  }

  async setCredentials(apiKey: string, environment: string): Promise<void> {
    this.apiKey = apiKey;
    this.environment = environment;
    this.baseUrl = `https://${this.indexName}-${environment}.svc.pinecone.io`;
    
    await AsyncStorage.setItem('PINECONE_API_KEY', apiKey);
    await AsyncStorage.setItem('PINECONE_ENVIRONMENT', environment);
  }

  async upsertVector(
    id: string,
    vector: number[],
    metadata: Record<string, any>
  ): Promise<boolean> {
    if (!this.apiKey) {
      throw new Error('Pinecone API key not configured');
    }

    try {
      const response = await fetch(`${this.baseUrl}/vectors/upsert`, {
        method: 'POST',
        headers: {
          'Api-Key': this.apiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          vectors: [
            {
              id,
              values: vector,
              metadata,
            },
          ],
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Vector upsert failed:', error);
      return false;
    }
  }

  async queryVectors(
    vector: number[],
    topK: number = 5,
    filter?: Record<string, any>
  ): Promise<VectorSearchResult[]> {
    if (!this.apiKey) {
      throw new Error('Pinecone API key not configured');
    }

    try {
      const body: any = {
        vector,
        topK,
        includeMetadata: true,
      };

      if (filter) {
        body.filter = filter;
      }

      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Api-Key': this.apiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Pinecone query failed: ${response.statusText}`);
      }

      const data = await response.json();
      return data.matches || [];
    } catch (error) {
      console.error('Vector query failed:', error);
      throw error;
    }
  }

  async deleteVectors(ids: string[]): Promise<boolean> {
    if (!this.apiKey) {
      throw new Error('Pinecone API key not configured');
    }

    try {
      const response = await fetch(`${this.baseUrl}/vectors/delete`, {
        method: 'POST',
        headers: {
          'Api-Key': this.apiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ids,
        }),
      });

      return response.ok;
    } catch (error) {
      console.error('Vector deletion failed:', error);
      return false;
    }
  }

  async getIndexStats(): Promise<any> {
    if (!this.apiKey) {
      throw new Error('Pinecone API key not configured');
    }

    try {
      const response = await fetch(`${this.baseUrl}/describe_index_stats`, {
        method: 'POST',
        headers: {
          'Api-Key': this.apiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error(`Failed to get index stats: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get index stats:', error);
      throw error;
    }
  }
}