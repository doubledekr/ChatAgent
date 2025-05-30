import AsyncStorage from '@react-native-async-storage/async-storage';

export class OpenAIService {
  private apiKey: string | null = null;
  private baseUrl = 'https://api.openai.com/v1';

  async initialize(): Promise<boolean> {
    try {
      this.apiKey = await AsyncStorage.getItem('OPENAI_API_KEY');
      return this.apiKey !== null;
    } catch (error) {
      console.error('Failed to load OpenAI API key:', error);
      return false;
    }
  }

  async setApiKey(apiKey: string): Promise<void> {
    this.apiKey = apiKey;
    await AsyncStorage.setItem('OPENAI_API_KEY', apiKey);
  }

  async generateEmbeddings(text: string): Promise<number[]> {
    if (!this.apiKey) {
      throw new Error('OpenAI API key not configured');
    }

    const response = await fetch(`${this.baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'text-embedding-ada-002',
        input: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
  }

  async generateChatResponse(
    query: string,
    context: string,
    onStream?: (chunk: string) => void
  ): Promise<string> {
    if (!this.apiKey) {
      throw new Error('OpenAI API key not configured');
    }

    const messages = [
      {
        role: 'system',
        content: `You are an AI learning companion that helps users understand and learn from their documents. 
        Use the provided context to answer questions accurately and helpfully.
        
        Context: ${context}`,
      },
      {
        role: 'user',
        content: query,
      },
    ];

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        messages,
        stream: !!onStream,
        temperature: 0.7,
        max_tokens: 1000,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    if (onStream && response.body) {
      // Handle streaming response
      const reader = response.body.getReader();
      let fullResponse = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            
            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices[0]?.delta?.content || '';
              if (content) {
                fullResponse += content;
                onStream(content);
              }
            } catch (e) {
              // Skip malformed JSON
            }
          }
        }
      }
      
      return fullResponse;
    } else {
      // Handle non-streaming response
      const data = await response.json();
      return data.choices[0].message.content;
    }
  }

  async generateTags(text: string, filename: string): Promise<string[]> {
    if (!this.apiKey) {
      throw new Error('OpenAI API key not configured');
    }

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: `Generate 5-8 relevant tags for the following text content. 
            Consider the filename: ${filename}
            Return only a JSON array of strings, no other text.`,
          },
          {
            role: 'user',
            content: text.substring(0, 2000), // Limit text for tag generation
          },
        ],
        temperature: 0.3,
        max_tokens: 100,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data = await response.json();
    try {
      return JSON.parse(data.choices[0].message.content);
    } catch (e) {
      // Fallback to basic tags if JSON parsing fails
      return ['document', 'content', filename.split('.').pop() || 'file'];
    }
  }
}