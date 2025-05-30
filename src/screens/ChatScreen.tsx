import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  FlatList,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { ChatMessage, DocumentSource } from '../types';
import { ChatService } from '../services/ChatService';

interface ChatScreenProps {
  onNavigateToDocuments: () => void;
}

export const ChatScreen: React.FC<ChatScreenProps> = ({
  onNavigateToDocuments,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [chatService] = useState(() => new ChatService());
  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    initializeChat();
  }, []);

  const initializeChat = async () => {
    try {
      const initialized = await chatService.initialize();
      if (!initialized) {
        Alert.alert(
          'Setup Required',
          'Please configure your API keys to use the chat feature.',
          [
            { text: 'Go to Setup', onPress: onNavigateToDocuments },
            { text: 'Cancel', style: 'cancel' },
          ]
        );
        return;
      }

      const history = await chatService.getChatHistory();
      setMessages(history);
    } catch (error) {
      console.error('Failed to initialize chat:', error);
      Alert.alert('Error', 'Failed to initialize chat service');
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessageText = inputText.trim();
    setInputText('');
    setIsLoading(true);

    try {
      // Save user message
      const userMessage = await chatService.saveUserMessage(userMessageText);
      setMessages(prev => [...prev, userMessage]);

      // Start streaming response
      setStreamingMessage('');
      
      const aiResponse = await chatService.sendMessage(
        userMessageText,
        (chunk) => {
          setStreamingMessage(prev => prev + chunk);
        }
      );

      // Add complete AI response
      setMessages(prev => [...prev, aiResponse]);
      setStreamingMessage('');
    } catch (error) {
      Alert.alert(
        'Error',
        'Failed to send message. Please check your API keys and try again.'
      );
      console.error('Send message error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    Alert.alert(
      'Clear Chat',
      'Are you sure you want to clear all chat messages?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            try {
              await chatService.clearChatHistory();
              setMessages([]);
            } catch (error) {
              Alert.alert('Error', 'Failed to clear chat history');
            }
          },
        },
      ]
    );
  };

  const renderMessage = ({ item }: { item: ChatMessage }) => (
    <View style={[
      styles.messageContainer,
      item.user ? styles.userMessage : styles.aiMessage
    ]}>
      <Text style={[
        styles.messageText,
        { color: item.user ? '#fff' : '#2c3e50' }
      ]}>
        {item.text}
      </Text>
      
      {item.sources && item.sources.length > 0 && (
        <View style={styles.sourcesContainer}>
          <Text style={styles.sourcesTitle}>Sources:</Text>
          {item.sources.map((source, index) => (
            <View key={index} style={styles.sourceItem}>
              <Text style={styles.sourceFilename}>
                {source.filename} (Score: {source.relevanceScore.toFixed(2)})
              </Text>
              <Text style={styles.sourceSnippet} numberOfLines={2}>
                {source.snippet}
              </Text>
            </View>
          ))}
        </View>
      )}
      
      <Text style={[
        styles.timestamp,
        { color: item.user ? 'rgba(255,255,255,0.7)' : '#7f8c8d' }
      ]}>
        {item.timestamp.toLocaleTimeString()}
      </Text>
    </View>
  );

  const renderStreamingMessage = () => {
    if (!streamingMessage) return null;
    
    return (
      <View style={[styles.messageContainer, styles.aiMessage]}>
        <Text style={[styles.messageText, { color: '#2c3e50' }]}>
          {streamingMessage}
        </Text>
        <View style={styles.typingIndicator}>
          <Text style={styles.typingText}>AI is typing...</Text>
        </View>
      </View>
    );
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <View style={styles.header}>
        <Text style={styles.title}>AI Chat</Text>
        <View style={styles.headerButtons}>
          <TouchableOpacity
            style={styles.headerButton}
            onPress={onNavigateToDocuments}
          >
            <Text style={styles.headerButtonText}>Documents</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.headerButton}
            onPress={clearChat}
          >
            <Text style={styles.headerButtonText}>Clear</Text>
          </TouchableOpacity>
        </View>
      </View>

      {messages.length === 0 && !streamingMessage ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>Welcome to AI Chat!</Text>
          <Text style={styles.emptyText}>
            Ask questions about your uploaded documents and I'll help you learn and understand the content.
          </Text>
        </View>
      ) : (
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.messagesList}
          showsVerticalScrollIndicator={false}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
          ListFooterComponent={renderStreamingMessage}
        />
      )}

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Ask a question about your documents..."
          multiline
          maxLength={1000}
          editable={!isLoading}
        />
        
        <TouchableOpacity
          style={[
            styles.sendButton,
            { opacity: (!inputText.trim() || isLoading) ? 0.5 : 1 }
          ]}
          onPress={sendMessage}
          disabled={!inputText.trim() || isLoading}
        >
          <Text style={styles.sendButtonText}>
            {isLoading ? '...' : 'Send'}
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e1e8ed',
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  headerButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#3498db',
    borderRadius: 6,
  },
  headerButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  messagesList: {
    padding: 20,
    paddingBottom: 10,
  },
  messageContainer: {
    marginBottom: 15,
    padding: 12,
    borderRadius: 12,
    maxWidth: '85%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#3498db',
  },
  aiMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e1e8ed',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
    marginBottom: 5,
  },
  sourcesContainer: {
    marginTop: 10,
    padding: 8,
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: 6,
  },
  sourcesTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#7f8c8d',
    marginBottom: 5,
  },
  sourceItem: {
    marginBottom: 5,
  },
  sourceFilename: {
    fontSize: 11,
    fontWeight: '500',
    color: '#2c3e50',
  },
  sourceSnippet: {
    fontSize: 10,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  timestamp: {
    fontSize: 11,
    marginTop: 5,
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 5,
  },
  typingText: {
    fontSize: 12,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 10,
    textAlign: 'center',
  },
  emptyText: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 20,
    paddingTop: 15,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e1e8ed',
    alignItems: 'flex-end',
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#e1e8ed',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginRight: 10,
    maxHeight: 100,
    fontSize: 16,
    color: '#2c3e50',
  },
  sendButton: {
    backgroundColor: '#3498db',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});