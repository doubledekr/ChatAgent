import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  StatusBar,
  SafeAreaView,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { SetupScreen } from './screens/SetupScreen';
import { DocumentScreen } from './screens/DocumentScreen';
import { ChatScreen } from './screens/ChatScreen';

type Screen = 'setup' | 'documents' | 'chat';

export const App: React.FC = () => {
  const [currentScreen, setCurrentScreen] = useState<Screen>('setup');
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    checkInitialSetup();
  }, []);

  const checkInitialSetup = async () => {
    try {
      const openAIKey = await AsyncStorage.getItem('OPENAI_API_KEY');
      const pineconeKey = await AsyncStorage.getItem('PINECONE_API_KEY');
      const pineconeEnv = await AsyncStorage.getItem('PINECONE_ENVIRONMENT');

      // If API keys are configured, skip setup
      if (openAIKey && pineconeKey && pineconeEnv) {
        setCurrentScreen('documents');
      }
    } catch (error) {
      console.error('Failed to check initial setup:', error);
    } finally {
      setIsInitialized(true);
    }
  };

  const handleSetupComplete = () => {
    setCurrentScreen('documents');
  };

  const navigateToChat = () => {
    setCurrentScreen('chat');
  };

  const navigateToDocuments = () => {
    setCurrentScreen('documents');
  };

  if (!isInitialized) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          {/* Loading screen could be added here */}
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />
      
      {currentScreen === 'setup' && (
        <SetupScreen onComplete={handleSetupComplete} />
      )}
      
      {currentScreen === 'documents' && (
        <DocumentScreen onNavigateToChat={navigateToChat} />
      )}
      
      {currentScreen === 'chat' && (
        <ChatScreen onNavigateToDocuments={navigateToDocuments} />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default App;