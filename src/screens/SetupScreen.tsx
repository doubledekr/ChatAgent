import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface SetupScreenProps {
  onComplete: () => void;
}

export const SetupScreen: React.FC<SetupScreenProps> = ({ onComplete }) => {
  const [openAIKey, setOpenAIKey] = useState('');
  const [pineconeKey, setPineconeKey] = useState('');
  const [pineconeEnv, setPineconeEnv] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadExistingKeys();
  }, []);

  const loadExistingKeys = async () => {
    try {
      const openAI = await AsyncStorage.getItem('OPENAI_API_KEY');
      const pinecone = await AsyncStorage.getItem('PINECONE_API_KEY');
      const environment = await AsyncStorage.getItem('PINECONE_ENVIRONMENT');
      
      if (openAI) setOpenAIKey(openAI);
      if (pinecone) setPineconeKey(pinecone);
      if (environment) setPineconeEnv(environment);
    } catch (error) {
      console.error('Failed to load existing keys:', error);
    }
  };

  const validateAndSave = async () => {
    if (!openAIKey.trim()) {
      Alert.alert('Error', 'OpenAI API key is required');
      return;
    }

    if (!pineconeKey.trim()) {
      Alert.alert('Error', 'Pinecone API key is required');
      return;
    }

    if (!pineconeEnv.trim()) {
      Alert.alert('Error', 'Pinecone environment is required');
      return;
    }

    setLoading(true);

    try {
      // Save API keys
      await AsyncStorage.setItem('OPENAI_API_KEY', openAIKey.trim());
      await AsyncStorage.setItem('PINECONE_API_KEY', pineconeKey.trim());
      await AsyncStorage.setItem('PINECONE_ENVIRONMENT', pineconeEnv.trim());

      Alert.alert(
        'Success',
        'API keys saved successfully!',
        [{ text: 'Continue', onPress: onComplete }]
      );
    } catch (error) {
      Alert.alert('Error', 'Failed to save API keys. Please try again.');
      console.error('Save error:', error);
    } finally {
      setLoading(false);
    }
  };

  const skipSetup = () => {
    Alert.alert(
      'Skip Setup',
      'You can configure API keys later in settings. The app will have limited functionality without proper configuration.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Skip', onPress: onComplete },
      ]
    );
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>AI Learning Companion</Text>
          <Text style={styles.subtitle}>
            Set up your API keys to get started with document processing and AI chat
          </Text>
        </View>

        <View style={styles.form}>
          <View style={styles.inputGroup}>
            <Text style={styles.label}>OpenAI API Key</Text>
            <Text style={styles.description}>
              Required for generating embeddings and AI responses
            </Text>
            <TextInput
              style={styles.input}
              value={openAIKey}
              onChangeText={setOpenAIKey}
              placeholder="sk-..."
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Pinecone API Key</Text>
            <Text style={styles.description}>
              Required for vector database storage and search
            </Text>
            <TextInput
              style={styles.input}
              value={pineconeKey}
              onChangeText={setPineconeKey}
              placeholder="Your Pinecone API key"
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Pinecone Environment</Text>
            <Text style={styles.description}>
              Your Pinecone environment (e.g., us-west1-gcp)
            </Text>
            <TextInput
              style={styles.input}
              value={pineconeEnv}
              onChangeText={setPineconeEnv}
              placeholder="us-west1-gcp"
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>
        </View>

        <View style={styles.buttons}>
          <TouchableOpacity
            style={[styles.button, styles.primaryButton]}
            onPress={validateAndSave}
            disabled={loading}
          >
            <Text style={styles.primaryButtonText}>
              {loading ? 'Saving...' : 'Save & Continue'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.secondaryButton]}
            onPress={skipSetup}
            disabled={loading}
          >
            <Text style={styles.secondaryButtonText}>Skip for Now</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.helpText}>
          <Text style={styles.help}>
            Need help getting API keys?{'\n'}
            • OpenAI: Visit platform.openai.com{'\n'}
            • Pinecone: Visit app.pinecone.io
          </Text>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
  },
  header: {
    marginBottom: 30,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 10,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
  },
  form: {
    marginBottom: 30,
  },
  inputGroup: {
    marginBottom: 25,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 5,
  },
  description: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e1e8ed',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    color: '#2c3e50',
  },
  buttons: {
    gap: 15,
  },
  button: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: '#3498db',
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#bdc3c7',
  },
  secondaryButtonText: {
    color: '#7f8c8d',
    fontSize: 16,
  },
  helpText: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#ecf0f1',
    borderRadius: 8,
  },
  help: {
    fontSize: 14,
    color: '#7f8c8d',
    lineHeight: 20,
  },
});