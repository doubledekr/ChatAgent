import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  FlatList,
  Modal,
} from 'react-native';
import DocumentPicker from 'react-native-document-picker';
import { Document, ProcessingStatus } from '../types';
import { DocumentService } from '../services/DocumentService';

interface DocumentScreenProps {
  onNavigateToChat: () => void;
}

export const DocumentScreen: React.FC<DocumentScreenProps> = ({
  onNavigateToChat,
}) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [processing, setProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [showStatusModal, setShowStatusModal] = useState(false);
  const [documentService] = useState(() => new DocumentService());

  useEffect(() => {
    initializeAndLoadDocuments();
  }, []);

  const initializeAndLoadDocuments = async () => {
    try {
      const initialized = await documentService.initialize();
      if (!initialized) {
        Alert.alert(
          'Setup Required',
          'Please configure your API keys in settings to use document processing.'
        );
        return;
      }
      
      const storedDocs = await documentService.getStoredDocuments();
      setDocuments(storedDocs);
    } catch (error) {
      console.error('Failed to initialize:', error);
      Alert.alert('Error', 'Failed to initialize document service');
    }
  };

  const pickDocument = async () => {
    try {
      const result = await DocumentPicker.pick({
        type: [
          DocumentPicker.types.pdf,
          DocumentPicker.types.plainText,
          'application/epub+zip',
        ],
        allowMultiSelection: false,
      });

      if (result && result.length > 0) {
        const file = result[0];
        await processDocument(file);
      }
    } catch (error) {
      if (!DocumentPicker.isCancel(error)) {
        Alert.alert('Error', 'Failed to pick document');
        console.error('Document picker error:', error);
      }
    }
  };

  const processDocument = async (file: any) => {
    setProcessing(true);
    setShowStatusModal(true);

    try {
      const document = await documentService.processDocument(
        file.uri,
        file.name,
        (status) => {
          setProcessingStatus(status);
        }
      );

      setDocuments(prev => [...prev, document]);
      
      Alert.alert(
        'Success',
        `Document "${file.name}" has been processed successfully!`,
        [
          { text: 'OK', onPress: () => setShowStatusModal(false) }
        ]
      );
    } catch (error) {
      Alert.alert(
        'Processing Failed',
        error instanceof Error ? error.message : 'Unknown error occurred'
      );
      console.error('Document processing error:', error);
    } finally {
      setProcessing(false);
      setProcessingStatus(null);
    }
  };

  const deleteDocument = async (documentId: string) => {
    Alert.alert(
      'Delete Document',
      'Are you sure you want to delete this document? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await documentService.deleteDocument(documentId);
              setDocuments(prev => prev.filter(doc => doc.id !== documentId));
            } catch (error) {
              Alert.alert('Error', 'Failed to delete document');
              console.error('Delete error:', error);
            }
          },
        },
      ]
    );
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const renderDocument = ({ item }: { item: Document }) => (
    <View style={styles.documentCard}>
      <View style={styles.documentHeader}>
        <Text style={styles.documentName} numberOfLines={2}>
          {item.filename}
        </Text>
        <View style={styles.documentMeta}>
          <Text style={styles.metaText}>
            {formatFileSize(item.size)} • {item.chunkCount} chunks
          </Text>
          <Text style={styles.metaText}>
            {item.uploadDate.toLocaleDateString()}
          </Text>
        </View>
      </View>
      
      <View style={styles.documentActions}>
        <View style={[
          styles.statusBadge,
          { backgroundColor: item.processed ? '#27ae60' : '#f39c12' }
        ]}>
          <Text style={styles.statusText}>
            {item.processed ? 'Processed' : 'Processing'}
          </Text>
        </View>
        
        <TouchableOpacity
          style={styles.deleteButton}
          onPress={() => deleteDocument(item.id)}
        >
          <Text style={styles.deleteButtonText}>Delete</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Your Documents</Text>
        <Text style={styles.subtitle}>
          Upload documents to create your AI learning library
        </Text>
      </View>

      <TouchableOpacity
        style={styles.uploadButton}
        onPress={pickDocument}
        disabled={processing}
      >
        <Text style={styles.uploadButtonText}>
          {processing ? 'Processing...' : '+ Upload Document'}
        </Text>
      </TouchableOpacity>

      {documents.length > 0 ? (
        <FlatList
          data={documents}
          renderItem={renderDocument}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.documentsList}
          showsVerticalScrollIndicator={false}
        />
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No documents yet</Text>
          <Text style={styles.emptyText}>
            Upload your first document to start building your AI learning library
          </Text>
        </View>
      )}

      {documents.length > 0 && (
        <TouchableOpacity
          style={styles.chatButton}
          onPress={onNavigateToChat}
        >
          <Text style={styles.chatButtonText}>Start Chatting</Text>
        </TouchableOpacity>
      )}

      <Modal
        visible={showStatusModal}
        transparent
        animationType="fade"
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Processing Document</Text>
            
            {processingStatus && (
              <View style={styles.statusContainer}>
                <Text style={styles.statusMessage}>
                  {processingStatus.message}
                </Text>
                
                <View style={styles.progressBar}>
                  <View
                    style={[
                      styles.progressFill,
                      { width: `${processingStatus.progress}%` }
                    ]}
                  />
                </View>
                
                <Text style={styles.progressText}>
                  {Math.round(processingStatus.progress)}%
                </Text>
              </View>
            )}
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 20,
  },
  header: {
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    lineHeight: 22,
  },
  uploadButton: {
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 20,
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  documentsList: {
    paddingBottom: 100,
  },
  documentCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  documentHeader: {
    marginBottom: 10,
  },
  documentName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 5,
  },
  documentMeta: {
    gap: 2,
  },
  metaText: {
    fontSize: 12,
    color: '#7f8c8d',
  },
  documentActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
  },
  deleteButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
    backgroundColor: '#e74c3c',
  },
  deleteButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
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
  },
  emptyText: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
  },
  chatButton: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: '#27ae60',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  chatButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#fff',
    margin: 20,
    padding: 20,
    borderRadius: 8,
    minWidth: 300,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15,
    textAlign: 'center',
  },
  statusContainer: {
    alignItems: 'center',
  },
  statusMessage: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 15,
    textAlign: 'center',
  },
  progressBar: {
    width: '100%',
    height: 8,
    backgroundColor: '#ecf0f1',
    borderRadius: 4,
    marginBottom: 10,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#3498db',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c3e50',
  },
});