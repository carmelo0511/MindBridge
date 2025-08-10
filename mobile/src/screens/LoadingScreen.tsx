import React from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { Colors } from '../utils/Colors';

const LoadingScreen: React.FC = () => (
  <View style={styles.container}>
    <ActivityIndicator size="large" color={Colors.primary} />
    <Text style={styles.text}>Loading MindBridge...</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  text: {
    marginTop: 16,
    fontSize: 16,
    color: Colors.textSecondary,
  },
});

export default LoadingScreen;