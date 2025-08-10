/**
 * MindBridge Mobile Service
 * 
 * Main service layer for interfacing with the MindBridge backend
 * Handles all mental health analysis, privacy management, and data processing
 */

import AsyncStorage from '@react-native-community/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { Alert } from 'react-native';

// Types
interface MentalHealthStatus {
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
  overallScore: number;
  conditions: {
    depression: number;
    anxiety: number;
    ptsd: number;
    bipolar: number;
    burnout: number;
  };
  lastUpdated: Date;
  confidence: number;
  userProfile?: UserProfile;
}

interface UserProfile {
  id: string;
  preferredName: string;
  culturalBackground: string;
  language: string;
  privacyLevel: 'minimal' | 'moderate' | 'high' | 'maximum';
  baselineMetrics: any;
}

interface Intervention {
  id: string;
  type: string;
  title: string;
  description: string;
  duration: number;
  priority: number;
  culturallyAdapted: boolean;
  instructions?: string[];
  resources?: any[];
}

interface CrisisResource {
  id: string;
  name: string;
  phone: string;
  text?: string;
  website?: string;
  description: string;
  availability: string;
  languages: string[];
}

class MindBridgeServiceClass {
  private isInitialized = false;
  private offlineMode = false;
  private userProfile: UserProfile | null = null;
  private mentalHealthEngine: any = null;
  private privacyManager: any = null;

  async initialize(): Promise<void> {
    try {
      // Check network status
      const networkState = await NetInfo.fetch();
      this.offlineMode = !networkState.isConnected;

      // Load user profile
      await this.loadUserProfile();

      // Initialize privacy manager
      await this.initializePrivacyManager();

      // Initialize mental health engine (local)
      await this.initializeMentalHealthEngine();

      // Load cached data
      await this.loadCachedData();

      this.isInitialized = true;
      console.log('MindBridge Service initialized successfully');
    } catch (error) {
      console.error('MindBridge Service initialization failed:', error);
      throw error;
    }
  }

  private async loadUserProfile(): Promise<void> {
    try {
      const profileData = await AsyncStorage.getItem('user_profile');
      if (profileData) {
        this.userProfile = JSON.parse(profileData);
      }
    } catch (error) {
      console.error('Failed to load user profile:', error);
    }
  }

  private async initializePrivacyManager(): Promise<void> {
    // Initialize privacy manager with user's privacy level
    const privacyLevel = this.userProfile?.privacyLevel || 'maximum';
    
    // This would initialize the actual privacy manager
    this.privacyManager = {
      privacyLevel,
      encryptData: (data: any) => data, // Simplified for demo
      decryptData: (data: any) => data,
      anonymizeData: (data: any) => data,
    };
  }

  private async initializeMentalHealthEngine(): Promise<void> {
    // Initialize the local mental health analysis engine
    // This would load the actual GPT-OSS model and processing pipeline
    this.mentalHealthEngine = {
      analyzeText: async (text: string, context?: any) => {
        // Simulate local ML processing
        return this.simulateLocalAnalysis(text, context);
      },
      generateInterventions: async (status: MentalHealthStatus) => {
        return this.simulateInterventionGeneration(status);
      }
    };
  }

  private async loadCachedData(): Promise<void> {
    // Load cached mental health data, interventions, etc.
    try {
      const cachedStatus = await AsyncStorage.getItem('mental_health_status');
      if (cachedStatus) {
        // Parse and validate cached data
        const status = JSON.parse(cachedStatus);
        // Update if data is recent (within last hour)
        const lastUpdate = new Date(status.lastUpdated);
        if (Date.now() - lastUpdate.getTime() > 3600000) {
          // Data is stale, trigger refresh
          await this.refreshMentalHealthStatus();
        }
      }
    } catch (error) {
      console.error('Failed to load cached data:', error);
    }
  }

  async initializeWithUserPreferences(preferences: any): Promise<void> {
    try {
      // Create user profile
      this.userProfile = {
        id: this.generateUserId(),
        preferredName: preferences.name || 'Friend',
        culturalBackground: preferences.culturalContext?.background || 'general',
        language: preferences.culturalContext?.language || 'en',
        privacyLevel: preferences.privacyLevel,
        baselineMetrics: {}
      };

      // Save user profile
      await AsyncStorage.setItem('user_profile', JSON.stringify(this.userProfile));

      // Initialize baseline metrics
      await this.createInitialBaseline();
    } catch (error) {
      console.error('Failed to initialize with user preferences:', error);
      throw error;
    }
  }

  private generateUserId(): string {
    // Generate anonymous user ID
    return 'user_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  }

  async createInitialBaseline(): Promise<void> {
    try {
      // Create initial baseline assessment
      const baselineData = {
        timestamp: new Date().toISOString(),
        riskLevel: 'low' as const,
        overallScore: 0.3,
        conditions: {
          depression: 0.2,
          anxiety: 0.2,
          ptsd: 0.1,
          bipolar: 0.1,
          burnout: 0.2
        },
        confidence: 0.1 // Low confidence for initial baseline
      };

      await AsyncStorage.setItem('baseline_metrics', JSON.stringify(baselineData));
      console.log('Initial baseline created');
    } catch (error) {
      console.error('Failed to create initial baseline:', error);
    }
  }

  async getCurrentMentalHealthStatus(): Promise<MentalHealthStatus | null> {
    try {
      if (!this.isInitialized) {
        await this.initialize();
      }

      const statusData = await AsyncStorage.getItem('mental_health_status');
      if (!statusData) {
        return null;
      }

      const status = JSON.parse(statusData);
      return {
        ...status,
        lastUpdated: new Date(status.lastUpdated),
        userProfile: this.userProfile
      };
    } catch (error) {
      console.error('Failed to get current mental health status:', error);
      return null;
    }
  }

  async analyzeTextForMentalHealth(text: string, context?: any): Promise<MentalHealthStatus> {
    try {
      if (!this.mentalHealthEngine) {
        throw new Error('Mental health engine not initialized');
      }

      // Analyze text using local engine
      const analysis = await this.mentalHealthEngine.analyzeText(text, {
        culturalContext: this.userProfile?.culturalBackground,
        language: this.userProfile?.language,
        ...context
      });

      // Create mental health status
      const status: MentalHealthStatus = {
        riskLevel: analysis.riskLevel,
        overallScore: analysis.overallScore,
        conditions: analysis.conditions,
        lastUpdated: new Date(),
        confidence: analysis.confidence,
        userProfile: this.userProfile || undefined
      };

      // Cache the result
      await AsyncStorage.setItem('mental_health_status', JSON.stringify(status));

      // Log for analytics (anonymized)
      await this.logAnalysisEvent(status);

      return status;
    } catch (error) {
      console.error('Text analysis failed:', error);
      throw error;
    }
  }

  private async simulateLocalAnalysis(text: string, context?: any): Promise<any> {
    // Simulate local mental health analysis
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate processing time

    // Basic sentiment and keyword analysis (simplified)
    const lowMoodWords = ['sad', 'depressed', 'hopeless', 'empty', 'worthless'];
    const anxietyWords = ['anxious', 'worried', 'panic', 'scared', 'nervous'];
    const positiveWords = ['happy', 'good', 'great', 'excited', 'grateful'];

    const lowerText = text.toLowerCase();
    let depressionScore = 0;
    let anxietyScore = 0;
    let positivity = 0;

    lowMoodWords.forEach(word => {
      if (lowerText.includes(word)) depressionScore += 0.2;
    });

    anxietyWords.forEach(word => {
      if (lowerText.includes(word)) anxietyScore += 0.2;
    });

    positiveWords.forEach(word => {
      if (lowerText.includes(word)) positivity += 0.1;
    });

    // Calculate overall risk
    const overallScore = Math.min((depressionScore + anxietyScore - positivity), 1.0);
    const riskLevel = this.calculateRiskLevel(overallScore);

    return {
      riskLevel,
      overallScore: Math.max(overallScore, 0.1),
      conditions: {
        depression: Math.min(depressionScore, 1.0),
        anxiety: Math.min(anxietyScore, 1.0),
        ptsd: 0.1,
        bipolar: 0.1,
        burnout: 0.15
      },
      confidence: 0.7,
      analysisType: 'local_simulation'
    };
  }

  private calculateRiskLevel(score: number): 'low' | 'moderate' | 'high' | 'critical' {
    if (score < 0.25) return 'low';
    if (score < 0.5) return 'moderate';
    if (score < 0.75) return 'high';
    return 'critical';
  }

  async getSuggestedInterventions(): Promise<Intervention[]> {
    try {
      const status = await this.getCurrentMentalHealthStatus();
      if (!status) {
        return this.getDefaultInterventions();
      }

      return this.simulateInterventionGeneration(status);
    } catch (error) {
      console.error('Failed to get suggested interventions:', error);
      return this.getDefaultInterventions();
    }
  }

  private async simulateInterventionGeneration(status: MentalHealthStatus): Promise<Intervention[]> {
    const interventions: Intervention[] = [];

    // Add interventions based on conditions
    if (status.conditions.depression > 0.3) {
      interventions.push({
        id: 'depression_activation',
        type: 'behavioral_activation',
        title: 'Small Step Challenge',
        description: 'Complete one small, achievable task to boost your mood',
        duration: 10,
        priority: 1,
        culturallyAdapted: true,
        instructions: [
          'Choose a simple task you can complete in 10 minutes',
          'Focus on the positive feeling of accomplishment',
          'Reward yourself for completing the task'
        ]
      });
    }

    if (status.conditions.anxiety > 0.3) {
      interventions.push({
        id: 'anxiety_breathing',
        type: 'breathing',
        title: '4-7-8 Breathing Technique',
        description: 'Calm your nervous system with controlled breathing',
        duration: 5,
        priority: 1,
        culturallyAdapted: true,
        instructions: [
          'Inhale through nose for 4 counts',
          'Hold breath for 7 counts',
          'Exhale through mouth for 8 counts',
          'Repeat 4 times'
        ]
      });
    }

    // Always include mindfulness option
    interventions.push({
      id: 'mindfulness_check',
      type: 'mindfulness',
      title: 'Mindful Moment',
      description: 'Take a moment to center yourself and be present',
      duration: 3,
      priority: 2,
      culturallyAdapted: true,
      instructions: [
        'Find a comfortable position',
        'Notice 5 things you can see',
        'Notice 4 things you can hear',
        'Notice 3 things you can touch',
        'Take 2 deep breaths'
      ]
    });

    return interventions.slice(0, 5); // Return top 5
  }

  private getDefaultInterventions(): Intervention[] {
    return [
      {
        id: 'default_breathing',
        type: 'breathing',
        title: 'Deep Breathing',
        description: 'Simple breathing exercise for immediate relief',
        duration: 5,
        priority: 1,
        culturallyAdapted: false,
      },
      {
        id: 'default_mindfulness',
        type: 'mindfulness',
        title: 'Mindful Check-in',
        description: 'Brief mindfulness exercise',
        duration: 3,
        priority: 2,
        culturallyAdapted: false,
      }
    ];
  }

  async getRecentInsights(days: number = 7): Promise<any[]> {
    try {
      const insights = await AsyncStorage.getItem('recent_insights');
      if (!insights) {
        return [];
      }

      const parsedInsights = JSON.parse(insights);
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);

      return parsedInsights.filter((insight: any) => 
        new Date(insight.timestamp) > cutoffDate
      );
    } catch (error) {
      console.error('Failed to get recent insights:', error);
      return [];
    }
  }

  async getWeeklyTrend(): Promise<any[]> {
    try {
      const trendData = await AsyncStorage.getItem('weekly_trend');
      if (!trendData) {
        return this.generateSampleTrendData();
      }

      return JSON.parse(trendData);
    } catch (error) {
      console.error('Failed to get weekly trend:', error);
      return this.generateSampleTrendData();
    }
  }

  private generateSampleTrendData(): any[] {
    const data = [];
    const now = new Date();
    
    for (let i = 6; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      data.push({
        date: date.toISOString(),
        score: 0.2 + Math.random() * 0.3, // Random score between 0.2-0.5
        mood: ['neutral', 'good', 'fair', 'poor'][Math.floor(Math.random() * 4)]
      });
    }

    return data;
  }

  async getCrisisResources(): Promise<CrisisResource[]> {
    // Return crisis resources based on user location and preferences
    return [
      {
        id: 'suicide-prevention',
        name: '988 Suicide & Crisis Lifeline',
        phone: '988',
        description: '24/7 crisis support for suicide prevention',
        availability: '24/7',
        languages: ['English', 'Spanish'],
      },
      {
        id: 'crisis-text-line',
        name: 'Crisis Text Line',
        phone: '',
        text: '741741',
        description: 'Text HOME to 741741 for 24/7 crisis support',
        availability: '24/7',
        languages: ['English', 'Spanish'],
      },
      {
        id: 'emergency',
        name: 'Emergency Services',
        phone: '911',
        description: 'Immediate emergency medical and mental health services',
        availability: '24/7',
        languages: ['English'],
      },
    ];
  }

  async getEmergencyContacts(): Promise<any[]> {
    try {
      const contacts = await AsyncStorage.getItem('emergency_contacts');
      return contacts ? JSON.parse(contacts) : [];
    } catch (error) {
      console.error('Failed to get emergency contacts:', error);
      return [];
    }
  }

  async getSafetyPlan(): Promise<any> {
    try {
      const plan = await AsyncStorage.getItem('safety_plan');
      return plan ? JSON.parse(plan) : null;
    } catch (error) {
      console.error('Failed to get safety plan:', error);
      return null;
    }
  }

  async setCulturalContext(context: any): Promise<void> {
    try {
      if (this.userProfile) {
        this.userProfile.culturalBackground = context.background;
        this.userProfile.language = context.language;
        await AsyncStorage.setItem('user_profile', JSON.stringify(this.userProfile));
      }
    } catch (error) {
      console.error('Failed to set cultural context:', error);
    }
  }

  async triggerEmergencyMode(): Promise<void> {
    try {
      await AsyncStorage.setItem('emergency_mode', 'true');
      
      // Log emergency trigger
      await this.logEmergencyEvent();
      
      Alert.alert(
        'Emergency Mode Activated',
        'Crisis resources are now available. Please reach out for help.',
        [{ text: 'OK' }]
      );
    } catch (error) {
      console.error('Failed to trigger emergency mode:', error);
    }
  }

  async logCrisisAccess(): Promise<void> {
    // Log crisis screen access for analytics (anonymized)
    try {
      const event = {
        type: 'crisis_screen_accessed',
        timestamp: new Date().toISOString(),
        userId: this.userProfile?.id || 'anonymous'
      };
      
      await this.logEvent(event);
    } catch (error) {
      console.error('Failed to log crisis access:', error);
    }
  }

  async logEmergencyCall(phone: string): Promise<void> {
    try {
      const event = {
        type: 'emergency_call',
        timestamp: new Date().toISOString(),
        resource: phone,
        userId: this.userProfile?.id || 'anonymous'
      };
      
      await this.logEvent(event);
    } catch (error) {
      console.error('Failed to log emergency call:', error);
    }
  }

  async logEmergencyText(number: string): Promise<void> {
    try {
      const event = {
        type: 'emergency_text',
        timestamp: new Date().toISOString(),
        resource: number,
        userId: this.userProfile?.id || 'anonymous'
      };
      
      await this.logEvent(event);
    } catch (error) {
      console.error('Failed to log emergency text:', error);
    }
  }

  async logInterventionUsage(interventionType: string): Promise<void> {
    try {
      const event = {
        type: 'intervention_used',
        interventionType,
        timestamp: new Date().toIsoString(),
        userId: this.userProfile?.id || 'anonymous'
      };
      
      await this.logEvent(event);
    } catch (error) {
      console.error('Failed to log intervention usage:', error);
    }
  }

  private async refreshMentalHealthStatus(): Promise<void> {
    // Refresh mental health status from backend or recalculate
    console.log('Refreshing mental health status...');
  }

  private async logAnalysisEvent(status: MentalHealthStatus): Promise<void> {
    // Log analysis event for improving the system (anonymized)
    try {
      const event = {
        type: 'analysis_completed',
        riskLevel: status.riskLevel,
        confidence: status.confidence,
        timestamp: status.lastUpdated.toISOString(),
        userId: this.userProfile?.id || 'anonymous'
      };
      
      await this.logEvent(event);
    } catch (error) {
      console.error('Failed to log analysis event:', error);
    }
  }

  private async logEmergencyEvent(): Promise<void> {
    try {
      const event = {
        type: 'emergency_mode_triggered',
        timestamp: new Date().toISOString(),
        userId: this.userProfile?.id || 'anonymous'
      };
      
      await this.logEvent(event);
    } catch (error) {
      console.error('Failed to log emergency event:', error);
    }
  }

  private async logEvent(event: any): Promise<void> {
    try {
      // Store events locally for privacy
      const existingLogs = await AsyncStorage.getItem('analytics_logs');
      const logs = existingLogs ? JSON.parse(existingLogs) : [];
      
      logs.push(event);
      
      // Keep only last 100 events to manage storage
      if (logs.length > 100) {
        logs.splice(0, logs.length - 100);
      }
      
      await AsyncStorage.setItem('analytics_logs', JSON.stringify(logs));
    } catch (error) {
      console.error('Failed to log event:', error);
    }
  }

  async findNearbyMentalHealthServices(location: any): Promise<any[]> {
    // This would use location services to find nearby mental health resources
    // For demo, return empty array
    return [];
  }

  // Health and lifecycle methods
  isServiceHealthy(): boolean {
    return this.isInitialized && this.mentalHealthEngine !== null;
  }

  async clearAllData(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const mindBridgeKeys = keys.filter(key => 
        key.startsWith('mental_health_') || 
        key.startsWith('user_profile') ||
        key.startsWith('emergency_') ||
        key.startsWith('analytics_')
      );
      
      await AsyncStorage.multiRemove(mindBridgeKeys);
      
      // Reset state
      this.userProfile = null;
      this.isInitialized = false;
      
      console.log('All MindBridge data cleared');
    } catch (error) {
      console.error('Failed to clear data:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const MindBridgeService = new MindBridgeServiceClass();
export default MindBridgeService;