/**
 * MindBridge Dashboard Screen
 * 
 * Main dashboard showing mental health overview, insights,
 * and quick actions for users
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  Alert,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';

// Services
import { MindBridgeService } from '../services/MindBridgeService';
import { PrivacyService } from '../services/PrivacyService';
import { NotificationService } from '../services/NotificationService';

// Components
import { MentalHealthCard } from '../components/MentalHealthCard';
import { QuickActionButton } from '../components/QuickActionButton';
import { InsightCard } from '../components/InsightCard';
import { RiskIndicator } from '../components/RiskIndicator';
import { PrivacyStatusBanner } from '../components/PrivacyStatusBanner';
import { EmergencyButton } from '../components/EmergencyButton';
import { TrendChart } from '../components/TrendChart';

// Utils
import { Colors } from '../utils/Colors';
import { Fonts } from '../utils/Fonts';
import { formatDate, getTimeOfDay } from '../utils/DateUtils';

// Types
import { MentalHealthStatus, RiskLevel, Intervention } from '../types/mental-health';

const { width: screenWidth } = Dimensions.get('window');

interface DashboardState {
  mentalHealthStatus: MentalHealthStatus | null;
  recentInsights: any[];
  suggestedInterventions: Intervention[];
  riskLevel: RiskLevel;
  privacyStatus: any;
  weeklyTrend: any[];
  lastCheckIn: Date | null;
  isLoading: boolean;
  refreshing: boolean;
}

const DashboardScreen: React.FC = () => {
  const [state, setState] = useState<DashboardState>({
    mentalHealthStatus: null,
    recentInsights: [],
    suggestedInterventions: [],
    riskLevel: 'low',
    privacyStatus: null,
    weeklyTrend: [],
    lastCheckIn: null,
    isLoading: true,
    refreshing: false,
  });

  useFocusEffect(
    useCallback(() => {
      loadDashboardData();
    }, [])
  );

  useEffect(() => {
    // Setup periodic data refresh
    const refreshInterval = setInterval(() => {
      loadDashboardData(false);
    }, 5 * 60 * 1000); // Every 5 minutes

    return () => clearInterval(refreshInterval);
  }, []);

  const loadDashboardData = async (showLoading = true) => {
    try {
      if (showLoading) {
        setState(prev => ({ ...prev, isLoading: true }));
      }

      // Load all dashboard data in parallel
      const [
        mentalHealthStatus,
        recentInsights,
        suggestedInterventions,
        privacyStatus,
        weeklyTrend
      ] = await Promise.all([
        MindBridgeService.getCurrentMentalHealthStatus(),
        MindBridgeService.getRecentInsights(7), // Last 7 days
        MindBridgeService.getSuggestedInterventions(),
        PrivacyService.getPrivacyStatus(),
        MindBridgeService.getWeeklyTrend()
      ]);

      setState(prev => ({
        ...prev,
        mentalHealthStatus,
        recentInsights,
        suggestedInterventions: suggestedInterventions.slice(0, 3), // Show top 3
        riskLevel: mentalHealthStatus?.riskLevel || 'low',
        privacyStatus,
        weeklyTrend,
        lastCheckIn: mentalHealthStatus?.lastUpdated || null,
        isLoading: false,
        refreshing: false,
      }));

      // Schedule notifications based on risk level
      await scheduleNotificationsBasedOnRisk(mentalHealthStatus?.riskLevel);

    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        refreshing: false,
      }));
      
      if (showLoading) {
        Alert.alert(
          'Loading Error',
          'Could not load your mental health data. Please try again.',
          [{ text: 'OK' }]
        );
      }
    }
  };

  const scheduleNotificationsBasedOnRisk = async (riskLevel?: RiskLevel) => {
    if (!riskLevel) return;

    switch (riskLevel) {
      case 'high':
      case 'critical':
        await NotificationService.scheduleCheckInReminder(2); // 2 hours
        break;
      case 'moderate':
        await NotificationService.scheduleCheckInReminder(12); // 12 hours
        break;
      case 'low':
        await NotificationService.scheduleCheckInReminder(24); // 24 hours
        break;
    }
  };

  const handleRefresh = async () => {
    setState(prev => ({ ...prev, refreshing: true }));
    await loadDashboardData(false);
  };

  const handleQuickCheckIn = async () => {
    try {
      // Navigate to check-in with quick mode
      // navigation.navigate('CheckIn', { quickMode: true });
    } catch (error) {
      console.error('Quick check-in failed:', error);
    }
  };

  const handleEmergency = async () => {
    try {
      await MindBridgeService.triggerEmergencyMode();
      // Navigation to crisis screen will be handled by App.tsx
    } catch (error) {
      console.error('Emergency trigger failed:', error);
    }
  };

  const handleInterventionTap = (intervention: Intervention) => {
    // navigation.navigate('Intervention', { intervention });
  };

  const renderWelcomeHeader = () => {
    const timeOfDay = getTimeOfDay();
    const userName = state.mentalHealthStatus?.userProfile?.preferredName || 'Friend';
    
    return (
      <View style={styles.welcomeContainer}>
        <Text style={styles.welcomeText}>
          Good {timeOfDay}, {userName}
        </Text>
        <Text style={styles.dateText}>
          {formatDate(new Date(), 'EEEE, MMMM d')}
        </Text>
      </View>
    );
  };

  const renderMentalHealthOverview = () => {
    if (!state.mentalHealthStatus) {
      return (
        <View style={styles.placeholderCard}>
          <Text style={styles.placeholderText}>
            Complete your first check-in to see your mental health overview
          </Text>
          <TouchableOpacity style={styles.checkInButton} onPress={handleQuickCheckIn}>
            <Text style={styles.checkInButtonText}>Start Check-In</Text>
          </TouchableOpacity>
        </View>
      );
    }

    return (
      <MentalHealthCard
        status={state.mentalHealthStatus}
        onTap={() => {/* navigation.navigate('Insights') */}}
        showTrend
      />
    );
  };

  const renderQuickActions = () => (
    <View style={styles.quickActionsContainer}>
      <Text style={styles.sectionTitle}>Quick Actions</Text>
      <View style={styles.quickActionsRow}>
        <QuickActionButton
          icon="favorite"
          label="Check In"
          onPress={handleQuickCheckIn}
          color={Colors.primary}
        />
        <QuickActionButton
          icon="book"
          label="Journal"
          onPress={() => {/* navigation.navigate('Journal') */}}
          color={Colors.success}
        />
        <QuickActionButton
          icon="analytics"
          label="Insights"
          onPress={() => {/* navigation.navigate('Insights') */}}
          color={Colors.warning}
        />
        <QuickActionButton
          icon="people"
          label="Support"
          onPress={() => {/* navigation.navigate('Community') */}}
          color={Colors.info}
        />
      </View>
    </View>
  );

  const renderRiskIndicator = () => {
    if (state.riskLevel === 'low') return null;

    return (
      <RiskIndicator
        level={state.riskLevel}
        onActionPress={() => {
          if (state.riskLevel === 'critical') {
            handleEmergency();
          } else {
            // navigation.navigate('Intervention');
          }
        }}
      />
    );
  };

  const renderSuggestedInterventions = () => {
    if (state.suggestedInterventions.length === 0) return null;

    return (
      <View style={styles.interventionsContainer}>
        <Text style={styles.sectionTitle}>Suggested for You</Text>
        {state.suggestedInterventions.map((intervention, index) => (
          <TouchableOpacity
            key={intervention.id || index}
            style={styles.interventionCard}
            onPress={() => handleInterventionTap(intervention)}
          >
            <View style={styles.interventionHeader}>
              <Icon 
                name={getInterventionIcon(intervention.type)} 
                size={24} 
                color={Colors.primary} 
              />
              <View style={styles.interventionContent}>
                <Text style={styles.interventionTitle}>{intervention.title}</Text>
                <Text style={styles.interventionDescription}>
                  {intervention.description}
                </Text>
              </View>
              <Icon name="chevron-right" size={24} color={Colors.textSecondary} />
            </View>
            <View style={styles.interventionFooter}>
              <Text style={styles.interventionDuration}>
                {intervention.duration} min
              </Text>
              <View style={[
                styles.priorityBadge,
                { backgroundColor: getPriorityColor(intervention.priority) }
              ]}>
                <Text style={styles.priorityText}>
                  Priority {intervention.priority}
                </Text>
              </View>
            </View>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const renderRecentInsights = () => {
    if (state.recentInsights.length === 0) return null;

    return (
      <View style={styles.insightsContainer}>
        <Text style={styles.sectionTitle}>Recent Insights</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {state.recentInsights.map((insight, index) => (
            <InsightCard
              key={insight.id || index}
              insight={insight}
              onTap={() => {/* navigation.navigate('Insights', { insightId: insight.id }) */}}
            />
          ))}
        </ScrollView>
      </View>
    );
  };

  const renderWeeklyTrend = () => {
    if (state.weeklyTrend.length === 0) return null;

    return (
      <View style={styles.trendContainer}>
        <Text style={styles.sectionTitle}>This Week's Trend</Text>
        <TrendChart
          data={state.weeklyTrend}
          width={screenWidth - 40}
          height={120}
          color={Colors.primary}
        />
      </View>
    );
  };

  const getInterventionIcon = (type: string): string => {
    switch (type) {
      case 'mindfulness': return 'self-improvement';
      case 'breathing': return 'air';
      case 'exercise': return 'directions-run';
      case 'journaling': return 'create';
      case 'social': return 'people';
      case 'therapy': return 'psychology';
      default: return 'lightbulb';
    }
  };

  const getPriorityColor = (priority: number): string => {
    switch (priority) {
      case 1: return Colors.error;
      case 2: return Colors.warning;
      case 3: return Colors.info;
      default: return Colors.success;
    }
  };

  if (state.isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading your dashboard...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl
            refreshing={state.refreshing}
            onRefresh={handleRefresh}
            colors={[Colors.primary]}
            tintColor={Colors.primary}
          />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Privacy Status Banner */}
        {state.privacyStatus && (
          <PrivacyStatusBanner status={state.privacyStatus} />
        )}

        {/* Welcome Header */}
        {renderWelcomeHeader()}

        {/* Risk Indicator */}
        {renderRiskIndicator()}

        {/* Mental Health Overview */}
        {renderMentalHealthOverview()}

        {/* Quick Actions */}
        {renderQuickActions()}

        {/* Suggested Interventions */}
        {renderSuggestedInterventions()}

        {/* Weekly Trend */}
        {renderWeeklyTrend()}

        {/* Recent Insights */}
        {renderRecentInsights()}

        {/* Emergency Button - Always accessible */}
        <View style={styles.emergencyContainer}>
          <EmergencyButton onPress={handleEmergency} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollView: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: Colors.textSecondary,
    fontFamily: Fonts.regular,
  },
  welcomeContainer: {
    padding: 20,
    paddingBottom: 10,
  },
  welcomeText: {
    fontSize: 24,
    fontWeight: '700',
    color: Colors.text,
    fontFamily: Fonts.bold,
  },
  dateText: {
    fontSize: 16,
    color: Colors.textSecondary,
    fontFamily: Fonts.regular,
    marginTop: 4,
  },
  placeholderCard: {
    margin: 20,
    padding: 20,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 16,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: 16,
    fontFamily: Fonts.regular,
  },
  checkInButton: {
    backgroundColor: Colors.primary,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  checkInButtonText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: Fonts.semiBold,
  },
  quickActionsContainer: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 16,
    fontFamily: Fonts.semiBold,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  interventionsContainer: {
    padding: 20,
  },
  interventionCard: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  interventionHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  interventionContent: {
    flex: 1,
    marginLeft: 12,
    marginRight: 8,
  },
  interventionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 4,
    fontFamily: Fonts.semiBold,
  },
  interventionDescription: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 20,
    fontFamily: Fonts.regular,
  },
  interventionFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  interventionDuration: {
    fontSize: 14,
    color: Colors.textSecondary,
    fontFamily: Fonts.regular,
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  priorityText: {
    fontSize: 12,
    color: Colors.white,
    fontWeight: '600',
    fontFamily: Fonts.semiBold,
  },
  insightsContainer: {
    padding: 20,
  },
  trendContainer: {
    padding: 20,
  },
  emergencyContainer: {
    padding: 20,
    alignItems: 'center',
  },
});

export default DashboardScreen;