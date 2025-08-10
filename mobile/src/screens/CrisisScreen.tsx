/**
 * MindBridge Crisis Screen
 * 
 * Emergency intervention screen for users experiencing crisis
 * or high-risk mental health situations
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Linking,
  Vibration,
  Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';

// Services
import { MindBridgeService } from '../services/MindBridgeService';
import { NotificationService } from '../services/NotificationService';
import { LocationService } from '../services/LocationService';

// Components
import { BreathingExercise } from '../components/BreathingExercise';
import { CrisisHotlineCard } from '../components/CrisisHotlineCard';
import { EmergencyContactCard } from '../components/EmergencyContactCard';
import { SafetyPlanCard } from '../components/SafetyPlanCard';

// Utils
import { Colors } from '../utils/Colors';
import { Fonts } from '../utils/Fonts';
import { HapticFeedback } from '../utils/HapticUtils';

// Types
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

interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  relationship: string;
}

const CrisisScreen: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'immediate' | 'resources' | 'plan'>('immediate');
  const [crisisResources, setCrisisResources] = useState<CrisisResource[]>([]);
  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([]);
  const [safetyPlan, setSafetyPlan] = useState<any>(null);
  const [isBreathing, setIsBreathing] = useState(false);
  const [pulseAnimation] = useState(new Animated.Value(1));

  useEffect(() => {
    initializeCrisisResources();
    startPulseAnimation();
    
    // Vibrate to get attention
    Vibration.vibrate([0, 500, 200, 500]);
    
    // Log crisis screen access for analytics
    MindBridgeService.logCrisisAccess();
  }, []);

  const initializeCrisisResources = async () => {
    try {
      // Load crisis resources based on user location and preferences
      const resources = await MindBridgeService.getCrisisResources();
      const contacts = await MindBridgeService.getEmergencyContacts();
      const plan = await MindBridgeService.getSafetyPlan();

      setCrisisResources(resources);
      setEmergencyContacts(contacts);
      setSafetyPlan(plan);
    } catch (error) {
      console.error('Failed to load crisis resources:', error);
      // Load default resources
      loadDefaultCrisisResources();
    }
  };

  const loadDefaultCrisisResources = () => {
    const defaultResources: CrisisResource[] = [
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
    setCrisisResources(defaultResources);
  };

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnimation, {
          toValue: 1.1,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnimation, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const handleCallHotline = async (phone: string) => {
    try {
      HapticFeedback.impact('heavy');
      
      const url = `tel:${phone}`;
      const canOpen = await Linking.canOpenURL(url);
      
      if (canOpen) {
        await Linking.openURL(url);
        // Log the call for follow-up
        MindBridgeService.logEmergencyCall(phone);
      } else {
        Alert.alert(
          'Unable to make call',
          `Please dial ${phone} manually to get help.`,
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('Failed to make emergency call:', error);
      Alert.alert(
        'Call Failed',
        `Please dial ${phone} manually to get immediate help.`,
        [{ text: 'OK' }]
      );
    }
  };

  const handleTextCrisis = async (number: string) => {
    try {
      HapticFeedback.impact('medium');
      
      const url = `sms:${number}`;
      const canOpen = await Linking.canOpenURL(url);
      
      if (canOpen) {
        await Linking.openURL(url);
        MindBridgeService.logEmergencyText(number);
      } else {
        Alert.alert(
          'Unable to send text',
          `Please text ${number} manually to get help.`,
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('Failed to send crisis text:', error);
    }
  };

  const handleCallEmergencyContact = async (contact: EmergencyContact) => {
    Alert.alert(
      `Call ${contact.name}?`,
      `This will call ${contact.name} (${contact.relationship}) at ${contact.phone}`,
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Call', 
          onPress: () => handleCallHotline(contact.phone),
          style: 'default'
        }
      ]
    );
  };

  const handleStartBreathing = () => {
    setIsBreathing(true);
    HapticFeedback.success();
    // Track breathing exercise usage
    MindBridgeService.logInterventionUsage('breathing_crisis');
  };

  const handleFindNearbyHelp = async () => {
    try {
      const location = await LocationService.getCurrentLocation();
      const nearbyResources = await MindBridgeService.findNearbyMentalHealthServices(location);
      
      // Show nearby resources or navigate to maps
      if (nearbyResources.length > 0) {
        // navigation.navigate('NearbyResources', { resources: nearbyResources });
      } else {
        Alert.alert(
          'Find Help Nearby',
          'Opening maps to find nearby mental health services',
          [
            { text: 'Cancel', style: 'cancel' },
            { 
              text: 'Open Maps',
              onPress: () => Linking.openURL('maps://search?query=mental+health+crisis+center+near+me')
            }
          ]
        );
      }
    } catch (error) {
      console.error('Failed to find nearby help:', error);
      Alert.alert(
        'Location Error',
        'Unable to find nearby resources. Please use the hotlines above for immediate help.',
        [{ text: 'OK' }]
      );
    }
  };

  const renderImmediateHelp = () => (
    <View style={styles.section}>
      <Text style={styles.urgentTitle}>You're Not Alone</Text>
      <Text style={styles.urgentSubtitle}>
        If you're having thoughts of suicide or self-harm, please reach out for help immediately.
      </Text>

      {/* Primary Crisis Hotlines */}
      <View style={styles.hotlinesContainer}>
        {crisisResources.slice(0, 3).map(resource => (
          <CrisisHotlineCard
            key={resource.id}
            resource={resource}
            onCall={() => handleCallHotline(resource.phone)}
            onText={resource.text ? () => handleTextCrisis(resource.text!) : undefined}
            priority="high"
          />
        ))}
      </View>

      {/* Immediate Coping Actions */}
      <View style={styles.copingActionsContainer}>
        <Text style={styles.sectionSubtitle}>Immediate Coping Strategies</Text>
        
        <TouchableOpacity
          style={[styles.actionButton, styles.breathingButton]}
          onPress={handleStartBreathing}
        >
          <Animated.View style={[styles.actionIcon, { transform: [{ scale: pulseAnimation }] }]}>
            <Icon name="air" size={32} color={Colors.white} />
          </Animated.View>
          <View style={styles.actionContent}>
            <Text style={styles.actionTitle}>Start Breathing Exercise</Text>
            <Text style={styles.actionDescription}>
              Guided breathing to help calm your mind right now
            </Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.contactButton]}
          onPress={() => setActiveSection('resources')}
        >
          <View style={styles.actionIcon}>
            <Icon name="people" size={32} color={Colors.white} />
          </View>
          <View style={styles.actionContent}>
            <Text style={styles.actionTitle}>Contact Someone</Text>
            <Text style={styles.actionDescription}>
              Reach out to your support network
            </Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.locationButton]}
          onPress={handleFindNearbyHelp}
        >
          <View style={styles.actionIcon}>
            <Icon name="location-on" size={32} color={Colors.white} />
          </View>
          <View style={styles.actionContent}>
            <Text style={styles.actionTitle}>Find Help Nearby</Text>
            <Text style={styles.actionDescription}>
              Locate nearby crisis centers and emergency rooms
            </Text>
          </View>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderResourcesAndContacts = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Support Resources</Text>

      {/* Emergency Contacts */}
      {emergencyContacts.length > 0 && (
        <View style={styles.emergencyContactsContainer}>
          <Text style={styles.sectionSubtitle}>Your Emergency Contacts</Text>
          {emergencyContacts.map(contact => (
            <EmergencyContactCard
              key={contact.id}
              contact={contact}
              onCall={() => handleCallEmergencyContact(contact)}
            />
          ))}
        </View>
      )}

      {/* Additional Crisis Resources */}
      <View style={styles.additionalResourcesContainer}>
        <Text style={styles.sectionSubtitle}>Additional Resources</Text>
        {crisisResources.slice(3).map(resource => (
          <CrisisHotlineCard
            key={resource.id}
            resource={resource}
            onCall={() => handleCallHotline(resource.phone)}
            onText={resource.text ? () => handleTextCrisis(resource.text!) : undefined}
            priority="medium"
          />
        ))}
      </View>

      {/* Warning Signs */}
      <View style={styles.warningSignsContainer}>
        <Text style={styles.sectionSubtitle}>When to Seek Immediate Help</Text>
        <View style={styles.warningSignsList}>
          <Text style={styles.warningSign}>• Thoughts of suicide or self-harm</Text>
          <Text style={styles.warningSign}>• Feeling hopeless or trapped</Text>
          <Text style={styles.warningSign}>• Severe mood swings or agitation</Text>
          <Text style={styles.warningSign}>• Withdrawal from friends and family</Text>
          <Text style={styles.warningSign}>• Increased use of alcohol or drugs</Text>
          <Text style={styles.warningSign}>• Giving away possessions</Text>
        </View>
      </View>
    </View>
  );

  const renderSafetyPlan = () => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>Your Safety Plan</Text>
      
      {safetyPlan ? (
        <SafetyPlanCard
          plan={safetyPlan}
          onEdit={() => {/* navigation.navigate('SafetyPlanEditor') */}}
        />
      ) : (
        <View style={styles.noSafetyPlanContainer}>
          <Icon name="shield" size={48} color={Colors.textSecondary} />
          <Text style={styles.noSafetyPlanTitle}>No Safety Plan Yet</Text>
          <Text style={styles.noSafetyPlanDescription}>
            A safety plan helps you identify warning signs and coping strategies for crisis situations.
          </Text>
          <TouchableOpacity
            style={styles.createSafetyPlanButton}
            onPress={() => {/* navigation.navigate('SafetyPlanCreator') */}}
          >
            <Text style={styles.createSafetyPlanText}>Create Safety Plan</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <LinearGradient
          colors={[Colors.error, Colors.errorDark]}
          style={styles.headerGradient}
        >
          <View style={styles.headerContent}>
            <Icon name="emergency" size={32} color={Colors.white} />
            <Text style={styles.headerTitle}>Crisis Support</Text>
            <Text style={styles.headerSubtitle}>
              You matter. Help is available 24/7.
            </Text>
          </View>
        </LinearGradient>
      </View>

      {/* Navigation Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeSection === 'immediate' && styles.activeTab]}
          onPress={() => setActiveSection('immediate')}
        >
          <Text style={[styles.tabText, activeSection === 'immediate' && styles.activeTabText]}>
            Immediate Help
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, activeSection === 'resources' && styles.activeTab]}
          onPress={() => setActiveSection('resources')}
        >
          <Text style={[styles.tabText, activeSection === 'resources' && styles.activeTabText]}>
            Resources
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.tab, activeSection === 'plan' && styles.activeTab]}
          onPress={() => setActiveSection('plan')}
        >
          <Text style={[styles.tabText, activeSection === 'plan' && styles.activeTabText]}>
            Safety Plan
          </Text>
        </TouchableOpacity>
      </View>

      {/* Content */}
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {activeSection === 'immediate' && renderImmediateHelp()}
        {activeSection === 'resources' && renderResourcesAndContacts()}
        {activeSection === 'plan' && renderSafetyPlan()}
      </ScrollView>

      {/* Breathing Exercise Modal */}
      {isBreathing && (
        <BreathingExercise
          visible={isBreathing}
          onClose={() => setIsBreathing(false)}
          duration={300} // 5 minutes for crisis situations
        />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    overflow: 'hidden',
  },
  headerGradient: {
    paddingHorizontal: 20,
    paddingVertical: 24,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: Colors.white,
    marginTop: 8,
    fontFamily: Fonts.bold,
  },
  headerSubtitle: {
    fontSize: 16,
    color: Colors.white,
    marginTop: 4,
    textAlign: 'center',
    opacity: 0.9,
    fontFamily: Fonts.regular,
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  tab: {
    flex: 1,
    paddingVertical: 16,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: Colors.error,
  },
  tabText: {
    fontSize: 14,
    color: Colors.textSecondary,
    fontWeight: '500',
    fontFamily: Fonts.medium,
  },
  activeTabText: {
    color: Colors.error,
    fontWeight: '600',
  },
  scrollView: {
    flex: 1,
  },
  section: {
    padding: 20,
  },
  urgentTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: Colors.text,
    textAlign: 'center',
    marginBottom: 8,
    fontFamily: Fonts.bold,
  },
  urgentSubtitle: {
    fontSize: 16,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 24,
    fontFamily: Fonts.regular,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: Colors.text,
    marginBottom: 16,
    fontFamily: Fonts.bold,
  },
  sectionSubtitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 12,
    marginTop: 24,
    fontFamily: Fonts.semiBold,
  },
  hotlinesContainer: {
    marginBottom: 24,
  },
  copingActionsContainer: {
    marginTop: 8,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  breathingButton: {
    backgroundColor: Colors.primary,
  },
  contactButton: {
    backgroundColor: Colors.success,
  },
  locationButton: {
    backgroundColor: Colors.info,
  },
  actionIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  actionContent: {
    flex: 1,
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.white,
    marginBottom: 4,
    fontFamily: Fonts.semiBold,
  },
  actionDescription: {
    fontSize: 14,
    color: Colors.white,
    opacity: 0.9,
    fontFamily: Fonts.regular,
  },
  emergencyContactsContainer: {
    marginBottom: 24,
  },
  additionalResourcesContainer: {
    marginBottom: 24,
  },
  warningSignsContainer: {
    backgroundColor: Colors.surface,
    padding: 16,
    borderRadius: 12,
    marginTop: 16,
  },
  warningSignsList: {
    marginTop: 8,
  },
  warningSign: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 22,
    fontFamily: Fonts.regular,
  },
  noSafetyPlanContainer: {
    alignItems: 'center',
    padding: 32,
    backgroundColor: Colors.surface,
    borderRadius: 12,
  },
  noSafetyPlanTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.text,
    marginTop: 16,
    marginBottom: 8,
    fontFamily: Fonts.semiBold,
  },
  noSafetyPlanDescription: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 20,
    fontFamily: Fonts.regular,
  },
  createSafetyPlanButton: {
    backgroundColor: Colors.primary,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  createSafetyPlanText: {
    color: Colors.white,
    fontSize: 16,
    fontWeight: '600',
    fontFamily: Fonts.semiBold,
  },
});

export default CrisisScreen;