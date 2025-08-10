/**
 * MindBridge Onboarding Screen
 * 
 * Privacy-focused onboarding flow with informed consent
 * and comprehensive privacy setup
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Animated,
  Dimensions,
  Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Icon from 'react-native-vector-icons/MaterialIcons';
import LinearGradient from 'react-native-linear-gradient';

// Services
import { PrivacyService } from '../services/PrivacyService';
import { MindBridgeService } from '../services/MindBridgeService';

// Utils
import { Colors } from '../utils/Colors';
import { Fonts } from '../utils/Fonts';

// Components
import { PrivacyConsentModal } from '../components/PrivacyConsentModal';
import { CulturalContextSetup } from '../components/CulturalContextSetup';
import { PrivacyLevelSelector } from '../components/PrivacyLevelSelector';

const { width: screenWidth } = Dimensions.get('window');

interface OnboardingScreenProps {
  onComplete: () => void;
}

interface OnboardingStep {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  icon: string;
  component?: React.ComponentType<any>;
  privacyLevel: 'low' | 'medium' | 'high' | 'critical';
}

const onboardingSteps: OnboardingStep[] = [
  {
    id: 'welcome',
    title: 'Welcome to MindBridge',
    subtitle: 'Your Private Mental Health Companion',
    description: 'MindBridge uses AI to detect early signs of mental health concerns while keeping your data completely private and secure.',
    icon: 'psychology',
    privacyLevel: 'low'
  },
  {
    id: 'privacy',
    title: 'Privacy First',
    subtitle: '100% Local Processing',
    description: 'All analysis happens on your device. Your personal data never leaves your phone, ensuring complete privacy.',
    icon: 'security',
    privacyLevel: 'critical'
  },
  {
    id: 'how-it-works',
    title: 'How It Works',
    subtitle: 'AI-Powered Early Detection',
    description: 'Our AI analyzes patterns in your text and behavior to identify potential mental health concerns and suggest personalized interventions.',
    icon: 'auto_awesome',
    privacyLevel: 'medium'
  },
  {
    id: 'consent',
    title: 'Informed Consent',
    subtitle: 'Understanding Your Rights',
    description: 'Review our privacy policy and consent to data processing. You can withdraw consent at any time.',
    icon: 'verified_user',
    component: PrivacyConsentModal,
    privacyLevel: 'critical'
  },
  {
    id: 'privacy-level',
    title: 'Privacy Settings',
    subtitle: 'Choose Your Protection Level',
    description: 'Select how much privacy protection you want. Higher levels provide more security but may reduce some features.',
    icon: 'tune',
    component: PrivacyLevelSelector,
    privacyLevel: 'high'
  },
  {
    id: 'cultural-context',
    title: 'Cultural Adaptation',
    subtitle: 'Personalized for You',
    description: 'Help us understand your cultural background so we can provide more relevant and sensitive mental health support.',
    icon: 'public',
    component: CulturalContextSetup,
    privacyLevel: 'medium'
  },
  {
    id: 'ready',
    title: 'You\'re All Set!',
    subtitle: 'Start Your Mental Health Journey',
    description: 'MindBridge is now configured for your privacy preferences. Begin with a quick check-in to establish your baseline.',
    icon: 'celebration',
    privacyLevel: 'low'
  }
];

const OnboardingScreen: React.FC<OnboardingScreenProps> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [consentGiven, setConsentGiven] = useState(false);
  const [privacyLevel, setPrivacyLevel] = useState<'minimal' | 'moderate' | 'high' | 'maximum'>('maximum');
  const [culturalContext, setCulturalContext] = useState<any>(null);

  const scrollX = useRef(new Animated.Value(0)).current;
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    // Initialize privacy service
    initializePrivacySettings();
  }, []);

  const initializePrivacySettings = async () => {
    try {
      await PrivacyService.initializeOnboarding();
    } catch (error) {
      console.error('Privacy initialization failed:', error);
      Alert.alert(
        'Privacy Setup Error',
        'There was a problem setting up privacy protections. Please try again.',
        [{ text: 'OK' }]
      );
    }
  };

  const handleNextStep = async () => {
    const step = onboardingSteps[currentStep];
    
    // Mark current step as completed
    setCompletedSteps(prev => new Set(prev).add(step.id));

    // Handle step-specific logic
    switch (step.id) {
      case 'consent':
        if (!consentGiven) {
          setShowPrivacyModal(true);
          return;
        }
        break;
      
      case 'privacy-level':
        await PrivacyService.setPrivacyLevel(privacyLevel);
        break;
      
      case 'cultural-context':
        if (culturalContext) {
          await MindBridgeService.setCulturalContext(culturalContext);
        }
        break;
    }

    // Move to next step or complete onboarding
    if (currentStep < onboardingSteps.length - 1) {
      const nextStep = currentStep + 1;
      setCurrentStep(nextStep);
      
      // Animate to next slide
      scrollViewRef.current?.scrollTo({
        x: nextStep * screenWidth,
        animated: true
      });
    } else {
      await completeOnboarding();
    }
  };

  const handlePreviousStep = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      setCurrentStep(prevStep);
      
      scrollViewRef.current?.scrollTo({
        x: prevStep * screenWidth,
        animated: true
      });
    }
  };

  const handleConsentGiven = async (consent: boolean, preferences: any) => {
    setConsentGiven(consent);
    setShowPrivacyModal(false);
    
    if (consent) {
      await PrivacyService.recordConsent(preferences);
      handleNextStep();
    } else {
      Alert.alert(
        'Consent Required',
        'You must consent to data processing to use MindBridge. Your privacy is protected by our zero-knowledge architecture.',
        [{ text: 'Review Again', onPress: () => setShowPrivacyModal(true) }]
      );
    }
  };

  const completeOnboarding = async () => {
    try {
      // Finalize privacy setup
      await PrivacyService.finalizeOnboardingSetup();
      
      // Initialize MindBridge service with user preferences
      await MindBridgeService.initializeWithUserPreferences({
        privacyLevel,
        culturalContext,
        consentTimestamp: new Date().toISOString()
      });

      // Create initial baseline assessment
      await MindBridgeService.createInitialBaseline();

      // Complete onboarding
      onComplete();
      
    } catch (error) {
      console.error('Onboarding completion failed:', error);
      Alert.alert(
        'Setup Error',
        'There was a problem completing the setup. Please try again.',
        [{ text: 'OK' }]
      );
    }
  };

  const renderStep = (step: OnboardingStep, index: number) => {
    const isActive = index === currentStep;
    
    return (
      <View key={step.id} style={[styles.stepContainer, { width: screenWidth }]}>
        <View style={styles.stepContent}>
          {/* Privacy Level Indicator */}
          <View style={[styles.privacyIndicator, { backgroundColor: getPrivacyColor(step.privacyLevel) }]}>
            <Text style={styles.privacyText}>
              {step.privacyLevel.toUpperCase()} PRIVACY
            </Text>
          </View>

          {/* Icon */}
          <View style={styles.iconContainer}>
            <LinearGradient
              colors={[Colors.primary, Colors.primaryLight]}
              style={styles.iconGradient}
            >
              <Icon name={step.icon} size={48} color={Colors.white} />
            </LinearGradient>
          </View>

          {/* Content */}
          <Text style={styles.stepTitle}>{step.title}</Text>
          <Text style={styles.stepSubtitle}>{step.subtitle}</Text>
          <Text style={styles.stepDescription}>{step.description}</Text>

          {/* Step-specific Component */}
          {step.component && isActive && (
            <View style={styles.componentContainer}>
              <step.component
                onPrivacyLevelChange={setPrivacyLevel}
                onCulturalContextChange={setCulturalContext}
                currentPrivacyLevel={privacyLevel}
                currentCulturalContext={culturalContext}
              />
            </View>
          )}
        </View>
      </View>
    );
  };

  const getPrivacyColor = (level: string) => {
    switch (level) {
      case 'low': return Colors.success;
      case 'medium': return Colors.warning;
      case 'high': return Colors.primary;
      case 'critical': return Colors.error;
      default: return Colors.primary;
    }
  };

  const currentStepData = onboardingSteps[currentStep];
  const canProceed = currentStepData.id === 'consent' ? consentGiven : true;

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.progressContainer}>
          {onboardingSteps.map((_, index) => (
            <View
              key={index}
              style={[
                styles.progressDot,
                {
                  backgroundColor: index <= currentStep ? Colors.primary : Colors.lightGray,
                  width: index === currentStep ? 20 : 8,
                }
              ]}
            />
          ))}
        </View>
        
        <Text style={styles.stepCounter}>
          {currentStep + 1} of {onboardingSteps.length}
        </Text>
      </View>

      {/* Content */}
      <ScrollView
        ref={scrollViewRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        scrollEnabled={false}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { x: scrollX } } }],
          { useNativeDriver: false }
        )}
      >
        {onboardingSteps.map(renderStep)}
      </ScrollView>

      {/* Footer */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.button, styles.secondaryButton]}
          onPress={handlePreviousStep}
          disabled={currentStep === 0}
        >
          <Text style={[styles.buttonText, styles.secondaryButtonText]}>
            Back
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            styles.primaryButton,
            !canProceed && styles.disabledButton
          ]}
          onPress={handleNextStep}
          disabled={!canProceed}
        >
          <Text style={[styles.buttonText, styles.primaryButtonText]}>
            {currentStep === onboardingSteps.length - 1 ? 'Get Started' : 'Continue'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Privacy Consent Modal */}
      <PrivacyConsentModal
        visible={showPrivacyModal}
        onConsentChange={handleConsentGiven}
        onClose={() => setShowPrivacyModal(false)}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    alignItems: 'center',
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  progressDot: {
    height: 8,
    borderRadius: 4,
    marginHorizontal: 3,
    transition: 'all 0.3s ease',
  },
  stepCounter: {
    fontSize: 14,
    color: Colors.textSecondary,
    fontFamily: Fonts.regular,
  },
  stepContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  stepContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  privacyIndicator: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginBottom: 24,
    alignSelf: 'center',
  },
  privacyText: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.white,
    fontFamily: Fonts.semiBold,
  },
  iconContainer: {
    marginBottom: 32,
  },
  iconGradient: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: Colors.text,
    textAlign: 'center',
    marginBottom: 8,
    fontFamily: Fonts.bold,
  },
  stepSubtitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.primary,
    textAlign: 'center',
    marginBottom: 16,
    fontFamily: Fonts.semiBold,
  },
  stepDescription: {
    fontSize: 16,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
    paddingHorizontal: 20,
    marginBottom: 32,
    fontFamily: Fonts.regular,
  },
  componentContainer: {
    width: '100%',
    marginTop: 20,
  },
  footer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 16,
    gap: 12,
  },
  button: {
    flex: 1,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButton: {
    backgroundColor: Colors.primary,
  },
  secondaryButton: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  disabledButton: {
    backgroundColor: Colors.lightGray,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    fontFamily: Fonts.semiBold,
  },
  primaryButtonText: {
    color: Colors.white,
  },
  secondaryButtonText: {
    color: Colors.textSecondary,
  },
});

export default OnboardingScreen;