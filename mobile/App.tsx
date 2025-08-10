/**
 * MindBridge Mobile App
 * 
 * Privacy-first mental health detection and support application
 * Built with React Native for iOS and Android
 * 
 * Features:
 * - Offline-first architecture
 * - End-to-end encryption
 * - Real-time mental health monitoring
 * - Personalized interventions
 * - Cultural adaptation
 * - Crisis intervention
 */

import React, { useEffect, useState } from 'react';
import {
  StatusBar,
  StyleSheet,
  useColorScheme,
  Alert,
  AppState,
  AppStateStatus,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import NetInfo from '@react-native-community/netinfo';
import DeviceInfo from 'react-native-device-info';
import { request, PERMISSIONS, RESULTS } from 'react-native-permissions';
import AsyncStorage from '@react-native-community/async-storage';

// Redux store
import { store, persistor } from './src/store';

// Screens
import OnboardingScreen from './src/screens/OnboardingScreen';
import DashboardScreen from './src/screens/DashboardScreen';
import CheckInScreen from './src/screens/CheckInScreen';
import JournalScreen from './src/screens/JournalScreen';
import InsightsScreen from './src/screens/InsightsScreen';
import InterventionScreen from './src/screens/InterventionScreen';
import CommunityScreen from './src/screens/CommunityScreen';
import CrisisScreen from './src/screens/CrisisScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import LoadingScreen from './src/screens/LoadingScreen';

// Services
import { MindBridgeService } from './src/services/MindBridgeService';
import { PrivacyService } from './src/services/PrivacyService';
import { NotificationService } from './src/services/NotificationService';
import { HealthKitService } from './src/services/HealthKitService';

// Utils
import { Colors } from './src/utils/Colors';
import { Themes } from './src/utils/Themes';

// Types
import { RootStackParamList, TabParamList } from './src/types/navigation';

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<TabParamList>();

interface AppState {
  isLoading: boolean;
  isOnboarded: boolean;
  isOfflineMode: boolean;
  darkMode: boolean;
  emergencyMode: boolean;
}

const TabNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Dashboard':
              iconName = focused ? 'dashboard' : 'dashboard';
              break;
            case 'CheckIn':
              iconName = focused ? 'favorite' : 'favorite-border';
              break;
            case 'Journal':
              iconName = focused ? 'book' : 'book';
              break;
            case 'Insights':
              iconName = focused ? 'analytics' : 'analytics';
              break;
            case 'Community':
              iconName = focused ? 'people' : 'people';
              break;
            default:
              iconName = 'circle';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: Colors.primary,
        tabBarInactiveTintColor: Colors.gray,
        tabBarStyle: {
          backgroundColor: Colors.surface,
          borderTopWidth: 1,
          borderTopColor: Colors.border,
          paddingBottom: 5,
          paddingTop: 5,
          height: 60,
        },
        headerStyle: {
          backgroundColor: Colors.surface,
          elevation: 0,
          shadowOpacity: 0,
        },
        headerTitleStyle: {
          color: Colors.text,
          fontWeight: '600',
        },
      })}
    >
      <Tab.Screen 
        name="Dashboard" 
        component={DashboardScreen}
        options={{ 
          title: 'Dashboard',
          tabBarLabel: 'Home' 
        }}
      />
      <Tab.Screen 
        name="CheckIn" 
        component={CheckInScreen}
        options={{ 
          title: 'Check In',
          tabBarLabel: 'Check In' 
        }}
      />
      <Tab.Screen 
        name="Journal" 
        component={JournalScreen}
        options={{ 
          title: 'Journal',
          tabBarLabel: 'Journal' 
        }}
      />
      <Tab.Screen 
        name="Insights" 
        component={InsightsScreen}
        options={{ 
          title: 'Insights',
          tabBarLabel: 'Insights' 
        }}
      />
      <Tab.Screen 
        name="Community" 
        component={CommunityScreen}
        options={{ 
          title: 'Community',
          tabBarLabel: 'Support' 
        }}
      />
    </Tab.Navigator>
  );
};

const App: React.FC = () => {
  const isDarkMode = useColorScheme() === 'dark';
  
  const [appState, setAppState] = useState<AppState>({
    isLoading: true,
    isOnboarded: false,
    isOfflineMode: false,
    darkMode: isDarkMode,
    emergencyMode: false
  });

  useEffect(() => {
    initializeApp();
  }, []);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', handleAppStateChange);
    return () => subscription?.remove();
  }, []);

  const initializeApp = async () => {
    try {
      // Check if user has completed onboarding
      const onboardingComplete = await AsyncStorage.getItem('onboarding_complete');
      
      // Initialize core services
      await MindBridgeService.initialize();
      await PrivacyService.initialize();
      await NotificationService.initialize();
      
      // Request necessary permissions
      await requestPermissions();
      
      // Setup network monitoring
      setupNetworkMonitoring();
      
      // Check for emergency mode
      const emergencyMode = await checkEmergencyMode();
      
      // Initialize health data if available
      if (await DeviceInfo.hasSystemFeature('healthkit')) {
        await HealthKitService.initialize();
      }
      
      setAppState(prev => ({
        ...prev,
        isLoading: false,
        isOnboarded: !!onboardingComplete,
        emergencyMode
      }));
      
    } catch (error) {
      console.error('App initialization failed:', error);
      Alert.alert(
        'Initialization Error',
        'There was a problem starting the app. Please try again.',
        [{ text: 'OK', onPress: () => setAppState(prev => ({ ...prev, isLoading: false })) }]
      );
    }
  };

  const requestPermissions = async () => {
    try {
      // Notification permissions
      const notificationResult = await request(
        PERMISSIONS.IOS.NOTIFICATIONS || PERMISSIONS.ANDROID.POST_NOTIFICATIONS
      );
      
      // Health data permissions (iOS)
      if (DeviceInfo.getSystemName() === 'iOS') {
        await request(PERMISSIONS.IOS.HEALTH_SHARE);
        await request(PERMISSIONS.IOS.HEALTH_UPDATE);
      }
      
      // Microphone permission for voice journaling
      await request(
        PERMISSIONS.IOS.MICROPHONE || PERMISSIONS.ANDROID.RECORD_AUDIO
      );
      
      // Camera permission for future features
      await request(
        PERMISSIONS.IOS.CAMERA || PERMISSIONS.ANDROID.CAMERA
      );
      
    } catch (error) {
      console.error('Permission request failed:', error);
    }
  };

  const setupNetworkMonitoring = () => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setAppState(prev => ({
        ...prev,
        isOfflineMode: !state.isConnected
      }));
    });

    return unsubscribe;
  };

  const checkEmergencyMode = async (): Promise<boolean> => {
    try {
      // Check if emergency mode was activated
      const emergencyFlag = await AsyncStorage.getItem('emergency_mode');
      return emergencyFlag === 'true';
    } catch (error) {
      return false;
    }
  };

  const handleAppStateChange = async (nextAppState: AppStateStatus) => {
    if (nextAppState === 'background') {
      // Save current state and enable privacy protections
      await PrivacyService.enableBackgroundProtection();
    } else if (nextAppState === 'active') {
      // Resume normal operation
      await PrivacyService.disableBackgroundProtection();
      
      // Check for emergency situations
      const emergencyMode = await checkEmergencyMode();
      if (emergencyMode !== appState.emergencyMode) {
        setAppState(prev => ({ ...prev, emergencyMode }));
      }
    }
  };

  const handleOnboardingComplete = async () => {
    await AsyncStorage.setItem('onboarding_complete', 'true');
    setAppState(prev => ({ ...prev, isOnboarded: true }));
  };

  if (appState.isLoading) {
    return <LoadingScreen />;
  }

  // Emergency mode - show crisis screen immediately
  if (appState.emergencyMode) {
    return (
      <GestureHandlerRootView style={styles.container}>
        <NavigationContainer theme={isDarkMode ? Themes.dark : Themes.light}>
          <StatusBar
            barStyle={isDarkMode ? 'light-content' : 'dark-content'}
            backgroundColor={Colors.surface}
          />
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Crisis" component={CrisisScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </GestureHandlerRootView>
    );
  }

  return (
    <Provider store={store}>
      <PersistGate loading={<LoadingScreen />} persistor={persistor}>
        <GestureHandlerRootView style={styles.container}>
          <NavigationContainer theme={isDarkMode ? Themes.dark : Themes.light}>
            <StatusBar
              barStyle={isDarkMode ? 'light-content' : 'dark-content'}
              backgroundColor={Colors.surface}
            />
            <Stack.Navigator 
              screenOptions={{ 
                headerShown: false,
                gestureEnabled: true,
                animation: 'slide_from_right'
              }}
            >
              {!appState.isOnboarded ? (
                <Stack.Screen name="Onboarding">
                  {props => (
                    <OnboardingScreen 
                      {...props} 
                      onComplete={handleOnboardingComplete}
                    />
                  )}
                </Stack.Screen>
              ) : (
                <>
                  <Stack.Screen name="Main" component={TabNavigator} />
                  <Stack.Screen 
                    name="Intervention" 
                    component={InterventionScreen}
                    options={{
                      headerShown: true,
                      title: 'Intervention',
                      presentation: 'modal'
                    }}
                  />
                  <Stack.Screen 
                    name="Settings" 
                    component={SettingsScreen}
                    options={{
                      headerShown: true,
                      title: 'Settings',
                      presentation: 'modal'
                    }}
                  />
                  <Stack.Screen 
                    name="Crisis" 
                    component={CrisisScreen}
                    options={{
                      headerShown: false,
                      gestureEnabled: false,
                      presentation: 'fullScreenModal'
                    }}
                  />
                </>
              )}
            </Stack.Navigator>
          </NavigationContainer>
        </GestureHandlerRootView>
      </PersistGate>
    </Provider>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
});

export default App;