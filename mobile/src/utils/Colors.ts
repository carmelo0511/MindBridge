/**
 * MindBridge Color System
 * 
 * Consistent color palette for the entire application
 * Includes accessibility-compliant colors and theme support
 */

export const Colors = {
  // Primary Colors
  primary: '#2196F3',
  primaryLight: '#64B5F6',
  primaryDark: '#1976D2',
  
  // Secondary Colors
  secondary: '#9C27B0',
  secondaryLight: '#BA68C8',
  secondaryDark: '#7B1FA2',
  
  // Status Colors
  success: '#4CAF50',
  successLight: '#81C784',
  successDark: '#388E3C',
  
  warning: '#FF9800',
  warningLight: '#FFB74D',
  warningDark: '#F57C00',
  
  error: '#F44336',
  errorLight: '#EF5350',
  errorDark: '#C62828',
  
  info: '#00BCD4',
  infoLight: '#4DD0E1',
  infoDark: '#0097A7',
  
  // Neutral Colors
  text: '#212121',
  textSecondary: '#757575',
  textTertiary: '#9E9E9E',
  
  background: '#FAFAFA',
  surface: '#FFFFFF',
  surfaceVariant: '#F5F5F5',
  
  // Gray Scale
  black: '#000000',
  darkGray: '#424242',
  gray: '#757575',
  lightGray: '#BDBDBD',
  lighterGray: '#E0E0E0',
  white: '#FFFFFF',
  
  // Border Colors
  border: '#E0E0E0',
  borderLight: '#F0F0F0',
  borderDark: '#BDBDBD',
  
  // Mental Health Specific Colors
  mentalHealth: {
    excellent: '#4CAF50',    // Green
    good: '#8BC34A',         // Light Green
    fair: '#FFC107',         // Yellow
    poor: '#FF9800',         // Orange
    critical: '#F44336',     // Red
  },
  
  // Risk Level Colors
  riskLevels: {
    low: '#4CAF50',
    moderate: '#FF9800',
    high: '#F44336',
    critical: '#D32F2F',
  },
  
  // Intervention Colors
  interventions: {
    mindfulness: '#9C27B0',
    breathing: '#2196F3',
    exercise: '#FF5722',
    journaling: '#795548',
    social: '#607D8B',
    therapy: '#3F51B5',
  },
  
  // Privacy Indicator Colors
  privacy: {
    maximum: '#1B5E20',      // Dark Green
    high: '#388E3C',         // Green
    moderate: '#FFA000',     // Orange
    minimal: '#F57C00',      // Dark Orange
  },
  
  // Accessibility Colors (WCAG AA Compliant)
  accessible: {
    primaryOnWhite: '#1976D2',
    secondaryOnWhite: '#7B1FA2',
    errorOnWhite: '#C62828',
    successOnWhite: '#388E3C',
    warningOnWhite: '#F57C00',
  },
  
  // Gradient Colors
  gradients: {
    primary: ['#2196F3', '#1976D2'],
    secondary: ['#9C27B0', '#7B1FA2'],
    success: ['#4CAF50', '#388E3C'],
    warning: ['#FF9800', '#F57C00'],
    error: ['#F44336', '#C62828'],
    sunset: ['#FF5722', '#FF9800'],
    ocean: ['#2196F3', '#00BCD4'],
    forest: ['#4CAF50', '#8BC34A'],
  },
  
  // Dark Theme Colors
  dark: {
    primary: '#64B5F6',
    primaryLight: '#90CAF9',
    primaryDark: '#42A5F5',
    
    text: '#FFFFFF',
    textSecondary: '#B0BEC5',
    textTertiary: '#78909C',
    
    background: '#121212',
    surface: '#1E1E1E',
    surfaceVariant: '#2D2D2D',
    
    border: '#404040',
    borderLight: '#303030',
    borderDark: '#505050',
  },
  
  // Condition-Specific Colors
  conditions: {
    depression: '#3F51B5',
    anxiety: '#FF9800',
    ptsd: '#795548',
    bipolar: '#9C27B0',
    burnout: '#FF5722',
    generalDistress: '#607D8B',
  },
};

// Helper function to get color with opacity
export const getColorWithOpacity = (color: string, opacity: number): string => {
  // Convert hex to rgba
  const hex = color.replace('#', '');
  const r = parseInt(hex.substr(0, 2), 16);
  const g = parseInt(hex.substr(2, 2), 16);
  const b = parseInt(hex.substr(4, 2), 16);
  
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
};

// Helper function to get risk level color
export const getRiskLevelColor = (riskLevel: 'low' | 'moderate' | 'high' | 'critical'): string => {
  return Colors.riskLevels[riskLevel] || Colors.riskLevels.low;
};

// Helper function to get condition color
export const getConditionColor = (condition: string): string => {
  return Colors.conditions[condition as keyof typeof Colors.conditions] || Colors.conditions.generalDistress;
};

// Helper function to get mental health status color
export const getMentalHealthColor = (status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical'): string => {
  return Colors.mentalHealth[status] || Colors.mentalHealth.fair;
};

// Helper function to get intervention color
export const getInterventionColor = (type: string): string => {
  return Colors.interventions[type as keyof typeof Colors.interventions] || Colors.primary;
};

// Helper function to get privacy level color
export const getPrivacyColor = (level: 'maximum' | 'high' | 'moderate' | 'minimal'): string => {
  return Colors.privacy[level] || Colors.privacy.maximum;
};

export default Colors;