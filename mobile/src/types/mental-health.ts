/**
 * Mental Health Type Definitions
 */

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

export type MentalHealthCondition = 'depression' | 'anxiety' | 'ptsd' | 'bipolar' | 'burnout';

export interface MentalHealthStatus {
  riskLevel: RiskLevel;
  overallScore: number;
  conditions: Record<MentalHealthCondition, number>;
  lastUpdated: Date;
  confidence: number;
  userProfile?: UserProfile;
}

export interface UserProfile {
  id: string;
  preferredName: string;
  culturalBackground: string;
  language: string;
  privacyLevel: 'minimal' | 'moderate' | 'high' | 'maximum';
  baselineMetrics: any;
}

export interface Intervention {
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