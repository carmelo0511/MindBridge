/**
 * Navigation Type Definitions
 */

export type RootStackParamList = {
  Onboarding: undefined;
  Main: undefined;
  Intervention: { intervention: any };
  Settings: undefined;
  Crisis: undefined;
};

export type TabParamList = {
  Dashboard: undefined;
  CheckIn: { quickMode?: boolean };
  Journal: undefined;
  Insights: { insightId?: string };
  Community: undefined;
};