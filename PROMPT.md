# 🧠 MindBridge - Prompt Complet pour Claude Code

## Contexte du Projet
Je participe à un hackathon OpenAI dans la catégorie "For Humanity". Je veux créer MindBridge, un système d'IA décentralisé pour la détection précoce et la prévention en santé mentale utilisant les modèles GPT-OSS. Le système doit être 100% privé (traitement local), capable de détecter les signes précoces de détresse mentale, et accessible gratuitement.

## Objectif Principal
Créer une application complète qui:
- Fonctionne entièrement en local avec GPT-OSS
- Analyse les patterns de communication pour détecter la détresse mentale
- Préserve totalement la vie privée (zero-knowledge architecture)
- Fournit des interventions personnalisées et culturellement adaptées
- Connecte les utilisateurs à des ressources de support appropriées

## Architecture Technique Requise

### 1. Structure du Projet
```
mindbridge/
├── backend/
│   ├── core/
│   │   ├── mental_health_engine.py
│   │   ├── privacy_manager.py
│   │   ├── pattern_detector.py
│   │   └── intervention_system.py
│   ├── models/
│   │   ├── gpt_oss_wrapper.py
│   │   ├── fine_tuning/
│   │   └── federated_learning/
│   ├── api/
│   │   ├── local_server.py
│   │   └── p2p_network.py
│   └── database/
│       └── local_encrypted_db.py
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   ├── screens/
│   │   └── utils/
│   └── web/
│       ├── dashboard/
│       └── landing/
├── mobile/
│   ├── ios/
│   └── android/
├── ml/
│   ├── training/
│   ├── datasets/
│   └── evaluation/
└── deployment/
    ├── docker/
    └── kubernetes/
```

### 2. Frontend Requirements - Application Individuelle
**Screens nécessaires:**
- **OnboardingScreen**: consent éclairé, configuration privacy
- **DashboardScreen**: vue d'ensemble santé mentale personnelle
- **CheckInScreen**: questionnaires adaptatifs quotidiens
- **JournalScreen**: entrées texte/voix avec analyse temps réel
- **InsightsScreen**: visualisation patterns et tendances personnelles
- **InterventionScreen**: exercices et ressources personnalisés
- **CommunityScreen**: connexion anonyme avec pairs
- **CrisisScreen**: ressources urgence et hotlines
- **SettingsScreen**: privacy controls, data export/delete

**Dashboard Web**: Interface professionnels de santé (avec consentement seulement)

### 3. Fonctionnalités Core à Implémenter

#### A. Module de Détection (mental_health_engine.py)
Créer un système qui:
- Analyse les patterns linguistiques (sentiment, vocabulaire, structure)
- Détecte les changements comportementaux (fréquence, timing, engagement)
- Identifie les marqueurs de détresse spécifiques:
  - Dépression: isolation, négativité, troubles du sommeil
  - Anxiété: catastrophisation, évitement, symptômes physiques
  - Burnout: cynisme, épuisement, détachement
  - PTSD: hypervigilance, flashbacks linguistiques, évitement
- Calcule un score de risque multi-factoriel (0-1)
- Génère des rapports anonymisés pour apprentissage fédéré

#### B. Système de Privacy (privacy_manager.py)
Implémenter:
- Chiffrement AES-256 pour toutes les données locales
- Architecture zero-knowledge proof
- Differential privacy pour les métriques agrégées
- Système de clés dérivées pour multi-device sync
- Auto-destruction des données après X jours
- Mode "panic button" pour effacement immédiat

#### C. Interventions Personnalisées (intervention_system.py)
Développer:
- Système de recommandations adaptatif basé sur:
  - Niveau de risque détecté
  - Contexte culturel de l'utilisateur
  - Historique des interventions efficaces
  - Préférences personnelles
- Bibliothèque d'interventions:
  - Exercices de respiration guidés
  - Techniques CBT (Cognitive Behavioral Therapy)
  - Mindfulness et méditation
  - Journaling prompts
  - Activités physiques adaptées
- Escalade intelligente vers professionnels
- Système de buddy anonyme pour support peer-to-peer

#### D. Fine-tuning GPT-OSS (gpt_oss_wrapper.py)
Configurer:
- Modèle de base GPT-OSS pour santé mentale
- Adaptation aux dialectes et expressions culturelles
- Détection de signaux subtils dans 50+ langues
- Optimisation pour edge devices (quantization, pruning)
- Cache intelligent pour réponses offline
- Mécanisme de mise à jour incrémentale

### 4. Algorithmes Spécifiques

#### A. Pattern Detection Algorithm
```python
def detect_mental_health_patterns(user_data):
    """
    Implémente:
    1. NLP analysis: sentiment, entities, syntax complexity
    2. Temporal analysis: circadian disruption, response delays
    3. Social analysis: isolation indicators, relationship changes
    4. Behavioral analysis: app usage, activity levels
    5. Multimodal fusion: combine all signals with weights
    """
```

#### B. Risk Scoring System
```python
def calculate_risk_score(patterns):
    """
    Utilise:
    - Bayesian inference pour probabilités
    - LSTM pour séquences temporelles
    - Ensemble methods pour robustesse
    - Calibration avec données cliniques
    """
```

#### C. Intervention Matching
```python
def match_intervention(user_profile, risk_level):
    """
    Algorithme de recommandation:
    - Collaborative filtering basé sur succès passés
    - Contextual bandits pour exploration/exploitation
    - Reinforcement learning pour optimisation continue
    """
```

### 5. Intégrations Externes
**APIs et Services**
- Intégration WhatsApp/Telegram pour check-ins
- Apple HealthKit/Google Fit pour données biométriques
- Crisis Text Line API pour urgences
- OpenStreetMap pour ressources locales
- Calendrier pour rappels interventions

**Datasets pour Fine-tuning**
- Reddit Mental Health forums (anonymisés)
- Therapy transcripts (avec consent)
- Clinical assessment tools (PHQ-9, GAD-7)
- Multilingual emotion lexicons
- Cultural expression databases

### 6. Tests et Validation
**Test Suite Complète**
```python
# Unit tests pour chaque module
# Integration tests pour workflows
# Performance tests (latence < 100ms)
# Security tests (penetration, fuzzing)
# Clinical validation tests
# A/B testing framework pour interventions
```

**Métriques de Succès**
- Precision/Recall pour détection
- Time-to-intervention
- User engagement rates
- Clinical outcomes (PHQ-9 reduction)
- Privacy preservation metrics
- System performance (CPU, RAM, battery)

### 7. Instructions Spécifiques pour Claude Code

1. **Privacy first** - Chaque fonction doit respecter le principe de zero-knowledge
2. **Optimisation mobile** - Le code doit tourner sur smartphone mid-range
3. **Offline-first** - Toutes les fonctionnalités core sans internet
4. **Culturellement sensible** - Éviter assumptions occidentales
5. **Accessibilité** - Support screen readers, contraste élevé
6. **Open source ready** - Code propre, bien structuré

### 8. Exemple de Code Core

```python
# mental_health_engine.py
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from cryptography.fernet import Fernet
import torch
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta

class MindBridgeEngine:
    def __init__(self, model_path: str, privacy_mode: str = "maximum"):
        """
        Initialize MindBridge Mental Health Detection Engine
        
        Args:
            model_path: Path to fine-tuned GPT-OSS model
            privacy_mode: Level of privacy protection
        """
        self.model = self._load_local_model(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.privacy_manager = PrivacyManager(privacy_mode)
        self.pattern_detector = PatternDetector()
        self.risk_calculator = RiskCalculator()
        
    def analyze_user_state(self, 
                          text_data: List[str],
                          behavioral_data: Dict,
                          cultural_context: Dict) -> Dict:
        """
        Main analysis pipeline for mental health assessment
        
        Privacy: All processing happens locally, no external API calls
        Performance: ~100ms on mobile device
        
        Returns:
            Dictionary with risk score, recommendations, and resources
        """
        # Encrypt all data immediately
        encrypted_data = self.privacy_manager.encrypt_batch({
            'text': text_data,
            'behavior': behavioral_data,
            'context': cultural_context
        })
        
        # Extract features using local GPT-OSS
        features = self._extract_features(encrypted_data)
        
        # Detect patterns indicative of mental health issues
        patterns = self.pattern_detector.detect(features)
        
        # Calculate multi-factorial risk score
        risk_score = self.risk_calculator.calculate(patterns)
        
        # Generate personalized interventions
        interventions = self._generate_interventions(
            risk_score, 
            cultural_context
        )
        
        # Find appropriate support resources
        resources = self._match_resources(risk_score, cultural_context)
        
        # Return anonymized results
        return self.privacy_manager.anonymize_output({
            'risk_score': risk_score,
            'confidence': patterns['confidence'],
            'primary_concerns': patterns['concerns'],
            'interventions': interventions,
            'resources': resources,
            'next_check_in': self._schedule_next_checkin(risk_score)
        })
```

### 9. Critères de Succès Hackathon
Le projet doit démontrer:
- **Innovation**: Première solution 100% privée et décentralisée
- **Impact**: Potentiel pour sauver des vies à l'échelle mondiale
- **Faisabilité**: Prototype fonctionnel en 48h
- **Scalabilité**: Architecture supportant millions d'utilisateurs
- **Éthique**: Respect total de la privacy et de l'autonomie

### 10. Focus: Application Personnelle pour Individus
L'app est conçue pour l'auto-monitoring et le self-care, pas pour le diagnostic professionnel. L'utilisateur garde le contrôle total de ses données et de son parcours de bien-être mental.