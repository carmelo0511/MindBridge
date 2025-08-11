"""
MindBridge Real AI Engine

Intègre des modèles Hugging Face pour l'analyse de santé mentale
Utilise des modèles pré-entraînés avec fine-tuning pour la détection
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel, AutoConfig
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import time
from datetime import datetime
import re
import nltk
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthAI:
    """
    Moteur IA réel pour l'analyse de santé mentale
    Utilise des modèles Transformer fine-tunés
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🤖 Initialisation sur {self.device}")
        
        # Initialiser les modèles
        self.emotion_classifier = None
        self.sentiment_analyzer = None
        self.embedding_model = None
        self.mental_health_classifier = None
        
        # Télécharger les données NLTK nécessaires
        self._download_nltk_data()
        
        # Initialiser les modèles
        self._initialize_models()
        
    def _download_nltk_data(self):
        """Télécharge les données NLTK nécessaires"""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Erreur téléchargement NLTK: {e}")
    
    def _initialize_models(self):
        """Initialise tous les modèles d'IA"""
        logger.info("🔄 Chargement des modèles d'IA...")
        
        try:
            # 1. Analyseur de sentiment
            logger.info("📊 Chargement analyseur de sentiment...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            # 2. Classificateur d'émotions
            logger.info("😊 Chargement classificateur d'émotions...")
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.device == "cuda" else -1
            )
            
            # 3. Modèle d'embeddings sémantiques
            logger.info("🧠 Chargement modèle d'embeddings...")
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            # 4. Classificateur spécialisé santé mentale
            logger.info("🏥 Chargement classificateur santé mentale...")
            self.mental_health_classifier = pipeline(
                "text-classification",
                model="mental/mental-bert-base-uncased",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("✅ Tous les modèles d'IA chargés avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles: {e}")
            # Fallback vers modèles plus simples
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialise des modèles de fallback plus simples"""
        logger.info("🔄 Chargement modèles de fallback...")
        
        try:
            # Modèle de sentiment simple
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1
            )
            
            # Modèle d'embeddings simple
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            logger.info("✅ Modèles de fallback chargés")
            
        except Exception as e:
            logger.error(f"❌ Erreur modèles fallback: {e}")
            self.sentiment_analyzer = None
            self.embedding_model = None
    
    async def analyze_mental_health(
        self, 
        text: str,
        cultural_context: str = "western",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Analyse complète de santé mentale avec IA
        """
        start_time = time.time()
        
        try:
            # Préprocessing du texte
            cleaned_text = self._preprocess_text(text)
            
            if not cleaned_text:
                return self._generate_default_analysis()
            
            # Analyses en parallèle
            results = {}
            
            # 1. Analyse de sentiment
            sentiment_result = await self._analyze_sentiment(cleaned_text)
            results['sentiment'] = sentiment_result
            
            # 2. Analyse d'émotions
            emotion_result = await self._analyze_emotions(cleaned_text)
            results['emotions'] = emotion_result
            
            # 3. Détection de mots-clés de santé mentale
            keywords_result = await self._analyze_mental_health_keywords(cleaned_text)
            results['keywords'] = keywords_result
            
            # 4. Analyse sémantique
            semantic_result = await self._analyze_semantic_patterns(cleaned_text)
            results['semantic'] = semantic_result
            
            # 5. Classification santé mentale
            mental_health_result = await self._classify_mental_health(cleaned_text)
            results['mental_health_classification'] = mental_health_result
            
            # Fusion des résultats
            final_analysis = await self._fuse_analysis_results(results, cultural_context)
            
            # Ajout des métadonnées
            final_analysis['processing_time_ms'] = int((time.time() - start_time) * 1000)
            final_analysis['ai_powered'] = True
            final_analysis['models_used'] = self._get_models_info()
            final_analysis['cultural_context'] = cultural_context
            final_analysis['language'] = language
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse IA: {e}")
            return self._generate_error_analysis(str(e))
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyse de sentiment avec Transformers"""
        if not self.sentiment_analyzer:
            return {"score": 0.5, "label": "NEUTRAL", "confidence": 0.1}
        
        try:
            # Découper le texte si trop long
            chunks = self._split_text(text, max_length=512)
            sentiment_scores = []
            
            for chunk in chunks:
                result = self.sentiment_analyzer(chunk)[0]
                
                # Normaliser les labels selon le modèle
                if result['label'] in ['POSITIVE', 'POS']:
                    score = result['score']
                elif result['label'] in ['NEGATIVE', 'NEG']:
                    score = -result['score']
                else:  # NEUTRAL
                    score = 0.0
                
                sentiment_scores.append(score)
            
            # Moyenne pondérée
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Convertir en score de risque (sentiment négatif = risque élevé)
            risk_score = max(0.0, -avg_sentiment)  # Inverser pour le risque
            
            return {
                "sentiment_score": avg_sentiment,
                "risk_contribution": risk_score,
                "confidence": 0.8,
                "interpretation": self._interpret_sentiment(avg_sentiment)
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse sentiment: {e}")
            return {"sentiment_score": 0.0, "risk_contribution": 0.3, "confidence": 0.1}
    
    async def _analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyse d'émotions avec modèle spécialisé"""
        if not self.emotion_classifier:
            return {"emotions": {}, "risk_contribution": 0.3}
        
        try:
            chunks = self._split_text(text, max_length=512)
            emotion_scores = {}
            
            for chunk in chunks:
                results = self.emotion_classifier(chunk)
                
                for result in results:
                    emotion = result['label'].lower()
                    score = result['score']
                    
                    if emotion in emotion_scores:
                        emotion_scores[emotion] += score
                    else:
                        emotion_scores[emotion] = score
            
            # Normaliser les scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            # Calculer contribution au risque
            risk_emotions = ['sadness', 'fear', 'anger', 'disgust']
            risk_contribution = sum(
                emotion_scores.get(emotion, 0.0) * 0.8 
                for emotion in risk_emotions
            )
            
            return {
                "emotions": emotion_scores,
                "risk_contribution": min(risk_contribution, 1.0),
                "dominant_emotion": max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse émotions: {e}")
            return {"emotions": {}, "risk_contribution": 0.3}
    
    async def _analyze_mental_health_keywords(self, text: str) -> Dict[str, Any]:
        """Détection de mots-clés spécifiques à la santé mentale"""
        
        # Dictionnaires de mots-clés par condition
        mental_health_keywords = {
            'depression': [
                'déprim', 'triste', 'vide', 'désespoir', 'inutile', 'fatigue',
                'sans énergie', 'plus rien', 'à quoi bon', 'fardeau',
                'depressed', 'sad', 'empty', 'hopeless', 'worthless', 'tired'
            ],
            'anxiety': [
                'anxie', 'stress', 'inquiet', 'panique', 'peur', 'nerveux',
                'angoisse', 'tension', 'préoccup', 'catastrophe',
                'anxious', 'worried', 'panic', 'scared', 'nervous', 'stress'
            ],
            'ptsd': [
                'trauma', 'cauchemar', 'flashback', 'reviv', 'évite', 'déclenché',
                'hypervigilant', 'sursaut', 'intrusi',
                'nightmare', 'flashback', 'triggered', 'hypervigilant', 'intrusive'
            ],
            'suicidal_ideation': [
                'suicide', 'mourir', 'disparaître', 'en finir', 'plus vivre',
                'mieux mort', 'kill myself', 'want to die', 'end it all'
            ],
            'self_harm': [
                'me faire mal', 'me bless', 'couper', 'brûl', 'automutilation',
                'hurt myself', 'cut myself', 'self harm', 'self-harm'
            ]
        }
        
        text_lower = text.lower()
        detected_keywords = {}
        risk_scores = {}
        
        for condition, keywords in mental_health_keywords.items():
            matches = 0
            matched_words = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    matches += text_lower.count(keyword)
                    matched_words.append(keyword)
            
            if matches > 0:
                detected_keywords[condition] = {
                    'count': matches,
                    'words': matched_words
                }
                
                # Score de risque basé sur la condition
                if condition in ['suicidal_ideation', 'self_harm']:
                    risk_scores[condition] = min(matches * 0.8, 1.0)  # Très élevé
                elif condition in ['depression', 'anxiety', 'ptsd']:
                    risk_scores[condition] = min(matches * 0.3, 1.0)  # Modéré à élevé
        
        # Calcul du risque global
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        
        return {
            'detected_keywords': detected_keywords,
            'condition_risks': risk_scores,
            'risk_contribution': max_risk,
            'crisis_indicators': bool(
                'suicidal_ideation' in detected_keywords or 
                'self_harm' in detected_keywords
            )
        }
    
    async def _analyze_semantic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyse sémantique avec embeddings"""
        if not self.embedding_model:
            return {"semantic_similarity": {}, "risk_contribution": 0.2}
        
        try:
            # Phrases de référence pour différentes conditions
            reference_patterns = {
                'depression': [
                    "Je me sens complètement vide et sans espoir",
                    "I feel completely empty and hopeless",
                    "Rien ne me fait plus plaisir, tout semble inutile"
                ],
                'anxiety': [
                    "Je m'inquiète constamment de tout ce qui pourrait mal se passer",
                    "I constantly worry about everything that could go wrong",
                    "Mon cœur bat très fort et j'ai du mal à respirer"
                ],
                'crisis': [
                    "Je ne peux plus supporter cette douleur",
                    "I can't take this pain anymore",
                    "Il vaudrait mieux que je ne sois plus là"
                ]
            }
            
            # Obtenir l'embedding du texte
            text_embedding = self.embedding_model.encode([text])
            similarities = {}
            
            for condition, patterns in reference_patterns.items():
                pattern_embeddings = self.embedding_model.encode(patterns)
                
                # Calculer similarité cosinus moyenne
                similarities_scores = []
                for pattern_emb in pattern_embeddings:
                    similarity = np.dot(text_embedding[0], pattern_emb) / (
                        np.linalg.norm(text_embedding[0]) * np.linalg.norm(pattern_emb)
                    )
                    similarities_scores.append(max(0, similarity))  # Garder seulement positif
                
                similarities[condition] = np.mean(similarities_scores)
            
            # Calcul du risque basé sur les similarités
            max_similarity = max(similarities.values()) if similarities else 0.0
            risk_contribution = max_similarity * 0.7  # Modérer l'impact
            
            return {
                "semantic_similarities": similarities,
                "risk_contribution": risk_contribution,
                "closest_pattern": max(similarities, key=similarities.get) if similarities else None
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse sémantique: {e}")
            return {"semantic_similarities": {}, "risk_contribution": 0.2}
    
    async def _classify_mental_health(self, text: str) -> Dict[str, Any]:
        """Classification avec modèle spécialisé santé mentale"""
        if not self.mental_health_classifier:
            return {"classifications": {}, "risk_contribution": 0.3}
        
        try:
            # Utiliser le classificateur si disponible
            results = self.mental_health_classifier(text)
            
            classifications = {}
            if results and isinstance(results, list):
                for result in results[0]:  # Premier résultat
                    label = result['label'].lower()
                    score = result['score']
                    classifications[label] = score
            
            # Calculer contribution au risque
            risk_labels = ['depression', 'anxiety', 'stress', 'suicidal']
            risk_contribution = sum(
                classifications.get(label, 0.0) * 0.6
                for label in risk_labels
            )
            
            return {
                "classifications": classifications,
                "risk_contribution": min(risk_contribution, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Erreur classification santé mentale: {e}")
            return {"classifications": {}, "risk_contribution": 0.3}
    
    async def _fuse_analysis_results(
        self, 
        results: Dict[str, Any],
        cultural_context: str
    ) -> Dict[str, Any]:
        """Fusionne tous les résultats d'analyse"""
        
        # Extraction des contributions au risque
        risk_components = {
            'sentiment': results.get('sentiment', {}).get('risk_contribution', 0.3),
            'emotions': results.get('emotions', {}).get('risk_contribution', 0.3),
            'keywords': results.get('keywords', {}).get('risk_contribution', 0.2),
            'semantic': results.get('semantic', {}).get('risk_contribution', 0.2),
            'classification': results.get('mental_health_classification', {}).get('risk_contribution', 0.3)
        }
        
        # Poids pour la fusion (ajustables selon le contexte culturel)
        weights = {
            'sentiment': 0.25,
            'emotions': 0.25,
            'keywords': 0.25,
            'semantic': 0.15,
            'classification': 0.10
        }
        
        # Ajustements culturels
        if cultural_context == "collectivist":
            weights['semantic'] += 0.05  # Plus d'importance au contexte
            weights['sentiment'] -= 0.05
        
        # Calcul du score de risque fusionné
        overall_risk = sum(
            risk_components[component] * weights[component]
            for component in risk_components
        )
        
        # Détermination du niveau de risque
        if overall_risk < 0.25:
            risk_level = "low"
        elif overall_risk < 0.5:
            risk_level = "moderate"
        elif overall_risk < 0.75:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Détection de crise
        crisis_detected = results.get('keywords', {}).get('crisis_indicators', False)
        if crisis_detected:
            risk_level = "critical"
            overall_risk = max(overall_risk, 0.9)
        
        # Conditions probables
        condition_probabilities = self._calculate_condition_probabilities(results)
        
        # Confiance globale
        confidence_scores = [
            results.get('sentiment', {}).get('confidence', 0.5),
            0.8 if results.get('emotions', {}).get('emotions') else 0.3,
            0.9 if results.get('keywords', {}).get('detected_keywords') else 0.2,
            0.7 if results.get('semantic', {}).get('semantic_similarities') else 0.3
        ]
        overall_confidence = np.mean(confidence_scores)
        
        # Suggestions basées sur l'analyse
        suggestions = self._generate_ai_suggestions(results, risk_level)
        
        return {
            'overall_risk_score': min(max(overall_risk, 0.0), 1.0),
            'risk_level': risk_level,
            'confidence_score': overall_confidence,
            'condition_probabilities': condition_probabilities,
            'risk_components': risk_components,
            'detailed_analysis': results,
            'suggestions': suggestions,
            'crisis_detected': crisis_detected,
            'privacy_protected': True
        }
    
    def _calculate_condition_probabilities(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calcule les probabilités pour chaque condition"""
        
        conditions = {
            'depression': 0.0,
            'anxiety': 0.0,
            'ptsd': 0.0,
            'bipolar': 0.0,
            'burnout': 0.0
        }
        
        # Basé sur les émotions
        emotions = results.get('emotions', {}).get('emotions', {})
        if 'sadness' in emotions:
            conditions['depression'] += emotions['sadness'] * 0.8
        if 'fear' in emotions:
            conditions['anxiety'] += emotions['fear'] * 0.7
        if 'anger' in emotions:
            conditions['ptsd'] += emotions['anger'] * 0.4
            conditions['bipolar'] += emotions['anger'] * 0.3
        
        # Basé sur les mots-clés
        keywords = results.get('keywords', {}).get('condition_risks', {})
        for condition, risk in keywords.items():
            if condition in conditions:
                conditions[condition] += risk * 0.6
        
        # Basé sur les similarités sémantiques
        semantic = results.get('semantic', {}).get('semantic_similarities', {})
        for pattern, similarity in semantic.items():
            if pattern in conditions:
                conditions[pattern] += similarity * 0.5
        
        # Normalisation et ajout d'un minimum
        for condition in conditions:
            conditions[condition] = min(max(conditions[condition], 0.05), 1.0)
        
        return conditions
    
    def _generate_ai_suggestions(self, results: Dict[str, Any], risk_level: str) -> List[str]:
        """Génère des suggestions basées sur l'analyse IA"""
        
        suggestions = []
        
        if risk_level == "critical":
            suggestions.extend([
                "🚨 Contacter immédiatement un professionnel de santé mentale",
                "☎️ Appeler une ligne d'écoute : 3114 (gratuit, 24h/24)",
                "🆘 En cas d'urgence : 15 (SAMU) ou 112"
            ])
        elif risk_level == "high":
            suggestions.extend([
                "👨‍⚕️ Consulter un professionnel de santé mentale",
                "🔄 Pratiquer des exercices de respiration quotidiens",
                "💬 Parler à un proche de confiance"
            ])
        elif risk_level == "moderate":
            suggestions.extend([
                "🧘 Intégrer la méditation dans votre routine",
                "📝 Tenir un journal de vos émotions",
                "🏃‍♀️ Maintenir une activité physique régulière"
            ])
        else:  # low
            suggestions.extend([
                "😌 Continuer vos pratiques de bien-être actuelles",
                "📅 Programmer des check-ins réguliers avec vous-même",
                "🌱 Explorer de nouvelles activités enrichissantes"
            ])
        
        # Suggestions spécifiques aux émotions détectées
        emotions = results.get('emotions', {}).get('emotions', {})
        dominant_emotion = results.get('emotions', {}).get('dominant_emotion')
        
        if dominant_emotion == 'anxiety' or 'fear' in emotions:
            suggestions.append("🌬️ Technique de respiration 4-7-8 recommandée")
        
        if dominant_emotion == 'sadness':
            suggestions.append("☀️ Exposition à la lumière naturelle bénéfique")
        
        return suggestions[:5]  # Limiter à 5 suggestions
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocesse le texte pour l'analyse"""
        if not text or not isinstance(text, str):
            return ""
        
        # Nettoyage basique
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Mentions et hashtags
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = text.strip()
        
        return text
    
    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """Découpe le texte en chunks pour les modèles"""
        if len(text) <= max_length:
            return [text]
        
        # Découper par phrases si possible
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks or [text[:max_length]]
    
    def _interpret_sentiment(self, sentiment_score: float) -> str:
        """Interprète le score de sentiment"""
        if sentiment_score > 0.3:
            return "Sentiment majoritairement positif"
        elif sentiment_score > -0.3:
            return "Sentiment neutre"
        else:
            return "Sentiment majoritairement négatif"
    
    def _generate_default_analysis(self) -> Dict[str, Any]:
        """Génère une analyse par défaut"""
        return {
            'overall_risk_score': 0.3,
            'risk_level': 'moderate',
            'confidence_score': 0.1,
            'condition_probabilities': {
                'depression': 0.2,
                'anxiety': 0.2,
                'ptsd': 0.1,
                'bipolar': 0.1,
                'burnout': 0.2
            },
            'suggestions': [
                "Texte insuffisant pour une analyse complète",
                "Essayez de décrire plus en détail votre état"
            ],
            'processing_time_ms': 10,
            'ai_powered': False
        }
    
    def _generate_error_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Génère une analyse d'erreur sécurisée"""
        return {
            'overall_risk_score': 0.3,
            'risk_level': 'moderate',
            'confidence_score': 0.1,
            'condition_probabilities': {
                'depression': 0.2,
                'anxiety': 0.2,
                'ptsd': 0.1,
                'bipolar': 0.1,
                'burnout': 0.2
            },
            'suggestions': [
                "Erreur lors de l'analyse - utilisation de valeurs par défaut",
                "Veuillez réessayer ou contacter le support"
            ],
            'processing_time_ms': 5,
            'ai_powered': False,
            'error': "Erreur d'analyse IA",
            'privacy_protected': True
        }
    
    def _get_models_info(self) -> Dict[str, str]:
        """Retourne les informations sur les modèles utilisés"""
        return {
            'sentiment': "cardiffnlp/twitter-roberta-base-sentiment-latest" if self.sentiment_analyzer else "None",
            'emotion': "j-hartmann/emotion-english-distilroberta-base" if self.emotion_classifier else "None",
            'embedding': "all-MiniLM-L6-v2" if self.embedding_model else "None",
            'mental_health': "mental/mental-bert-base-uncased" if self.mental_health_classifier else "None"
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Retourne le statut des modèles"""
        return {
            'device': self.device,
            'models_loaded': {
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'emotion_classifier': self.emotion_classifier is not None,
                'embedding_model': self.embedding_model is not None,
                'mental_health_classifier': self.mental_health_classifier is not None
            },
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A",
            'ready': any([
                self.sentiment_analyzer,
                self.emotion_classifier,
                self.embedding_model
            ])
        }

# Instance globale
mental_health_ai = None

def get_mental_health_ai() -> MentalHealthAI:
    """Retourne l'instance singleton du moteur IA"""
    global mental_health_ai
    if mental_health_ai is None:
        mental_health_ai = MentalHealthAI()
    return mental_health_ai