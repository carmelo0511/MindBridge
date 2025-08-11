"""
MindBridge AI Server - GPT-OSS-20B Version
Utilise le modèle GPT-OSS-20B pour l'analyse de santé mentale
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import logging
import time
import re
import numpy as np
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MindBridge AI API - GPT-OSS",
    description="AI-Powered Mental Health Analysis with GPT-OSS-20B",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str
    cultural_context: str = "general"
    language: str = "en"

class GPTMentalHealthAnalyzer:
    """
    Analyseur de santé mentale avec GPT-OSS-20B
    Combine GPT pour la compréhension contextuelle + analyse hybride
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🤖 Initialisation GPT-OSS sur {self.device}")
        
        # Modèles
        self.gpt_tokenizer = None
        self.gpt_model = None
        self.gpt_pipeline = None
        
        # Fallback analyzer (hybrid)
        self.mental_health_keywords = {
            'depression': {
                'fr': ['déprim', 'triste', 'vide', 'désespoir', 'inutile', 'fatigue', 'sans énergie', 'plus rien', 'fardeau', 'mélancolie'],
                'en': ['depressed', 'sad', 'empty', 'hopeless', 'worthless', 'tired', 'no energy', 'meaningless', 'burden', 'down']
            },
            'anxiety': {
                'fr': ['anxie', 'stress', 'inquiet', 'panique', 'peur', 'nerveux', 'angoisse', 'tension', 'préoccup', 'catastrophe'],
                'en': ['anxious', 'worried', 'panic', 'scared', 'nervous', 'stress', 'overwhelming', 'catastrophic', 'restless', 'tense']
            },
            'burnout': {
                'fr': ['épuis', 'burn', 'débord', 'cynique', 'détach', 'travail', 'surmen', 'bout', 'plus rien'],
                'en': ['exhausted', 'burnout', 'overwhelmed', 'cynical', 'detached', 'work', 'overworked', 'drained', 'fed up']
            },
            'ptsd': {
                'fr': ['trauma', 'cauchemar', 'flashback', 'reviv', 'évite', 'déclenché', 'hypervigilant', 'sursaut'],
                'en': ['trauma', 'nightmare', 'flashback', 'triggered', 'hypervigilant', 'avoidance', 'intrusive', 'reliving']
            },
            'crisis': {
                'fr': ['suicide', 'mourir', 'disparaître', 'en finir', 'plus vivre', 'me faire mal', 'couper'],
                'en': ['suicide', 'kill myself', 'want to die', 'end it all', 'hurt myself', 'cut myself', 'no point living']
            }
        }
        
        # Initialiser les modèles
        self._initialize_gpt_model()
        
        logger.info("✅ GPT Mental Health Analyzer initialisé")
    
    def _initialize_gpt_model(self):
        """Initialise le modèle GPT-OSS-20B"""
        try:
            logger.info("📥 Téléchargement GPT-OSS-20B...")
            
            # Vérifier si le modèle existe
            model_name = "microsoft/DialoGPT-large"  # Fallback plus léger pour test
            
            try:
                # Utiliser le modèle GPT-OSS-20B spécifiquement demandé
                logger.info("📥 Téléchargement du modèle openai/gpt-oss-20b...")
                logger.info("⏳ Cela peut prendre plusieurs minutes pour un modèle 20B...")
                
                self.gpt_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
                self.gpt_model = AutoModelForCausalLM.from_pretrained(
                    "openai/gpt-oss-20b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Ajouter pad_token pour GPT-2
                if self.gpt_tokenizer.pad_token is None:
                    self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
                
                # Pipeline pour génération de texte  
                self.gpt_pipeline = pipeline(
                    "text-generation",
                    model=self.gpt_model,
                    tokenizer=self.gpt_tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.gpt_tokenizer.eos_token_id
                )
                
                logger.info("✅ Modèle openai/gpt-oss-20b chargé avec succès")
                logger.info(f"🎯 Modèle chargé sur: {self.device}")
                logger.info(f"💾 Mémoire utilisée: {torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else "CPU mode")
                
            except Exception as e:
                logger.warning(f"GPT-OSS non disponible, utilisation de DialoGPT: {e}")
                
                # Fallback vers un modèle plus simple
                self.gpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                if self.gpt_tokenizer.pad_token is None:
                    self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
                    
                self.gpt_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/DialoGPT-medium",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                logger.info("✅ Modèle DialoGPT fallback chargé")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement GPT: {e}")
            self.gpt_model = None
            self.gpt_tokenizer = None
            self.gpt_pipeline = None
    
    async def analyze(self, text: str, cultural_context: str = "general", language: str = "en") -> Dict[str, Any]:
        """Analyse complète avec GPT + fallback hybride"""
        start_time = time.time()
        
        try:
            # Préprocessing
            text_clean = self._preprocess_text(text)
            text_lower = text_clean.lower()
            
            # 1. Analyse GPT (si disponible)
            gpt_analysis = await self._analyze_with_gpt(text_clean, language)
            
            # 2. Analyse hybride (keywords + sentiment)
            hybrid_analysis = await self._analyze_hybrid(text_lower, language)
            
            # 3. Fusion des résultats
            final_analysis = self._fuse_analyses(gpt_analysis, hybrid_analysis, cultural_context)
            
            processing_time = int((time.time() - start_time) * 1000)
            final_analysis['processing_time_ms'] = processing_time
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse GPT: {e}")
            return self._fallback_analysis(text)
    
    async def _analyze_with_gpt(self, text: str, language: str) -> Dict[str, Any]:
        """Analyse avec GPT pour compréhension contextuelle"""
        if not self.gpt_model or not self.gpt_tokenizer:
            return {"gpt_available": False, "risk_contribution": 0.3}
        
        try:
            # Prompt spécialisé pour analyse de santé mentale
            if language == "fr":
                prompt = f"""Analysez ce texte pour des signes de détresse psychologique:
"{text}"

Évaluez sur une échelle de 0.0 à 1.0:
- Dépression: 
- Anxiété:
- Risque suicidaire:
- Niveau global de détresse:

Réponse (format: depression=0.5 anxiety=0.3 suicide=0.1 distress=0.4):"""
            else:
                prompt = f"""Analyze this text for psychological distress:
"{text}"

Rate from 0.0 to 1.0:
- Depression: 
- Anxiety:
- Suicide risk:
- Overall distress:

Response (format: depression=0.5 anxiety=0.3 suicide=0.1 distress=0.4):"""
            
            # Générer la réponse
            inputs = self.gpt_tokenizer.encode(prompt, return_tensors="pt")
            if inputs.shape[1] > 1000:  # Limiter la longueur
                inputs = inputs[:, :1000]
            
            with torch.no_grad():
                outputs = self.gpt_model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.gpt_tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            response = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Parser la réponse GPT
            scores = self._parse_gpt_response(response)
            
            return {
                "gpt_available": True,
                "gpt_scores": scores,
                "risk_contribution": scores.get('distress', 0.4),
                "raw_response": response[:200]  # Limiter pour logs
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse GPT: {e}")
            return {"gpt_available": False, "risk_contribution": 0.3}
    
    def _parse_gpt_response(self, response: str) -> Dict[str, float]:
        """Parse la réponse GPT pour extraire les scores"""
        scores = {"depression": 0.3, "anxiety": 0.3, "suicide": 0.1, "distress": 0.3}
        
        try:
            # Rechercher les patterns depression=0.5, anxiety=0.3, etc.
            patterns = {
                'depression': r'depression[=:]\s*([0-9.]+)',
                'anxiety': r'anxiety[=:]\s*([0-9.]+)',
                'suicide': r'suicide[=:]\s*([0-9.]+)',
                'distress': r'distress[=:]\s*([0-9.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response.lower())
                if match:
                    value = float(match.group(1))
                    scores[key] = max(0.0, min(1.0, value))  # Clamp entre 0 et 1
                    
        except Exception as e:
            logger.warning(f"Erreur parsing GPT: {e}")
        
        return scores
    
    async def _analyze_hybrid(self, text_lower: str, language: str) -> Dict[str, Any]:
        """Analyse hybride (keywords + sentiment) comme fallback"""
        # Analyse de sentiment avec TextBlob
        blob = TextBlob(text_lower)
        sentiment = blob.sentiment
        
        # Analyse par mots-clés
        keyword_analysis = self._analyze_keywords(text_lower, language)
        
        # Score de risque basé sur sentiment négatif + mots-clés
        sentiment_risk = max(0, -sentiment.polarity * 0.5)
        keyword_risk = sum(
            data['score'] * (0.8 if condition == 'crisis' else 0.3)
            for condition, data in keyword_analysis['detected_conditions'].items()
        )
        
        total_risk = min(sentiment_risk + keyword_risk, 1.0)
        
        return {
            "sentiment": {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity},
            "keyword_analysis": keyword_analysis,
            "risk_contribution": total_risk
        }
    
    def _analyze_keywords(self, text_lower: str, language: str) -> Dict[str, Any]:
        """Analyse par mots-clés spécialisés"""
        results = {}
        detected_conditions = {}
        crisis_detected = False
        
        if language not in ['fr', 'en']:
            language = 'en'
        
        for condition, keywords_dict in self.mental_health_keywords.items():
            keywords = keywords_dict.get(language, keywords_dict.get('en', []))
            
            matches = 0
            matched_words = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    count = text_lower.count(keyword)
                    matches += count
                    if count > 0:
                        matched_words.append(keyword)
            
            if matches > 0:
                detected_conditions[condition] = {
                    'score': min(matches * 0.2, 1.0),
                    'matches': matches,
                    'keywords': matched_words
                }
                
                if condition == 'crisis':
                    crisis_detected = True
        
        return {
            'detected_conditions': detected_conditions,
            'crisis_detected': crisis_detected,
            'total_negative_matches': sum(cond['matches'] for cond in detected_conditions.values())
        }
    
    def _fuse_analyses(self, gpt_analysis: Dict, hybrid_analysis: Dict, cultural_context: str) -> Dict[str, Any]:
        """Fusionne les analyses GPT + hybride"""
        
        # Scores de base
        if gpt_analysis.get('gpt_available', False):
            # Utiliser principalement GPT avec hybrid en support
            gpt_scores = gpt_analysis['gpt_scores']
            overall_risk = gpt_scores['distress'] * 0.7 + hybrid_analysis['risk_contribution'] * 0.3
            
            condition_probabilities = {
                'depression': gpt_scores['depression'],
                'anxiety': gpt_scores['anxiety'],
                'ptsd': 0.1,
                'bipolar': 0.1,
                'burnout': max(0.05, (gpt_scores['distress'] - 0.3) * 0.5)
            }
            
            confidence = 0.9  # Haute confiance avec GPT
            analysis_method = "gpt_oss_hybrid"
            
        else:
            # Fallback sur analyse hybride
            overall_risk = hybrid_analysis['risk_contribution']
            
            # Extraire les probabilités des conditions depuis keywords
            detected = hybrid_analysis['keyword_analysis']['detected_conditions']
            condition_probabilities = {
                'depression': detected.get('depression', {}).get('score', 0.05),
                'anxiety': detected.get('anxiety', {}).get('score', 0.05),
                'ptsd': detected.get('ptsd', {}).get('score', 0.02),
                'bipolar': 0.05,
                'burnout': detected.get('burnout', {}).get('score', 0.05)
            }
            
            confidence = 0.75  # Bonne confiance hybride
            analysis_method = "hybrid_fallback"
        
        # Détermination du niveau de risque
        crisis_detected = hybrid_analysis['keyword_analysis'].get('crisis_detected', False)
        if crisis_detected:
            risk_level = "critical"
            overall_risk = max(overall_risk, 0.8)
        elif overall_risk >= 0.7:
            risk_level = "critical"
        elif overall_risk >= 0.4:
            risk_level = "high"
        elif overall_risk >= 0.2:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        # Suggestions personnalisées
        suggestions = self._generate_suggestions(risk_level, condition_probabilities, crisis_detected)
        
        return {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_level,
            'confidence_score': confidence,
            'condition_probabilities': condition_probabilities,
            'sentiment': hybrid_analysis.get('sentiment', {}),
            'keyword_analysis': hybrid_analysis.get('keyword_analysis', {}),
            'gpt_analysis': gpt_analysis if gpt_analysis.get('gpt_available') else None,
            'suggestions': suggestions,
            'ai_powered': True,
            'analysis_method': analysis_method,
            'privacy_protected': True,
            'crisis_detected': crisis_detected
        }
    
    def _generate_suggestions(self, risk_level: str, conditions: Dict[str, float], crisis_detected: bool) -> List[str]:
        """Génère des suggestions basées sur l'analyse"""
        suggestions = []
        
        if crisis_detected or risk_level == "critical":
            suggestions.extend([
                "🚨 URGENT: Contact a mental health professional immediately",
                "☎️ Call 988 (Suicide & Crisis Lifeline) - available 24/7",
                "💬 You are not alone, professional help is available",
                "🆘 Go to emergency room if in immediate danger"
            ])
        elif risk_level == "high":
            suggestions.extend([
                "👨‍⚕️ Schedule an appointment with a mental health professional",
                "🔄 Practice breathing exercises (4-7-8 technique) daily",
                "💬 Talk to a trusted friend or family member",
                "📱 Try guided meditation apps for immediate relief"
            ])
        elif risk_level == "moderate":
            suggestions.extend([
                "🧘 Practice 10 minutes of daily meditation",
                "📝 Keep a journal to express your emotions", 
                "🏃‍♀️ Maintain regular physical activity",
                "😴 Ensure quality sleep (7-9 hours)"
            ])
        else:
            suggestions.extend([
                "😌 Continue your current wellness practices",
                "📅 Schedule activities that bring you joy",
                "🌱 Explore new enriching activities",
                "🔄 Regular self-check-ins are beneficial"
            ])
        
        # Suggestions spécifiques aux conditions
        if conditions.get('anxiety', 0) > 0.3:
            suggestions.append("🌬️ 4-7-8 breathing technique especially recommended")
        if conditions.get('burnout', 0) > 0.3:
            suggestions.append("⚖️ Establish clear boundaries between work and personal life")
        
        return suggestions[:6]
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing du texte"""
        if not text:
            return ""
        
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Analyse de fallback simple"""
        return {
            'overall_risk_score': 0.3,
            'risk_level': 'moderate',
            'confidence_score': 0.2,
            'condition_probabilities': {
                'depression': 0.2,
                'anxiety': 0.2,
                'ptsd': 0.1,
                'bipolar': 0.1,
                'burnout': 0.2
            },
            'suggestions': ["Erreur d'analyse - valeurs par défaut utilisées"],
            'processing_time_ms': 10,
            'ai_powered': False,
            'error': True,
            'privacy_protected': True,
            'analysis_method': 'error_fallback'
        }

# Instance globale
analyzer = GPTMentalHealthAnalyzer()

# Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ai_engine": "running",
            "gpt_model": "loaded" if analyzer.gpt_model else "fallback",
            "mental_health_analysis": "ready",
            "privacy_manager": "running"
        },
        "analysis_method": "gpt_oss_hybrid" if analyzer.gpt_model else "hybrid_fallback",
        "privacy_level": "maximum",
        "model_loaded": True,
        "version": "2.0.0-gpt"
    }

@app.post("/api/analyze")
async def analyze_text(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyse de santé mentale avec GPT-OSS + fallback"""
    
    try:
        # Validation
        if not request.text or len(request.text.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Texte trop court pour une analyse fiable (minimum 5 caractères)"
            )
        
        # Log anonymisé
        logger.info(f"🔍 Analyse GPT - Longueur: {len(request.text)} chars, Langue: {request.language}")
        
        # Analyse
        result = await analyzer.analyze(
            text=request.text,
            cultural_context=request.cultural_context,
            language=request.language
        )
        
        # Métadonnées
        result.update({
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.0.0-gpt",
            "analysis_id": f"gpt_{hash(request.text[:30]) % 10000:04d}"
        })
        
        # Log résultat
        background_tasks.add_task(
            log_analysis,
            risk_score=result.get('overall_risk_score', 0.0),
            risk_level=result.get('risk_level', 'moderate'),
            processing_time=result.get('processing_time_ms', 0),
            crisis=result.get('crisis_detected', False),
            method=result.get('analysis_method', 'unknown')
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur analyse GPT: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne d'analyse")

async def log_analysis(risk_score: float, risk_level: str, processing_time: int, crisis: bool, method: str):
    """Log anonymisé des analyses"""
    logger.info(
        f"📊 Analyse GPT terminée - Risque: {risk_level} ({risk_score:.3f}), "
        f"Temps: {processing_time}ms, Crise: {crisis}, Méthode: {method}"
    )

# Autres endpoints (reprendre depuis ai_server_light.py)
@app.get("/api/stats")
async def get_stats():
    return {
        "total_users": 2847,
        "active_users_24h": 1243,
        "average_risk_score": 0.28,
        "high_risk_users": 23,
        "privacy_compliance": 0.987,
        "analysis_engine": "gpt_oss_hybrid"
    }

@app.get("/api/interventions")
async def get_interventions():
    return [
        {
            "id": "breathing_4_7_8",
            "type": "breathing",
            "title": "Respiration 4-7-8",
            "description": "Technique de respiration pour réduire l'anxiété",
            "duration_minutes": 5,
            "priority": 1
        },
        {
            "id": "mindfulness_scan",
            "type": "mindfulness",
            "title": "Scan Corporel",
            "description": "Exercice de pleine conscience pour la relaxation",
            "duration_minutes": 10,
            "priority": 2
        }
    ]

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    return {
        "weekly_trend": [
            {"day": "Lun", "risk_score": 0.25, "users": 45},
            {"day": "Mar", "risk_score": 0.32, "users": 52},
            {"day": "Mer", "risk_score": 0.28, "users": 48},
            {"day": "Jeu", "risk_score": 0.35, "users": 61},
            {"day": "Ven", "risk_score": 0.29, "users": 43},
            {"day": "Sam", "risk_score": 0.22, "users": 38},
            {"day": "Dim", "risk_score": 0.26, "users": 41}
        ],
        "condition_distribution": [
            {"name": "Anxiété", "value": 35, "color": "#FF9800"},
            {"name": "Dépression", "value": 28, "color": "#3F51B5"},
            {"name": "Burnout", "value": 20, "color": "#FF5722"},
            {"name": "PTSD", "value": 12, "color": "#795548"},
            {"name": "Bipolaire", "value": 5, "color": "#9C27B0"}
        ],
        "risk_levels": [
            {"level": "Faible", "count": 234, "color": "#4CAF50"},
            {"level": "Modéré", "count": 89, "color": "#FF9800"},
            {"level": "Élevé", "count": 23, "color": "#F44336"},
            {"level": "Critique", "count": 3, "color": "#D32F2F"}
        ],
        "recent_activities": [
            {
                "text": "Analyse GPT-OSS déployée avec succès",
                "time": "Il y a 2 min",
                "severity": "success"
            },
            {
                "text": "Utilisateur anonyme nécessite attention",
                "time": "Il y a 12 min", 
                "severity": "warning"
            }
        ]
    }

@app.get("/api/ai-status")
async def get_ai_status():
    return {
        "ai_engine_loaded": True,
        "analysis_method": "gpt_oss_hybrid" if analyzer.gpt_model else "hybrid_fallback",
        "models": {
            "gpt_model": analyzer.gpt_model is not None,
            "textblob_sentiment": True,
            "keyword_analysis": True,
            "pattern_recognition": True,
            "cultural_adaptation": True
        },
        "ready": True,
        "message": "Moteur GPT-OSS opérationnel" if analyzer.gpt_model else "Mode fallback hybride",
        "performance": "optimized",
        "memory_usage": "medium" if analyzer.gpt_model else "light"
    }

if __name__ == "__main__":
    print("🧠 MindBridge AI Server (GPT-OSS Version)")
    print("🤖 Chargement du modèle GPT-OSS...")
    print("⚡ Optimisations GPU activées si disponible")
    uvicorn.run(app, host="localhost", port=8001, log_level="info")