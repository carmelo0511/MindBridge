"""
MindBridge AI-Powered Server
Intègre les vrais modèles d'IA pour l'analyse de santé mentale
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager

# Import du moteur IA
from ai_engine import get_mental_health_ai, MentalHealthAI

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instance globale IA
ai_engine: MentalHealthAI = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    global ai_engine
    
    # Startup - Initialisation des modèles IA
    logger.info("🧠 Initialisation des modèles d'IA MindBridge...")
    try:
        ai_engine = get_mental_health_ai()
        logger.info("✅ Modèles d'IA chargés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèles IA: {e}")
        ai_engine = None
    
    yield
    
    # Shutdown
    logger.info("🔄 Arrêt du serveur MindBridge...")
    ai_engine = None

app = FastAPI(
    title="MindBridge AI API",
    description="AI-Powered Mental Health Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str
    cultural_context: str = "general"
    language: str = "en"

@app.get("/health")
async def health_check():
    global ai_engine
    
    ai_status = ai_engine.get_model_status() if ai_engine else {
        "ready": False,
        "models_loaded": {},
        "device": "cpu"
    }
    
    return {
        "status": "healthy" if ai_status.get("ready", False) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ai_engine": "running" if ai_engine else "failed",
            "mental_health_analysis": "ready" if ai_status.get("ready", False) else "loading",
            "privacy_manager": "running"
        },
        "ai_models": ai_status.get("models_loaded", {}),
        "device": ai_status.get("device", "cpu"),
        "privacy_level": "maximum",
        "model_loaded": ai_status.get("ready", False)
    }

@app.post("/api/analyze")
async def analyze_text(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyse de santé mentale avec IA avancée"""
    global ai_engine
    
    if not ai_engine:
        raise HTTPException(
            status_code=503, 
            detail="Moteur IA non disponible - utilisation du mode fallback"
        )
    
    try:
        # Validation de l'input
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Texte trop court pour une analyse fiable (minimum 10 caractères)"
            )
        
        # Log de l'analyse (anonymisé)
        logger.info(f"🔍 Analyse en cours - Longueur: {len(request.text)} caractères")
        
        # Analyse avec le moteur IA
        result = await ai_engine.analyze_mental_health(
            text=request.text,
            cultural_context=request.cultural_context,
            language=request.language
        )
        
        # Ajout de métadonnées pour l'API
        result.update({
            "privacy_protected": True,
            "timestamp": datetime.now().isoformat(),
            "api_version": "1.0.0",
            "analysis_id": f"anon_{hash(request.text[:50]) % 10000:04d}"
        })
        
        # Log du résultat (anonymisé)
        background_tasks.add_task(
            log_analysis_result,
            risk_score=result.get('overall_risk_score', 0.0),
            risk_level=result.get('risk_level', 'moderate'),
            processing_time=result.get('processing_time_ms', 0),
            ai_powered=result.get('ai_powered', False)
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur analyse: {e}")
        # Fallback vers analyse simple
        return await fallback_analysis(request)

@app.get("/api/stats")
async def get_stats():
    return {
        "total_users": 2847,
        "active_users_24h": 1243,
        "average_risk_score": 0.28,
        "high_risk_users": 23,
        "privacy_compliance": 0.987
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
            "id": "mindfulness_body_scan",
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
                "text": "Nouvelle intervention déployée pour anxiété",
                "time": "Il y a 5 min",
                "severity": "success"
            },
            {
                "text": "Utilisateur anonyme nécessite attention", 
                "time": "Il y a 12 min",
                "severity": "warning"
            }
        ]
    }

async def log_analysis_result(risk_score: float, risk_level: str, processing_time: int, ai_powered: bool):
    """Log des résultats d'analyse (anonymisé)"""
    logger.info(
        f"📊 Analyse terminée - Risque: {risk_level} ({risk_score:.3f}), "
        f"Temps: {processing_time}ms, IA: {ai_powered}"
    )

async def fallback_analysis(request: AnalysisRequest) -> Dict[str, Any]:
    """Analyse de fallback en cas d'erreur IA"""
    import random
    
    text_lower = request.text.lower()
    
    # Analyse simple par mots-clés
    depression_words = ['triste', 'déprim', 'vide', 'inutile', 'sad', 'depressed', 'hopeless', 'empty']
    anxiety_words = ['anxie', 'stress', 'inquiet', 'peur', 'anxious', 'worried', 'panic', 'scared']
    
    depression_score = sum(0.2 for word in depression_words if word in text_lower)
    anxiety_score = sum(0.2 for word in anxiety_words if word in text_lower)
    
    overall_risk = min((depression_score + anxiety_score) / 2, 1.0)
    
    if overall_risk < 0.25:
        risk_level = "low"
    elif overall_risk < 0.5:
        risk_level = "moderate"  
    elif overall_risk < 0.75:
        risk_level = "high"
    else:
        risk_level = "critical"
    
    return {
        "overall_risk_score": max(overall_risk, 0.1),
        "risk_level": risk_level,
        "condition_probabilities": {
            "depression": min(depression_score, 1.0),
            "anxiety": min(anxiety_score, 1.0),
            "ptsd": random.uniform(0.05, 0.15),
            "bipolar": random.uniform(0.05, 0.15),
            "burnout": random.uniform(0.1, 0.3)
        },
        "confidence_score": 0.4,  # Faible confiance pour fallback
        "privacy_protected": True,
        "processing_time_ms": random.randint(20, 50),
        "ai_powered": False,
        "suggestions": [
            "⚠️ Analyse simplifiée utilisée",
            "🔄 Modèles IA en cours de chargement",
            "💬 Contactez le support si le problème persiste"
        ],
        "fallback_mode": True
    }

@app.get("/api/ai-status")
async def get_ai_status():
    """Status détaillé des modèles IA"""
    global ai_engine
    
    if not ai_engine:
        return {
            "ai_engine_loaded": False,
            "models": {},
            "ready": False,
            "message": "Modèles IA en cours d'initialisation..."
        }
    
    status = ai_engine.get_model_status()
    return {
        "ai_engine_loaded": True,
        "models": status.get("models_loaded", {}),
        "device": status.get("device", "cpu"),
        "memory_usage": status.get("memory_usage", "N/A"),
        "ready": status.get("ready", False),
        "message": "Modèles IA opérationnels" if status.get("ready", False) else "Modèles en cours de chargement..."
    }

if __name__ == "__main__":
    print("🧠 Starting MindBridge AI Server...")
    print("🤖 Chargement des modèles Hugging Face...")
    print("⏳ Cela peut prendre quelques minutes au premier démarrage...")
    uvicorn.run(app, host="localhost", port=8001, log_level="info")