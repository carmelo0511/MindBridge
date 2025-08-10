"""
MindBridge Local API Server

FastAPI server for local mental health processing
Provides REST endpoints for the web frontend and mobile apps
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import MindBridge components
from backend.core.mental_health_engine import MindBridgeEngine, UserProfile, RiskLevel
from backend.core.privacy_manager import MindBridgePrivacyManager, PrivacyLevel, DataType
from backend.models.gpt_oss_wrapper import MindBridgeGPTWrapper, ModelConfig, ModelSize, InferenceMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
mental_health_engine: Optional[MindBridgeEngine] = None
privacy_manager: Optional[MindBridgePrivacyManager] = None
gpt_wrapper: Optional[MindBridgeGPTWrapper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🧠 Starting MindBridge API Server...")
    
    global mental_health_engine, privacy_manager, gpt_wrapper
    
    try:
        # Initialize privacy manager
        privacy_manager = MindBridgePrivacyManager(
            privacy_level=PrivacyLevel.MAXIMUM,
            storage_path="./secure_storage"
        )
        
        # Initialize GPT wrapper
        config = ModelConfig(
            model_size=ModelSize.SMALL,
            quantization_enabled=True,
            pruning_ratio=0.1,
            differential_privacy=True,
            cultural_adaptation=["western", "collectivist"],
            supported_languages=["en", "fr", "es"],
            inference_mode=InferenceMode.REALTIME,
            max_sequence_length=512,
            batch_size=1,
            device="cpu"
        )
        gpt_wrapper = MindBridgeGPTWrapper(config)
        
        # Initialize mental health engine
        mental_health_engine = MindBridgeEngine()
        
        logger.info("✅ MindBridge services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize MindBridge services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down MindBridge API Server...")


# Create FastAPI app
app = FastAPI(
    title="MindBridge API",
    description="Privacy-first mental health detection and support API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Pydantic models
class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    cultural_context: Optional[str] = "general"
    language: Optional[str] = "en"
    user_id: Optional[str] = None


class AnalysisResponse(BaseModel):
    overall_risk_score: float
    risk_level: str
    condition_probabilities: Dict[str, float]
    confidence_score: float
    privacy_protected: bool
    processing_time_ms: int
    suggestions: List[str]


class UserStatsResponse(BaseModel):
    total_users: int
    active_users_24h: int
    average_risk_score: float
    high_risk_users: int
    privacy_compliance: float


class InterventionResponse(BaseModel):
    id: str
    type: str
    title: str
    description: str
    duration_minutes: int
    priority: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    privacy_level: str
    model_loaded: bool


# Dependency functions
async def get_mental_health_engine():
    if mental_health_engine is None:
        raise HTTPException(status_code=503, detail="Mental health engine not initialized")
    return mental_health_engine


async def get_privacy_manager():
    if privacy_manager is None:
        raise HTTPException(status_code=503, detail="Privacy manager not initialized")
    return privacy_manager


async def get_gpt_wrapper():
    if gpt_wrapper is None:
        raise HTTPException(status_code=503, detail="GPT wrapper not initialized")
    return gpt_wrapper


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    return HealthResponse(
        status="healthy" if all([mental_health_engine, privacy_manager, gpt_wrapper]) else "degraded",
        timestamp=datetime.now().isoformat(),
        services={
            "mental_health_engine": "running" if mental_health_engine else "stopped",
            "privacy_manager": "running" if privacy_manager else "stopped",
            "gpt_wrapper": "running" if gpt_wrapper else "stopped",
        },
        privacy_level="maximum",
        model_loaded=gpt_wrapper is not None
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_mental_health(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    engine: MindBridgeEngine = Depends(get_mental_health_engine),
    privacy: MindBridgePrivacyManager = Depends(get_privacy_manager),
    gpt: MindBridgeGPTWrapper = Depends(get_gpt_wrapper)
):
    """Analyze text for mental health indicators"""
    try:
        start_time = datetime.now()
        
        # Process with privacy protections
        processed_request = privacy.process_sensitive_data(
            data={"text": request.text, "context": request.cultural_context},
            data_type=DataType.TEXT_ANALYSIS,
            operation="analyze"
        )
        
        # Analyze with GPT wrapper
        analysis_result = await gpt.analyze_mental_health_text(
            text=request.text,
            cultural_context=request.cultural_context,
            language=request.language
        )
        
        # Generate suggestions based on analysis
        suggestions = []
        if analysis_result['overall_risk_score'] > 0.7:
            suggestions = [
                "Considérer une consultation professionnelle immédiate",
                "Contacter une ligne d'écoute si nécessaire",
                "Utiliser les techniques de respiration guidée"
            ]
        elif analysis_result['overall_risk_score'] > 0.4:
            suggestions = [
                "Pratiquer des exercices de mindfulness",
                "Maintenir un journal personnel",
                "Considérer des techniques CBT"
            ]
        else:
            suggestions = [
                "Continuer les pratiques de bien-être actuelles",
                "Check-in réguliers recommandés",
                "Maintenir un mode de vie équilibré"
            ]
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Log analysis (anonymized)
        background_tasks.add_task(
            log_analysis_event,
            risk_score=analysis_result['overall_risk_score'],
            processing_time=processing_time
        )
        
        return AnalysisResponse(
            overall_risk_score=analysis_result['overall_risk_score'],
            risk_level=analysis_result.get('risk_level', 'low'),
            condition_probabilities=analysis_result['condition_probabilities'],
            confidence_score=analysis_result['confidence_score'],
            privacy_protected=True,
            processing_time_ms=processing_time,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.get("/api/stats", response_model=UserStatsResponse)
async def get_user_statistics(
    privacy: MindBridgePrivacyManager = Depends(get_privacy_manager)
):
    """Get anonymized user statistics"""
    try:
        # Simulate statistics (in real app, would query database)
        stats = {
            "total_users": privacy.differential_privacy.privatize_count(2847),
            "active_users_24h": privacy.differential_privacy.privatize_count(1243),
            "average_risk_score": privacy.differential_privacy.add_laplace_noise(0.28),
            "high_risk_users": privacy.differential_privacy.privatize_count(23),
            "privacy_compliance": 0.987
        }
        
        return UserStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed")


@app.get("/api/interventions", response_model=List[InterventionResponse])
async def get_interventions():
    """Get available interventions"""
    try:
        # Mock interventions data
        interventions = [
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
            },
            {
                "id": "cbt_thought_record",
                "type": "cbt",
                "title": "Journal de Pensées",
                "description": "Technique CBT pour identifier les schémas de pensée",
                "duration_minutes": 15,
                "priority": 3
            }
        ]
        
        return [InterventionResponse(**intervention) for intervention in interventions]
        
    except Exception as e:
        logger.error(f"Interventions retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Interventions retrieval failed")


@app.get("/api/privacy-status")
async def get_privacy_status(
    privacy: MindBridgePrivacyManager = Depends(get_privacy_manager)
):
    """Get privacy compliance status"""
    try:
        metrics = privacy.get_privacy_metrics()
        compliance = privacy.verify_privacy_compliance()
        
        return {
            "privacy_level": metrics['privacy_level'],
            "compliance_score": compliance['compliance_score'],
            "operations_count": metrics['operations']['total_operations'],
            "encryption_active": compliance['checks']['encryption_enabled'],
            "last_audit": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Privacy status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Privacy status retrieval failed")


@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        # Mock dashboard data
        data = {
            "weekly_trend": [
                {"day": "Lun", "risk_score": 0.25, "users": 45},
                {"day": "Mar", "risk_score": 0.32, "users": 52},
                {"day": "Mer", "risk_score": 0.28, "users": 48},
                {"day": "Jeu", "risk_score": 0.35, "users": 61},
                {"day": "Ven", "risk_score": 0.29, "users": 43},
                {"day": "Sam", "risk_score": 0.22, "users": 38},
                {"day": "Dim", "risk_score": 0.26, "users": 41},
            ],
            "condition_distribution": [
                {"name": "Anxiété", "value": 35, "color": "#FF9800"},
                {"name": "Dépression", "value": 28, "color": "#3F51B5"},
                {"name": "Burnout", "value": 20, "color": "#FF5722"},
                {"name": "PTSD", "value": 12, "color": "#795548"},
                {"name": "Bipolaire", "value": 5, "color": "#9C27B0"},
            ],
            "risk_levels": [
                {"level": "Faible", "count": 234, "color": "#4CAF50"},
                {"level": "Modéré", "count": 89, "color": "#FF9800"},
                {"level": "Élevé", "count": 23, "color": "#F44336"},
                {"level": "Critique", "count": 3, "color": "#D32F2F"},
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
                },
                {
                    "text": "Rapport de confidentialité généré",
                    "time": "Il y a 1h",
                    "severity": "info"
                }
            ]
        }
        
        return data
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard data retrieval failed")


# Background tasks
async def log_analysis_event(risk_score: float, processing_time: int):
    """Log analysis event for analytics (anonymized)"""
    try:
        logger.info(f"Analysis completed: risk={risk_score:.3f}, time={processing_time}ms")
    except Exception as e:
        logger.error(f"Failed to log analysis event: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    # Get configuration from environment
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "localhost")
    
    logger.info(f"🚀 Starting MindBridge API Server on {host}:{port}")
    
    uvicorn.run(
        "backend.api.local_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )