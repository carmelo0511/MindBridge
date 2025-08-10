"""
MindBridge Simple Demo Server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
import random

app = FastAPI(
    title="MindBridge API Demo",
    description="Demo API for MindBridge Mental Health Platform",
    version="1.0.0"
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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "mental_health_engine": "running",
            "privacy_manager": "running",
            "gpt_wrapper": "running"
        },
        "privacy_level": "maximum",
        "model_loaded": True
    }

@app.post("/api/analyze")
async def analyze_text(request: AnalysisRequest):
    # Simulate analysis based on text sentiment
    text_lower = request.text.lower()
    
    # Simple keyword-based analysis
    depression_words = ['sad', 'depressed', 'hopeless', 'empty', 'worthless']
    anxiety_words = ['anxious', 'worried', 'panic', 'scared', 'nervous']
    
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
        "confidence_score": random.uniform(0.7, 0.9),
        "privacy_protected": True,
        "processing_time_ms": random.randint(80, 120),
        "suggestions": [
            "Exercices de respiration recommandés",
            "Considérer un check-in quotidien",
            "Techniques de mindfulness disponibles"
        ]
    }

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

if __name__ == "__main__":
    print("🧠 Starting MindBridge Demo Server...")
    uvicorn.run(app, host="localhost", port=8001, log_level="info")