#!/bin/bash

# MindBridge Deployment Script
# Automated setup and deployment for MindBridge mental health platform

set -e

echo "🧠 MindBridge Deployment Script"
echo "=================================="

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/deployment/docker/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js 16+ first."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Create environment file
create_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating environment configuration..."
        
        cat > "$ENV_FILE" << EOF
# MindBridge Environment Configuration

# Privacy Configuration
PRIVACY_LEVEL=maximum
ENCRYPTION_KEY=$(openssl rand -hex 32)
DIFFERENTIAL_PRIVACY_EPSILON=1.0

# Model Configuration
MODEL_SIZE=small
QUANTIZATION_ENABLED=true
CULTURAL_ADAPTATION=true

# API Configuration
API_PORT=8000
API_HOST=localhost
CORS_ORIGINS=["http://localhost:3000"]

# Analytics (Privacy-Preserving)
ANALYTICS_ENABLED=true
FEDERATED_LEARNING=true

# Development
DEBUG=false
LOG_LEVEL=INFO
EOF
        
        log_success "Environment file created at $ENV_FILE"
    else
        log_info "Environment file already exists"
    fi
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/models"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/temp"
    
    log_success "Directories created"
}

# Build backend
build_backend() {
    log_info "Building backend..."
    
    cd "${PROJECT_ROOT}/backend"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    log_success "Backend built successfully"
}

# Build mobile app
build_mobile() {
    log_info "Building mobile app..."
    
    cd "${PROJECT_ROOT}/mobile"
    
    # Install dependencies
    npm install
    
    # Install pods for iOS (if on macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        cd ios
        pod install
        cd ..
    fi
    
    log_success "Mobile app dependencies installed"
}

# Build web frontend
build_frontend() {
    log_info "Building web frontend..."
    
    cd "${PROJECT_ROOT}/frontend/app"
    
    # Install dependencies
    npm install
    
    # Build for production
    npm run build
    
    log_success "Web frontend built successfully"
}

# Start services with Docker Compose
start_services() {
    log_info "Starting MindBridge services..."
    
    cd "${PROJECT_ROOT}"
    
    # Build and start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up --build -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check health
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "healthy"; then
        log_success "Services started successfully"
        echo ""
        echo "🌐 MindBridge is now running:"
        echo "   Backend API: http://localhost:8000"
        echo "   Web Dashboard: http://localhost:3000"
        echo "   Health Check: http://localhost:8000/health"
        echo ""
    else
        log_error "Some services failed to start properly"
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    cd "${PROJECT_ROOT}/backend"
    source venv/bin/activate
    
    # Run backend tests
    python -m pytest tests/ -v --cov=backend
    
    # Run frontend tests
    cd "${PROJECT_ROOT}/frontend/app"
    npm test -- --watchAll=false
    
    # Run mobile tests
    cd "${PROJECT_ROOT}/mobile"
    npm test -- --watchAll=false
    
    log_success "All tests passed"
}

# Privacy compliance check
privacy_check() {
    log_info "Running privacy compliance check..."
    
    cd "${PROJECT_ROOT}/backend"
    source venv/bin/activate
    
    python -c "
from backend.core.privacy_manager import MindBridgePrivacyManager
from backend.core.privacy_manager import PrivacyLevel

# Initialize privacy manager
pm = MindBridgePrivacyManager(PrivacyLevel.MAXIMUM)

# Run compliance check
compliance = pm.verify_privacy_compliance()
print(f'Privacy Compliance Score: {compliance[\"compliance_score\"]:.2%}')

if compliance['compliance_score'] >= 0.95:
    print('✅ Privacy compliance check PASSED')
    exit(0)
else:
    print('❌ Privacy compliance check FAILED')
    print('Issues:')
    for check, result in compliance['checks'].items():
        if not result:
            print(f'  - {check}')
    exit(1)
"
    
    log_success "Privacy compliance verified"
}

# Deployment modes
deploy_development() {
    log_info "Deploying MindBridge in development mode..."
    
    check_prerequisites
    create_env_file
    setup_directories
    build_backend
    build_mobile
    build_frontend
    start_services
    
    log_success "Development deployment completed!"
}

deploy_production() {
    log_info "Deploying MindBridge in production mode..."
    
    check_prerequisites
    create_env_file
    setup_directories
    build_backend
    build_frontend
    run_tests
    privacy_check
    
    # Update environment for production
    sed -i.bak 's/DEBUG=true/DEBUG=false/g' "$ENV_FILE"
    sed -i.bak 's/LOG_LEVEL=DEBUG/LOG_LEVEL=INFO/g' "$ENV_FILE"
    
    start_services
    
    log_success "Production deployment completed!"
}

stop_services() {
    log_info "Stopping MindBridge services..."
    
    cd "${PROJECT_ROOT}"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    log_success "Services stopped"
}

cleanup() {
    log_info "Cleaning up MindBridge deployment..."
    
    cd "${PROJECT_ROOT}"
    
    # Stop and remove containers
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --volumes --rmi all
    
    # Clean up build artifacts
    rm -rf backend/venv
    rm -rf frontend/app/build
    rm -rf mobile/node_modules
    rm -rf frontend/app/node_modules
    
    # Clean up data (with confirmation)
    read -p "Do you want to delete all user data? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/*
        rm -rf models/*
        log_warning "All user data deleted"
    fi
    
    log_success "Cleanup completed"
}

# Main execution
case "${1:-dev}" in
    "dev"|"development")
        deploy_development
        ;;
    "prod"|"production")
        deploy_production
        ;;
    "stop")
        stop_services
        ;;
    "cleanup")
        cleanup
        ;;
    "test")
        run_tests
        ;;
    "privacy-check")
        privacy_check
        ;;
    *)
        echo "Usage: $0 [dev|prod|stop|cleanup|test|privacy-check]"
        echo ""
        echo "Commands:"
        echo "  dev           - Deploy in development mode (default)"
        echo "  prod          - Deploy in production mode"
        echo "  stop          - Stop all services"
        echo "  cleanup       - Clean up deployment and data"
        echo "  test          - Run test suite"
        echo "  privacy-check - Verify privacy compliance"
        exit 1
        ;;
esac