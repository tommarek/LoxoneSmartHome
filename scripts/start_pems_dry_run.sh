#!/bin/bash
# PEMS v2 Dry Run Service Startup Script
#
# This script starts the PEMS v2 dry run service using Docker Compose,
# with proper data directory setup and configuration validation.

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/pems_dry_run"
LOG_DIR="$PROJECT_DIR/logs/pems_dry_run"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Setup data directories
setup_directories() {
    log "Setting up data directories..."
    
    # Create directories with proper permissions
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    
    # Set permissions (if needed)
    chmod 755 "$DATA_DIR" "$LOG_DIR"
    
    success "Data directories created: $DATA_DIR, $LOG_DIR"
}

# Validate configuration
validate_configuration() {
    log "Validating configuration..."
    
    # Check if docker-compose file exists
    COMPOSE_FILE="$PROJECT_DIR/docker-compose.pems-dry-run.yml"
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Check if PEMS service file exists
    SERVICE_FILE="$PROJECT_DIR/pems_dry_run_service.py"
    if [[ ! -f "$SERVICE_FILE" ]]; then
        error "PEMS service file not found: $SERVICE_FILE"
        exit 1
    fi
    
    # Check if Dockerfile exists
    DOCKERFILE="$PROJECT_DIR/Dockerfile.pems-dry-run"
    if [[ ! -f "$DOCKERFILE" ]]; then
        error "Dockerfile not found: $DOCKERFILE"
        exit 1
    fi
    
    success "Configuration validation passed"
}

# Build Docker image
build_image() {
    log "Building PEMS dry run Docker image..."
    
    cd "$PROJECT_DIR"
    
    # Build the image
    docker-compose -f docker-compose.pems-dry-run.yml build pems-dry-run
    
    if [[ $? -eq 0 ]]; then
        success "Docker image built successfully"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Start services
start_services() {
    log "Starting PEMS dry run services..."
    
    cd "$PROJECT_DIR"
    
    # Start services in detached mode
    docker-compose -f docker-compose.pems-dry-run.yml up -d pems-dry-run
    
    if [[ $? -eq 0 ]]; then
        success "PEMS dry run service started successfully"
    else
        error "Failed to start PEMS dry run service"
        exit 1
    fi
}

# Check service health
check_health() {
    log "Checking service health..."
    
    # Wait a moment for service to start
    sleep 10
    
    # Check if container is running
    if docker-compose -f "$PROJECT_DIR/docker-compose.pems-dry-run.yml" ps pems-dry-run | grep -q "Up"; then
        success "PEMS dry run service is running"
        
        # Show recent logs
        log "Recent service logs:"
        docker-compose -f "$PROJECT_DIR/docker-compose.pems-dry-run.yml" logs --tail=20 pems-dry-run
        
    else
        error "PEMS dry run service is not running properly"
        
        # Show logs for debugging
        warning "Service logs for debugging:"
        docker-compose -f "$PROJECT_DIR/docker-compose.pems-dry-run.yml" logs pems-dry-run
        
        exit 1
    fi
}

# Show service information
show_service_info() {
    echo ""
    echo "=================================================="
    echo "🚀 PEMS v2 Dry Run Service Information"
    echo "=================================================="
    echo ""
    echo "📂 Data Directory: $DATA_DIR"
    echo "📋 Log Directory: $LOG_DIR"
    echo ""
    echo "🎮 Useful Commands:"
    echo "  # View live logs"
    echo "  docker-compose -f docker-compose.pems-dry-run.yml logs -f pems-dry-run"
    echo ""
    echo "  # Stop service"
    echo "  docker-compose -f docker-compose.pems-dry-run.yml down"
    echo ""
    echo "  # Restart service"
    echo "  docker-compose -f docker-compose.pems-dry-run.yml restart pems-dry-run"
    echo ""
    echo "  # Check service status"
    echo "  docker-compose -f docker-compose.pems-dry-run.yml ps"
    echo ""
    echo "  # Analyze collected data"
    echo "  python scripts/analyze_pems_data.py --data-dir $DATA_DIR --output-dir ./analysis"
    echo ""
    echo "📊 Data Collection:"
    echo "  - Optimization cycles: $DATA_DIR/optimization_cycles.jsonl"
    echo "  - MQTT commands: $DATA_DIR/mqtt_commands.jsonl"
    echo "  - System states: $DATA_DIR/system_states.jsonl"
    echo "  - Service metrics: $DATA_DIR/service_metrics.json"
    echo ""
    echo "🌐 Optional Services:"
    echo "  # Start dashboard (Grafana)"
    echo "  docker-compose -f docker-compose.pems-dry-run.yml up -d pems-dashboard"
    echo "  # Access at: http://localhost:3000 (admin/admin)"
    echo ""
    echo "  # Start log viewer (Seq)"  
    echo "  docker-compose -f docker-compose.pems-dry-run.yml up -d pems-logs"
    echo "  # Access at: http://localhost:5341"
    echo ""
    echo "=================================================="
}

# Main execution
main() {
    echo "🚀 Starting PEMS v2 Dry Run Service"
    echo "===================================="
    echo ""
    
    check_prerequisites
    setup_directories
    validate_configuration
    build_image
    start_services
    check_health
    show_service_info
    
    echo ""
    success "PEMS v2 Dry Run Service is now running!"
    echo ""
    echo "💡 Monitor the service with:"
    echo "   docker-compose -f docker-compose.pems-dry-run.yml logs -f pems-dry-run"
}

# Handle script arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "stop")
        log "Stopping PEMS dry run services..."
        cd "$PROJECT_DIR"
        docker-compose -f docker-compose.pems-dry-run.yml down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting PEMS dry run services..."
        cd "$PROJECT_DIR"
        docker-compose -f docker-compose.pems-dry-run.yml restart
        success "Services restarted"
        ;;
    "logs")
        log "Showing service logs..."
        cd "$PROJECT_DIR"
        docker-compose -f docker-compose.pems-dry-run.yml logs -f pems-dry-run
        ;;
    "status")
        log "Checking service status..."
        cd "$PROJECT_DIR"
        docker-compose -f docker-compose.pems-dry-run.yml ps
        ;;
    "analyze")
        log "Running data analysis..."
        if [[ -d "$DATA_DIR" ]] && [[ -f "$DATA_DIR/optimization_cycles.jsonl" ]]; then
            python "$SCRIPT_DIR/analyze_pems_data.py" --data-dir "$DATA_DIR" --output-dir "./analysis"
        else
            warning "No data found to analyze. Service may not have collected data yet."
        fi
        ;;
    "help"|"-h"|"--help")
        echo "PEMS v2 Dry Run Service Management"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start     Start the service (default)"
        echo "  stop      Stop the service"
        echo "  restart   Restart the service"
        echo "  logs      Show live logs"
        echo "  status    Show service status"
        echo "  analyze   Analyze collected data"
        echo "  help      Show this help"
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac