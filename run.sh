#!/bin/bash

# Loxone Smart Home - Deployment Script

echo "🏠 Starting Loxone Smart Home system..."

# Build and start all services
echo "Building consolidated service..."
docker-compose build loxone_smart_home

echo "Starting all services..."
docker-compose up -d

# Wait a moment for services to start
sleep 5

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps

# Show access information
echo ""
echo "🌐 Access Information:"
echo "  Grafana:  http://localhost:3000 (admin/adminadmin)"
echo "  InfluxDB: http://localhost:8086"
echo ""

# Check logs for any immediate issues
echo "📝 Recent logs from consolidated service:"
docker-compose logs --tail=10 loxone_smart_home

echo ""
echo "✅ Deployment complete!"
echo ""
echo "💡 Useful commands:"
echo "  View logs:        docker-compose logs -f loxone_smart_home"
echo "  Check status:     docker-compose ps"
echo "  Stop services:    docker-compose down"
echo "  Restart service:  docker-compose restart loxone_smart_home"