// Loxone Smart Home Monitor - Dashboard JavaScript

// WebSocket connection
let ws = null;
let reconnectInterval = null;

// Chart instances
let productionChart = null;
let priceChart = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    initializeCharts();
    updateTime();
    setInterval(updateTime, 1000);
    fetchInitialData();
});

// Initialize WebSocket connection
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);

        // Clear reconnect interval
        if (reconnectInterval) {
            clearInterval(reconnectInterval);
            reconnectInterval = null;
        }
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);

        // Attempt to reconnect every 5 seconds
        if (!reconnectInterval) {
            reconnectInterval = setInterval(() => {
                console.log('Attempting to reconnect...');
                initializeWebSocket();
            }, 5000);
        }
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(message) {
    const { topic, data } = message;

    switch (topic) {
        case 'energy':
            updateEnergyFlow(data.energy);
            break;
        case 'battery':
            updateBatteryStatus(data.battery);
            break;
        case 'price':
            updatePrice(data.price);
            break;
        case 'weather':
            updateWeather(data.weather);
            break;
        case 'all':
            // Update all components
            if (data.energy) updateEnergyFlow(data.energy);
            if (data.battery) updateBatteryStatus(data.battery);
            if (data.price) updatePrice(data.price);
            if (data.weather) updateWeather(data.weather);
            break;
    }
}

// Update connection status
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    const statusTextEl = statusEl.querySelector('.status-text');

    if (connected) {
        statusEl.classList.add('connected');
        statusTextEl.textContent = 'Connected';
    } else {
        statusEl.classList.remove('connected');
        statusTextEl.textContent = 'Disconnected';
    }
}

// Update current time
function updateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('current-time').textContent = timeStr;
}

// Update energy flow display
function updateEnergyFlow(data) {
    if (!data) return;

    // Update power values
    updateElement('solar-power', formatPower(data.solar_power));
    updateElement('grid-power', formatPower(data.grid_power));
    updateElement('home-power', formatPower(data.home_power));
    updateElement('battery-power', formatPower(data.battery_power));

    // Update grid power color
    const gridEl = document.getElementById('grid-power');
    if (data.grid_power < 0) {
        gridEl.classList.add('export');
        gridEl.classList.remove('import');
    } else {
        gridEl.classList.add('import');
        gridEl.classList.remove('export');
    }
}

// Update battery status
function updateBatteryStatus(data) {
    if (!data) return;

    updateElement('battery-power', formatPower(data.power));
    updateElement('battery-soc', `${data.soc}%`);

    // Update SOC bar
    const socFill = document.getElementById('battery-soc-fill');
    if (socFill) {
        socFill.style.width = `${data.soc}%`;
    }
}

// Update price display
function updatePrice(data) {
    if (!data) return;

    updateElement('current-price', `${data.price_czk_kwh.toFixed(2)} CZK/kWh`);
    updateElement('price-level', data.level.toUpperCase());

    // Update price level styling
    const levelEl = document.getElementById('price-level');
    levelEl.className = `price-level ${data.level}`;

    // Update next change
    if (data.next_change) {
        const nextTime = new Date(data.next_change.time);
        updateElement('next-price-time', nextTime.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        }));
        updateElement('next-price-value', `${data.next_change.price_czk_kwh.toFixed(2)} CZK/kWh`);
    }
}

// Update weather display
function updateWeather(data) {
    if (!data) return;

    updateElement('temperature', `${data.temperature.toFixed(1)}°C`);
    updateElement('weather-desc', data.description);
    updateElement('cloud-cover', `${data.cloud_cover}%`);
    updateElement('uv-index', data.uv_index.toFixed(1));
}

// Initialize charts
function initializeCharts() {
    // Production Chart
    const productionCtx = document.getElementById('production-chart');
    if (productionCtx) {
        productionChart = new Chart(productionCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Production',
                    data: [],
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Consumption',
                    data: [],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'kWh'
                        }
                    }
                }
            }
        });
    }

    // Price Chart
    const priceCtx = document.getElementById('price-chart');
    if (priceCtx) {
        priceChart = new Chart(priceCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'CZK/kWh'
                        }
                    }
                }
            }
        });
    }
}

// Fetch initial data
async function fetchInitialData() {
    try {
        // Fetch current energy
        const energyRes = await fetch('/api/energy/current');
        const energyData = await energyRes.json();
        updateEnergyFlow(energyData);

        // Fetch current price
        const priceRes = await fetch('/api/prices/current');
        const priceData = await priceRes.json();
        updatePrice(priceData);

        // Fetch weather
        const weatherRes = await fetch('/api/weather/current');
        const weatherData = await weatherRes.json();
        updateWeather(weatherData);

        // Fetch energy history for chart
        fetchEnergyHistory();

        // Fetch price forecast for chart
        fetchPriceForecast();

        // Fetch schedule
        fetchSchedule();

        // Fetch statistics
        fetchStatistics();

    } catch (error) {
        console.error('Error fetching initial data:', error);
    }
}

// Fetch energy history
async function fetchEnergyHistory() {
    try {
        const res = await fetch('/api/energy/history?resolution=1h');
        const data = await res.json();

        if (productionChart && data.data) {
            const labels = data.data.map(d => {
                const date = new Date(d.timestamp);
                return date.toLocaleTimeString('en-US', { hour: '2-digit' });
            });

            const production = data.data.map(d => d.production);
            const consumption = data.data.map(d => d.consumption);

            productionChart.data.labels = labels;
            productionChart.data.datasets[0].data = production;
            productionChart.data.datasets[1].data = consumption;
            productionChart.update();
        }
    } catch (error) {
        console.error('Error fetching energy history:', error);
    }
}

// Fetch price forecast
async function fetchPriceForecast() {
    try {
        const res = await fetch('/api/prices/forecast?hours=48');
        const data = await res.json();

        if (priceChart && data.blocks) {
            const labels = data.blocks.map(b => {
                const date = new Date(b.timestamp);
                return date.toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit'
                });
            });

            const prices = data.blocks.map(b => b.price_czk_kwh);
            const colors = data.blocks.map(b => {
                if (b.level === 'low') return '#4CAF50';
                if (b.level === 'high') return '#F44336';
                return '#FF9800';
            });

            priceChart.data.labels = labels;
            priceChart.data.datasets[0].data = prices;
            priceChart.data.datasets[0].backgroundColor = colors;
            priceChart.data.datasets[0].borderColor = colors;
            priceChart.update();
        }
    } catch (error) {
        console.error('Error fetching price forecast:', error);
    }
}

// Fetch schedule
async function fetchSchedule() {
    try {
        console.log('Fetching schedule from /api/energy/schedule...');
        const res = await fetch('/api/energy/schedule');

        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        console.log('Schedule data received:', data);

        const container = document.getElementById('schedule-table-container');
        if (!container) {
            console.error('Schedule container element not found!');
            return;
        }

        if (!data.days || data.days.length === 0) {
            console.warn('No schedule days in response:', data);
            container.innerHTML = '<div class="error-message">No schedule data available.</div>';
            return;
        }

        // Build HTML for schedule table
        let html = '';

        // Add legend
        if (data.legend && data.legend.length > 0) {
            html += '<div class="schedule-legend">';
            data.legend.forEach(item => {
                html += `<span class="legend-item">
                    <span class="legend-icon">${item.icon}</span>
                    <span class="legend-label">${item.label}</span>
                </span>`;
            });
            html += '</div>';
        }

        // Add summary
        if (data.summary) {
            html += `<div class="schedule-summary">
                <span>Charge blocks: <strong>${data.summary.charge_blocks || 0}</strong></span>
                <span>Discharge blocks: <strong>${data.summary.discharge_blocks || 0}</strong></span>
                <span>Charge threshold: <strong>${(data.summary.charge_threshold || 0).toFixed(2)} CZK/kWh</strong></span>
                <span>Discharge threshold: <strong>${(data.summary.discharge_threshold || 0).toFixed(2)} CZK/kWh</strong></span>
            </div>`;
        }

        // Render each day
        data.days.forEach(day => {
            html += `<div class="schedule-day">
                <h4 class="schedule-day-title">${day.label} (${day.date})</h4>
                <div class="schedule-table-wrapper">
                    <table class="schedule-table">
                        <thead>
                            <tr>
                                <th>Hour</th>
                                <th>:00-:15</th>
                                <th>:15-:30</th>
                                <th>:30-:45</th>
                                <th>:45-:00</th>
                            </tr>
                        </thead>
                        <tbody>`;

            day.hours.forEach(hourData => {
                html += `<tr>
                    <td class="hour-cell">${String(hourData.hour).padStart(2, '0')}:00</td>`;

                // Render 4 blocks per hour
                for (let i = 0; i < 4; i++) {
                    if (hourData.blocks[i]) {
                        const block = hourData.blocks[i];
                        const modeClass = `mode-${block.mode}`;
                        html += `<td class="price-cell ${modeClass}">
                            <span class="price-value">${block.price_czk_kwh.toFixed(2)}</span>
                            <span class="mode-icon">${block.icon}</span>
                        </td>`;
                    } else {
                        html += `<td class="price-cell">-</td>`;
                    }
                }

                html += `</tr>`;
            });

            html += `</tbody>
                    </table>
                </div>
            </div>`;
        });

        container.innerHTML = html;
        console.log('Schedule table rendered successfully');
    } catch (error) {
        console.error('Error fetching schedule:', error);
        const container = document.getElementById('schedule-table-container');
        if (container) {
            container.innerHTML = `<div class="error-message">Failed to load schedule: ${error.message}</div>`;
        }
    }
}

// Fetch statistics
async function fetchStatistics() {
    try {
        const res = await fetch('/api/energy/statistics?period=day');
        const data = await res.json();

        updateElement('stat-production', `${data.production.total.toFixed(1)} kWh`);
        updateElement('stat-consumption', `${data.consumption.total.toFixed(1)} kWh`);
        updateElement('stat-import', `${data.grid.import.toFixed(1)} kWh`);
        updateElement('stat-export', `${data.grid.export.toFixed(1)} kWh`);
        updateElement('stat-savings', `${data.savings.amount.toFixed(0)} CZK`);
        updateElement('stat-co2', `${data.savings.co2_avoided.toFixed(1)} kg`);

        updateElement('self-sufficiency', `${data.self_sufficiency.toFixed(0)}%`);
        updateElement('efficiency', `${data.self_consumption.toFixed(0)}%`);
    } catch (error) {
        console.error('Error fetching statistics:', error);
    }
}

// Helper function to update element text
function updateElement(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = value;
        el.classList.add('fade-in');
    }
}

// Format power value
function formatPower(watts) {
    const absWatts = Math.abs(watts);
    if (absWatts >= 1000) {
        return `${(watts / 1000).toFixed(1)} kW`;
    }
    return `${watts.toFixed(0)} W`;
}

// Refresh data periodically
setInterval(() => {
    fetchEnergyHistory();
    fetchStatistics();
}, 60000); // Every minute

setInterval(() => {
    fetchPriceForecast();
    fetchSchedule();
}, 300000); // Every 5 minutes