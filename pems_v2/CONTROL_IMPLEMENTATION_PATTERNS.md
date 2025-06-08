# PEMS v2 Control Implementation Patterns

This document shows how the PEMS v2 control system now matches the actual implementation patterns found in `growatt_controller.py` and `mqtt_bridge.py`.

## Battery Control Pattern (Following growatt_controller.py)

### Battery-First Mode Control
```python
# PEMS v2 pattern matches growatt_controller.py:354-374
async def _set_battery_first_mode(self, start_hour: str, stop_hour: str) -> None:
    payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}
    await self.mqtt_client.publish(battery_first_topic, json.dumps(payload))
    self.logger.info(f"🔋 BATTERY-FIRST MODE SET: {start_hour}-{stop_hour}")
```

### AC Charging Control
```python
# PEMS v2 pattern matches growatt_controller.py:376-391
async def _enable_ac_charge(self, power_kw: float) -> None:
    payload = {"value": True, "power_kw": power_kw}
    await self.mqtt_client.publish(ac_charge_topic, json.dumps(payload))
    self.logger.info(f"⚡ AC CHARGING ENABLED at {power_kw}kW")
```

### Simulation Mode Support
```python
# PEMS v2 pattern matches growatt_controller.py simulation checks
if self.config.get('simulation_mode', False):
    current_time = datetime.now().strftime("%H:%M:%S")
    self.logger.info(f"🔋 [SIMULATE] BATTERY-FIRST MODE SET (simulated at {current_time})")
    return
```

## Export Control Pattern (Following growatt_controller.py)

### Enable Export
```python
# PEMS v2 pattern matches growatt_controller.py:430-447
async def enable_export(self, limit_kw: Optional[float] = None) -> bool:
    payload = {"value": True}
    if limit_kw is not None:
        payload["limit_kw"] = limit_kw
    
    await self.mqtt_client.publish(export_enable_topic, json.dumps(payload))
    self.logger.info(f"⬆️ EXPORT ENABLED at {current_time} → Topic: {export_enable_topic}")
```

### Disable Export
```python
# PEMS v2 pattern matches growatt_controller.py:449-466
async def disable_export(self) -> bool:
    payload = {"value": False}
    await self.mqtt_client.publish(export_disable_topic, json.dumps(payload))
    self.logger.info(f"⬇️ EXPORT DISABLED at {current_time} → Topic: {export_disable_topic}")
```

## MQTT Communication Pattern (Following mqtt_bridge.py)

### Heating Control to Loxone
```python
# PEMS v2 pattern follows mqtt_bridge.py:41-70 structure
# Send JSON to MQTT, let mqtt_bridge convert to Loxone format
payload_data = {
    "room": command.room,
    "state": "on" if command.state else "off",
    "timestamp": command.timestamp.isoformat()
}

# MQTT bridge will convert this to: "room=kitchen;state=on;timestamp=2024-01-01T12:00:00"
await self.mqtt_client.publish(topic, json.dumps(payload_data))
```

### Topic Structure
```python
# PEMS v2 follows the established topic patterns:
# Control commands: pems/{system}/{room}/set
# Status updates: loxone/{system}/{room}/status (from Loxone)
# Bridge topics: Configured in settings.loxone_bridge.bridge_topics
```

## Integration Points

### 1. Battery Control Integration
- **Topic**: Uses `battery_first_topic` and `ac_charge_topic` from growatt_controller config
- **Payload Format**: Matches exact JSON structure from growatt_controller.py
- **Logging**: Uses same emoji patterns and message format
- **Simulation**: Supports simulation_mode flag like growatt_controller

### 2. Export Control Integration  
- **Topic**: Uses `export_enable_topic` and `export_disable_topic`
- **Payload Format**: Simple `{"value": true/false}` pattern
- **Timing**: Includes timestamp logging like growatt_controller
- **Safety**: Follows same disable-on-emergency pattern

### 3. Heating Control Integration
- **Topic Structure**: `pems/heating/{room}/set` for commands
- **Status Topics**: `loxone/heating/{room}/status` for feedback
- **MQTT Bridge**: Relies on mqtt_bridge.py to convert JSON to Loxone format
- **Data Flow**: JSON → MQTT → Bridge → UDP → Loxone

## Configuration Compatibility

The PEMS v2 control system now expects these MQTT topics in configuration:

```python
mqtt_config = {
    # Battery control (from growatt_controller)
    'battery_first_topic': 'growatt/battery/first_mode',
    'ac_charge_topic': 'growatt/battery/ac_charge',
    
    # Export control (from growatt_controller)
    'export_enable_topic': 'growatt/export/enable',
    'export_disable_topic': 'growatt/export/disable',
    
    # Heating control (new for PEMS v2)
    'heating_topic_prefix': 'pems/heating'
}
```

## Operational Flow

1. **PEMS v2 Optimization** → Generates control schedules
2. **Control System** → Translates to specific MQTT commands
3. **MQTT Broker** → Routes commands to appropriate systems
4. **Growatt Controller** → Handles battery/export commands directly
5. **MQTT Bridge** → Converts heating commands for Loxone
6. **Loxone System** → Executes heating relay control
7. **Status Feedback** → Returns to PEMS v2 for validation

This integration ensures PEMS v2 works seamlessly with the existing infrastructure while maintaining all safety features and operational patterns.