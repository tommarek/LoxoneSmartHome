# TODO - Loxone Smart Home

## Current Priority 🎯

### Loxone Control Integration
- [ ] Implement ability to control charging/discharging/export from Loxone
  - [ ] Design MQTT topic structure for Loxone commands
  - [ ] Add MQTT subscribers for incoming control commands
  - [ ] Implement override logic for automated scheduling
  - [ ] Add manual mode vs automatic mode switching
  - [ ] Handle conflicts between manual commands and price-based automation
  - [ ] Add status feedback to Loxone (current mode, battery state, etc.)
  - [ ] Create configuration for control topics and override behavior
  - [ ] Add comprehensive testing for control logic
  - [ ] Update documentation with control API

## Implementation Notes

### Command Structure (Proposed)
- `loxone/growatt/charge/enable` - Manual charge enable/disable
- `loxone/growatt/discharge/enable` - Manual discharge control  
- `loxone/growatt/export/enable` - Manual export control
- `loxone/growatt/mode` - Switch between "auto" and "manual" modes
- `loxone/growatt/status` - Status feedback to Loxone

### Override Logic
- Manual commands should temporarily override automated scheduling
- Need timeout mechanism to return to automatic mode
- Preserve safety limits (battery protection, grid constraints)
- Log all manual interventions for audit trail

### Safety Considerations
- Validate all incoming commands
- Implement rate limiting for control messages
- Maintain battery protection regardless of control source
- Add emergency stop functionality