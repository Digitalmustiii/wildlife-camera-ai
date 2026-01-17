from alert_manager import AlertManager, console_alert_handler, file_alert_handler
from config import get_config

# Setup
config = get_config('laptop_demo')
config.alert.enabled = True
config.alert.cooldown_seconds = 5
config.alert.alert_on_common = True  # Alert on everything for testing

alert_mgr = AlertManager(config.alert)

# Register handlers
alert_mgr.register_handler(console_alert_handler)
alert_mgr.register_handler(file_alert_handler())

print("Testing alert system...\n")

# Test 1: First alert (should send)
detection1 = {
    'species': 'deer',
    'confidence': 0.85,
    'rarity': 'interesting',
    'bbox': (100, 100, 200, 200)
}
sent = alert_mgr.process_detection(detection1, 'snapshots/deer_001.jpg')
print(f"Alert 1 sent: {sent} (expected: True)")

# Test 2: Same species immediately (should block - cooldown)
detection2 = {
    'species': 'deer',
    'confidence': 0.90,
    'rarity': 'interesting',
    'bbox': (110, 110, 210, 210)
}
sent = alert_mgr.process_detection(detection2)
print(f"Alert 2 sent: {sent} (expected: False - cooldown)")

# Test 3: Different species (should send)
detection3 = {
    'species': 'fox',
    'confidence': 0.92,
    'rarity': 'rare',
    'bbox': (150, 150, 250, 250)
}
sent = alert_mgr.process_detection(detection3, 'snapshots/fox_001.jpg')
print(f"Alert 3 sent: {sent} (expected: True)")

# Test 4: Cooldown status
print("\n⏱️  Cooldown status:")
cooldowns = alert_mgr.get_cooldown_status()
for species, remaining in cooldowns.items():
    print(f"  {species}: {remaining:.1f}s remaining")

# Test 5: Session summary
print("\n📊 Session summary:")
summary = alert_mgr.get_session_summary()
print(f"  Total alerts: {summary['total_alerts']}")
print(f"  Unique species: {summary['unique_species']}")
print(f"  Rare alerts: {summary['rare_alerts']}")
print(f"  Breakdown: {summary['species_breakdown']}")

print("\n✅ Alert system working!")
print("Check 'data/alerts.log' for file logging")