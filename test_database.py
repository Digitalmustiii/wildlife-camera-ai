from database import WildlifeDatabase
from datetime import datetime

# Initialize database
db = WildlifeDatabase()

# Test 1: Log some detections
print("📝 Logging test detections...")
for i in range(5):
    detection = {
        'species': 'deer',
        'confidence': 0.85,
        'rarity': 'interesting',
        'bbox': (100, 100, 200, 200)
    }
    db.log_detection(detection)

detection = {
    'species': 'fox',
    'confidence': 0.92,
    'rarity': 'rare',
    'bbox': (150, 150, 250, 250)
}
db.log_detection(detection, snapshot_path='snapshots/fox_001.jpg')

print("✅ Logged 6 detections\n")

# Test 2: Get recent detections
print("📊 Recent detections:")
recent = db.get_recent_detections(limit=10)
for det in recent:
    print(f"  {det['timestamp'][:19]} - {det['species'].upper()} ({det['confidence']:.2f})")

# Test 3: Species statistics
print("\n📈 Species stats:")
stats = db.get_species_stats()
for s in stats:
    print(f"  {s['species']}: {s['total_sightings']} sightings (first: {s['first_seen'][:10]})")

# Test 4: Overall stats
print("\n🎯 Total stats:")
total = db.get_total_stats()
print(f"  Total detections: {total['total_detections']}")
print(f"  Unique species: {total['unique_species']}")
print(f"  Active days: {total['active_days']}")

# Test 5: Search
print("\n🔍 Search for 'deer':")
deer = db.search_species('deer', limit=3)
print(f"  Found {len(deer)} deer detections")

print("\n✅ Database working perfectly!")