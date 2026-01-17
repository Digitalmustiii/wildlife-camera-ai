from config import get_config

# Test 1: Laptop demo preset
config = get_config('laptop_demo')
print(config)
print("\n" + "="*50 + "\n")

# Test 2: Custom settings
config = get_config('laptop_demo', detection_confidence_threshold=0.7)
print(f"Confidence: {config.detection.confidence_threshold}")

# Test 3: Save config
config.save('my_config.json')
print("\n✅ Config saved to my_config.json")

# Test 4: Load config
config2 = Config('my_config.json')
print("✅ Config loaded successfully")