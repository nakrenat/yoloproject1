#!/usr/bin/env python3
"""
Test script for animal detection using YOLO8m microservice
"""

import requests
import json
import time
from pathlib import Path

def test_animal_detection():
    """Test animal detection with sample9_animals.jpg"""
    base_url = "http://localhost:8000"
    image_path = "test_images/sample9_animals.jpg"
    
    print("🐾 Testing Animal Detection with YOLO8m...")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Testing with image: {image_path}")
    
    # Test general object detection
    print("\n1. General Object Detection:")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('sample9_animals.jpg', f, 'image/jpeg')}
            params = {'confidence': 0.3}
            
            print("   Sending detection request...")
            start_time = time.time()
            response = requests.post(f"{base_url}/detect", files=files, params=params)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Detection successful!")
                print(f"   Processing time: {end_time - start_time:.2f} seconds")
                print(f"   Total objects detected: {result['count']}")
                
                if result['objects']:
                    print("\n   🎯 Detected objects:")
                    for i, obj in enumerate(result['objects']):
                        print(f"      {i+1}. {obj['label']} (confidence: {obj['confidence']}) "
                              f"[{obj['x']}, {obj['y']}, {obj['width']}x{obj['height']}]")
                else:
                    print("   No objects detected")
                    
            else:
                print(f"❌ Detection failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Detection error: {e}")
    
    # Test specific animal detection
    animals = ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    
    print(f"\n2. Specific Animal Detection:")
    print("   Testing common animals...")
    
    for animal in animals:
        try:
            with open(image_path, 'rb') as f:
                files = {'file': ('sample9_animals.jpg', f, 'image/jpeg')}
                params = {'confidence': 0.3}
                
                response = requests.post(f"{base_url}/detect/{animal}", files=files, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    if result['count'] > 0:
                        print(f"   🐾 {animal.upper()}: {result['count']} detected")
                        for obj in result['objects']:
                            print(f"      └─ Confidence: {obj['confidence']}")
                            
        except Exception as e:
            print(f"   ❌ Error testing {animal}: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Animal Detection Test Complete!")

if __name__ == "__main__":
    test_animal_detection() 