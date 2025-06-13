#!/usr/bin/env python3
"""
Test script for YOLO8m microservice
"""

import requests
import json
from pathlib import Path
import time

def test_microservice():
    """Test the YOLO8m microservice"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing YOLO8m Microservice...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Service is healthy")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   ONNX enabled: {health_data['onnx_enabled']}")
            print(f"   Available classes: {health_data['available_classes']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test available classes
    print("\n2. Available Classes:")
    try:
        response = requests.get(f"{base_url}/classes")
        if response.status_code == 200:
            classes_data = response.json()
            print(f"✅ Total classes: {classes_data['total_classes']}")
            print(f"   First 10 classes: {classes_data['classes'][:10]}")
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Classes endpoint error: {e}")
    
    # Test object detection with image
    print("\n3. Object Detection Test:")
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        if image_files:
            test_image = image_files[0]
            print(f"   Using test image: {test_image}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'file': (test_image.name, f, 'image/jpeg')}
                    params = {'confidence': 0.3}
                    
                    print("   Sending detection request...")
                    start_time = time.time()
                    response = requests.post(f"{base_url}/detect", files=files, params=params)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Detection successful!")
                        print(f"   Processing time: {end_time - start_time:.2f} seconds")
                        print(f"   Objects detected: {result['count']}")
                        
                        if result['objects']:
                            print("   Detected objects:")
                            for i, obj in enumerate(result['objects'][:5]):  # Show first 5
                                print(f"      {i+1}. {obj['label']} (confidence: {obj['confidence']})")
                            
                            if result['count'] > 5:
                                print(f"      ... and {result['count'] - 5} more objects")
                    else:
                        print(f"❌ Detection failed: {response.status_code}")
                        print(f"   Error: {response.text}")
                        
            except Exception as e:
                print(f"❌ Detection error: {e}")
        else:
            print("   ⚠️  No test images found in test_images directory")
    else:
        print("   ⚠️  test_images directory not found")
    
    # Test specific object detection (person)
    print("\n4. Specific Object Detection Test (person):")
    if test_images_dir.exists() and image_files:
        test_image = image_files[0]
        try:
            with open(test_image, 'rb') as f:
                files = {'file': (test_image.name, f, 'image/jpeg')}
                params = {'confidence': 0.3}
                
                start_time = time.time()
                response = requests.post(f"{base_url}/detect/person", files=files, params=params)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Person detection successful!")
                    print(f"   Processing time: {end_time - start_time:.2f} seconds")
                    print(f"   Persons detected: {result['count']}")
                    
                    if result['objects']:
                        for i, obj in enumerate(result['objects']):
                            print(f"      Person {i+1}: confidence {obj['confidence']}")
                else:
                    print(f"❌ Person detection failed: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Person detection error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 YOLO8m Microservice Test Complete!")

if __name__ == "__main__":
    test_microservice() 