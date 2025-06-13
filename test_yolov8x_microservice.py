#!/usr/bin/env python3
"""
Test script specifically for YOLOv8x microservice
This script verifies that the microservice is properly using YOLOv8x model
"""

import requests
import json
import os
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGES_DIR = "test_images"

class YOLOv8xMicroserviceTest:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_model_info(self):
        """Test and display model information."""
        print("🔍 Testing YOLOv8x model information...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            assert response.status_code == 200
            
            health_data = response.json()
            print(f"✅ Microservice running with model: {health_data.get('model_info', 'Unknown')}")
            
            # Get available classes
            classes_response = self.session.get(f"{self.base_url}/classes")
            if classes_response.status_code == 200:
                classes_data = classes_response.json()
                print(f"📋 Model can detect {len(classes_data.get('classes', []))} different object types")
                print(f"   First 10 classes: {classes_data.get('classes', [])[:10]}")
            
            return True
        except Exception as e:
            print(f"❌ Model info test failed: {e}")
            return False
    
    def test_yolov8x_accuracy(self, image_path: str):
        """Test YOLOv8x detection accuracy with different confidence levels."""
        print(f"\n🎯 Testing YOLOv8x accuracy on {image_path}...")
        
        confidence_levels = [0.1, 0.25, 0.5, 0.7]
        results = {}
        
        for confidence in confidence_levels:
            print(f"   Testing with confidence threshold: {confidence}")
            try:
                with open(image_path, 'rb') as f:
                    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                    response = self.session.post(
                        f"{self.base_url}/detect?confidence={confidence}", 
                        files=files
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    results[confidence] = {
                        'count': result['count'],
                        'objects': [obj['label'] for obj in result['objects']],
                        'avg_confidence': sum([obj['confidence'] for obj in result['objects']]) / len(result['objects']) if result['objects'] else 0
                    }
                    print(f"      Objects found: {result['count']}, Avg confidence: {results[confidence]['avg_confidence']:.3f}")
                else:
                    print(f"      ❌ Failed with status: {response.status_code}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Display summary
        print(f"\n📊 YOLOv8x Detection Summary for {os.path.basename(image_path)}:")
        for conf, data in results.items():
            unique_objects = set(data['objects'])
            print(f"   Confidence {conf}: {data['count']} objects, {len(unique_objects)} unique types")
        
        return results
    
    def test_specific_objects(self, image_path: str, target_objects: list):
        """Test detection of specific objects with YOLOv8x."""
        print(f"\n🎯 Testing specific object detection on {image_path}...")
        
        for obj_type in target_objects:
            print(f"   Looking for: {obj_type}")
            try:
                with open(image_path, 'rb') as f:
                    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                    response = self.session.post(
                        f"{self.base_url}/detect/{obj_type}?confidence=0.25", 
                        files=files
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['count'] > 0:
                        avg_conf = sum([obj['confidence'] for obj in result['objects']]) / len(result['objects'])
                        print(f"      ✅ Found {result['count']} {obj_type}(s), avg confidence: {avg_conf:.3f}")
                    else:
                        print(f"      ❌ No {obj_type} detected")
                else:
                    print(f"      ❌ Failed with status: {response.status_code}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    def performance_test(self, image_path: str, num_requests: int = 5):
        """Test YOLOv8x performance (note: x model is slower but more accurate)."""
        print(f"\n⚡ Testing YOLOv8x performance ({num_requests} requests)...")
        
        times = []
        for i in range(num_requests):
            start_time = time.time()
            try:
                with open(image_path, 'rb') as f:
                    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                    response = self.session.post(f"{self.base_url}/detect", files=files)
                
                end_time = time.time()
                request_time = end_time - start_time
                times.append(request_time)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   Request {i+1}: {request_time:.2f}s, {result['count']} objects detected")
                else:
                    print(f"   Request {i+1}: Failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   Request {i+1}: Error - {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"\n📈 Performance Summary:")
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Min response time: {min(times):.2f}s")
            print(f"   Max response time: {max(times):.2f}s")
            print(f"   Note: YOLOv8x is the largest model, optimized for accuracy over speed")
    
    def run_comprehensive_yolov8x_test(self):
        """Run comprehensive YOLOv8x tests."""
        print("🚀 Starting YOLOv8x Microservice Test Suite")
        print("=" * 60)
        
        # Test model info
        if not self.test_model_info():
            print("❌ Basic model test failed, stopping")
            return
        
        # Find test images
        test_images_dir = Path(TEST_IMAGES_DIR)
        if not test_images_dir.exists():
            print(f"❌ Test images directory '{TEST_IMAGES_DIR}' not found")
            return
        
        # Get available test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(list(test_images_dir.glob(ext)))
        
        if not test_images:
            print(f"❌ No test images found in '{TEST_IMAGES_DIR}'")
            return
        
        print(f"📁 Found {len(test_images)} test images")
        
        # Test with first available image
        test_image = test_images[0]
        
        # Accuracy test
        self.test_yolov8x_accuracy(str(test_image))
        
        # Specific object detection test
        common_objects = ['person', 'car', 'dog', 'cat', 'bicycle', 'truck']
        self.test_specific_objects(str(test_image), common_objects)
        
        # Performance test
        self.performance_test(str(test_image))
        
        print("\n" + "=" * 60)
        print("🎉 YOLOv8x test suite completed!")
        print("💡 YOLOv8x provides the highest accuracy in the YOLOv8 family")
        print("⚡ For faster inference, consider YOLOv8n or YOLOv8s models")

def main():
    """Main test function."""
    tester = YOLOv8xMicroserviceTest()
    tester.run_comprehensive_yolov8x_test()

if __name__ == "__main__":
    main() 