#!/usr/bin/env python3
"""
Test script for YOLO Object Detection Microservice

This script tests various endpoints of the microservice with different scenarios:
1. Health check
2. Object detection without label filter
3. Object detection with specific label filters
4. Error handling (invalid files, parameters)
"""

import requests
import base64
import json
import os
import time
from pathlib import Path
import argparse

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGES_DIR = "test_images"

class MicroserviceTest:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_health_check(self):
        """Test the health check endpoint."""
        print("🏥 Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            assert response.status_code == 200
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        print("🏠 Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            assert response.status_code == 200
            root_data = response.json()
            print(f"✅ Root endpoint passed: {root_data}")
            return True
        except Exception as e:
            print(f"❌ Root endpoint failed: {e}")
            return False
    
    def test_get_classes(self):
        """Test the get classes endpoint."""
        print("📋 Testing get classes endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/classes")
            assert response.status_code == 200
            classes_data = response.json()
            print(f"✅ Get classes passed. Available classes: {len(classes_data.get('classes', []))}")
            print(f"   Classes: {classes_data.get('classes', [])[:10]}...")  # Show first 10
            return True
        except Exception as e:
            print(f"❌ Get classes failed: {e}")
            return False
    
    def test_detect_all_objects(self, image_path: str):
        """Test object detection without label filter."""
        print(f"🔍 Testing detection on {image_path} (all objects)...")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/detect", files=files)
            
            assert response.status_code == 200
            result = response.json()
            
            print(f"✅ Detection successful:")
            print(f"   Objects detected: {result['count']}")
            print(f"   Object types: {set([obj['label'] for obj in result['objects']])}")
            
            # Validate response structure
            assert 'image' in result
            assert 'objects' in result
            assert 'count' in result
            assert isinstance(result['objects'], list)
            assert result['count'] == len(result['objects'])
            
            # Save result for inspection
            self.save_result(image_path, result, "all_objects")
            
            return True, result
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return False, None
    
    def test_detect_specific_label(self, image_path: str, label: str):
        """Test object detection with specific label filter."""
        print(f"🎯 Testing detection on {image_path} (label: {label})...")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = self.session.post(f"{self.base_url}/detect/{label}", files=files)
            
            assert response.status_code == 200
            result = response.json()
            
            print(f"✅ Detection successful:")
            print(f"   Objects with label '{label}': {result['count']}")
            
            # Validate that all objects have the correct label
            for obj in result['objects']:
                assert obj['label'].lower() == label.lower(), f"Expected {label}, got {obj['label']}"
            
            # Validate response structure
            assert 'image' in result
            assert 'objects' in result
            assert 'count' in result
            assert result['count'] == len(result['objects'])
            
            # Save result for inspection
            self.save_result(image_path, result, f"label_{label}")
            
            return True, result
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return False, None
    
    def test_invalid_file(self):
        """Test error handling with invalid file."""
        print("⚠️ Testing error handling with invalid file...")
        try:
            # Create a text file and try to send it
            files = {'file': ('test.txt', b'This is not an image', 'text/plain')}
            response = self.session.post(f"{self.base_url}/detect", files=files)
            
            assert response.status_code == 400
            print("✅ Invalid file handling passed")
            return True
        except Exception as e:
            print(f"❌ Invalid file handling failed: {e}")
            return False
    
    def test_invalid_confidence(self, image_path: str):
        """Test error handling with invalid confidence value."""
        print("⚠️ Testing error handling with invalid confidence...")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                # Test confidence > 1.0
                response = self.session.post(f"{self.base_url}/detect?confidence=1.5", files=files)
            
            assert response.status_code == 400
            print("✅ Invalid confidence handling passed")
            return True
        except Exception as e:
            print(f"❌ Invalid confidence handling failed: {e}")
            return False
    
    def save_result(self, image_path: str, result: dict, test_type: str):
        """Save test result for inspection."""
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detection result as JSON
        result_file = results_dir / f"{Path(image_path).stem}_{test_type}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save result image if needed
        if 'image' in result:
            image_data = base64.b64decode(result['image'])
            image_file = results_dir / f"{Path(image_path).stem}_{test_type}_result.png"
            with open(image_file, 'wb') as f:
                f.write(image_data)
            print(f"   💾 Results saved to {result_file} and {image_file}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite."""
        print("🚀 Starting comprehensive microservice tests...\n")
        
        # Test basic endpoints
        tests_passed = 0
        total_tests = 0
        
        # Basic endpoint tests
        basic_tests = [
            self.test_health_check,
            self.test_root_endpoint,
            self.test_get_classes
        ]
        
        for test in basic_tests:
            total_tests += 1
            if test():
                tests_passed += 1
            print()
        
        # Find test images
        test_images = []
        if os.path.exists(TEST_IMAGES_DIR):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(Path(TEST_IMAGES_DIR).glob(ext))
        
        if not test_images:
            print("⚠️ No test images found in test_images directory")
            return
        
        # Select a few representative test images
        selected_images = []
        for pattern in ['people', 'car', 'dog', 'bus', 'traffic']:
            for img in test_images:
                if pattern in img.name.lower() and img not in selected_images:
                    selected_images.append(img)
                    break
        
        # If no specific patterns found, use first few images
        if not selected_images:
            selected_images = test_images[:3]
        
        print(f"📸 Testing with {len(selected_images)} images...")
        
        # Test scenarios
        test_scenarios = [
            # (image_pattern, label, description)
            ('people', 'person', 'person detection'),
            ('car', 'car', 'car detection'),
            ('dog', 'dog', 'dog detection'),
            ('bus', 'bus', 'bus detection'),
            ('traffic', 'traffic light', 'traffic light detection'),
        ]
        
        for image_path in selected_images:
            # Test detection without filter
            total_tests += 1
            success, _ = self.test_detect_all_objects(str(image_path))
            if success:
                tests_passed += 1
            print()
            
            # Test detection with specific labels based on image name
            for pattern, label, description in test_scenarios:
                if pattern in image_path.name.lower():
                    total_tests += 1
                    success, _ = self.test_detect_specific_label(str(image_path), label)
                    if success:
                        tests_passed += 1
                    print()
                    break
        
        # Test error handling
        if selected_images:
            total_tests += 2
            if self.test_invalid_file():
                tests_passed += 1
            print()
            
            if self.test_invalid_confidence(str(selected_images[0])):
                tests_passed += 1
            print()
        
        # Summary
        print("=" * 60)
        print(f"🏁 Test Summary: {tests_passed}/{total_tests} tests passed")
        print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! Microservice is working correctly.")
        else:
            print("❌ Some tests failed. Check the output above for details.")
        
        return tests_passed == total_tests

def main():
    parser = argparse.ArgumentParser(description='Test YOLO Object Detection Microservice')
    parser.add_argument('--url', default=BASE_URL, help='Base URL of the microservice')
    parser.add_argument('--image', help='Path to a specific image to test')
    parser.add_argument('--label', help='Specific label to test (requires --image)')
    
    args = parser.parse_args()
    
    tester = MicroserviceTest(args.url)
    
    # Test health first
    if not tester.test_health_check():
        print("❌ Service not healthy, aborting tests")
        return
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        if args.label:
            success, result = tester.test_detect_specific_label(args.image, args.label)
        else:
            success, result = tester.test_detect_all_objects(args.image)
        
        if success:
            print(f"✅ Test completed successfully")
            print(f"📊 Detection summary: {result['count']} objects detected")

if __name__ == "__main__":
    main() 