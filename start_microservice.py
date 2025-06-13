#!/usr/bin/env python3
"""
Startup script for YOLO Object Detection Microservice
This script helps start the microservice and run basic tests
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'ultralytics', 'onnx', 'onnxruntime'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements-microservice.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model_file():
    """Check if YOLO model file exists."""
    model_files = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model file: {model_file}")
            return model_file
    
    print("❌ No YOLO model file found")
    print("Download with: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    return None

def start_microservice():
    """Start the microservice."""
    print("🚀 Starting YOLO Object Detection Microservice...")
    
    try:
        # Start the microservice
        process = subprocess.Popen([
            sys.executable, 'microservice.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ Microservice started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start microservice:")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting microservice: {e}")
        return None

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_basic_detection():
    """Test basic object detection."""
    test_images_dir = Path("test_images")
    
    if not test_images_dir.exists():
        print("⚠️ test_images directory not found, skipping detection test")
        return False
    
    # Find a test image
    test_image = None
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images = list(test_images_dir.glob(ext))
        if images:
            test_image = images[0]
            break
    
    if not test_image:
        print("⚠️ No test images found, skipping detection test")
        return False
    
    print(f"🔍 Testing detection with: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            response = requests.post(
                "http://localhost:8000/detect", 
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection test passed")
            print(f"   Objects detected: {result['count']}")
            print(f"   Object types: {set([obj['label'] for obj in result['objects']])}")
            return True
        else:
            print(f"❌ Detection test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def main():
    """Main startup function."""
    print("=" * 60)
    print("🎯 YOLO Object Detection Microservice Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    model_file = check_model_file()
    if not model_file:
        sys.exit(1)
    
    # Start microservice
    process = start_microservice()
    if not process:
        sys.exit(1)
    
    try:
        # Test health
        print("\n🏥 Testing health endpoint...")
        health_ok = test_health()
        
        if health_ok:
            # Test detection
            print("\n🔍 Testing object detection...")
            detection_ok = test_basic_detection()
            
            print("\n" + "=" * 60)
            if health_ok and detection_ok:
                print("🎉 All tests passed! Microservice is ready.")
                print("\n📚 API Documentation:")
                print("   Swagger UI: http://localhost:8000/docs")
                print("   ReDoc: http://localhost:8000/redoc")
                print("\n💡 Example usage:")
                print('   curl -X POST "http://localhost:8000/detect" \\')
                print('     -H "Content-Type: multipart/form-data" \\')
                print('     -F "file=@test_images/sample1_people.jpg"')
                print("\nPress Ctrl+C to stop the service...")
            else:
                print("⚠️ Some tests failed, but service is running")
            
            # Keep service running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping microservice...")
                process.terminate()
                process.wait()
                print("✅ Microservice stopped")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping microservice...")
        if process:
            process.terminate()
            process.wait()
        print("✅ Microservice stopped")

if __name__ == "__main__":
    main() 