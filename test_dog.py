import requests
import os

# Test dog image
image_path = 'test_images/sample6_dog.jpg'
print(f'🐕 Testing YOLOv8x with {image_path}...')

# General detection
with open(image_path, 'rb') as f:
    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/detect?confidence=0.25', files=files, timeout=30)

if response.status_code == 200:
    result = response.json()
    print(f'✅ Detection successful!')
    print(f'   Objects detected: {result["count"]}')
    
    for i, obj in enumerate(result['objects']):
        print(f'   {i+1}. {obj["label"]} (confidence: {obj["confidence"]:.1%})')
        print(f'      Location: x={obj["x"]}, y={obj["y"]}, w={obj["width"]}, h={obj["height"]}')

# Dog-specific detection
print(f'\n🐕 Testing dog-specific detection...')
with open(image_path, 'rb') as f:
    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/detect/dog?confidence=0.25', files=files)

if response.status_code == 200:
    result = response.json()
    print(f'✅ Dog detection: {result["count"]} dogs found')
    for i, dog in enumerate(result['objects']):
        print(f'   Dog {i+1}: {dog["confidence"]:.1%} confidence')
else:
    print(f'❌ Dog detection failed: {response.status_code}') 