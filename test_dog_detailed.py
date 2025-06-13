import requests
import os

image_path = 'test_images/sample6_dog.jpg'
print(f'🔍 Detailed YOLOv8x analysis of {image_path}...')

# Test with different confidence levels
confidence_levels = [0.1, 0.3, 0.5, 0.7]

for conf in confidence_levels:
    print(f'\n📊 Testing with {conf:.0%} confidence threshold:')
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f'http://localhost:8000/detect?confidence={conf}', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f'   Objects found: {result["count"]}')
        
        # Group by object type
        objects_by_type = {}
        for obj in result['objects']:
            obj_type = obj['label']
            if obj_type not in objects_by_type:
                objects_by_type[obj_type] = []
            objects_by_type[obj_type].append(obj['confidence'])
        
        for obj_type, confidences in objects_by_type.items():
            avg_conf = sum(confidences) / len(confidences)
            print(f'   - {len(confidences)} {obj_type}(s): avg {avg_conf:.1%} confidence')

# Test specific animals
print(f'\n🐾 Testing specific animal detection:')
animals = ['dog', 'bear', 'cat', 'wolf']

for animal in animals:
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f'http://localhost:8000/detect/{animal}?confidence=0.1', files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result['count'] > 0:
            avg_conf = sum([obj['confidence'] for obj in result['objects']]) / len(result['objects'])
            print(f'   ✅ {animal}: {result["count"]} found (avg {avg_conf:.1%})')
        else:
            print(f'   ❌ {animal}: none detected')
    else:
        print(f'   ❓ {animal}: test failed') 