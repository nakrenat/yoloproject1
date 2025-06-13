from yolo_detector import YOLODetector
import base64
from PIL import Image
import io
import os
import json

def test_detector():
    # Create test images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Please place some test images in the 'test_images' directory")
        return

    # Get list of images in test_images directory
    image_files = [f for f in os.listdir('test_images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in test_images directory")
        return

    # Initialize detector with small model
    print("Initializing YOLO detector with small model...")
    detector = YOLODetector(model_size="s")
    
    # Test each image
    for image_file in image_files:
        print(f"\nTesting image: {image_file}")
        image_path = os.path.join('test_images', image_file)
        
        # Load and convert image to base64
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode()
        
        # Test detection without label filter
        print("\nTesting detection without label filter:")
        result = detector.detect(image_base64)
        print(f"Number of objects detected: {result['count']}")
        print("Detected objects:")
        for obj in result['objects']:
            print(f"- {obj['label']}: confidence={obj['confidence']:.2f}, "
                  f"position=({obj['x']}, {obj['y']}), "
                  f"size={obj['width']}x{obj['height']}")
        
        # Save the result image
        result_image_bytes = base64.b64decode(result['image'])
        result_image = Image.open(io.BytesIO(result_image_bytes))
        result_path = os.path.join('test_images', f'result_{image_file}')
        result_image.save(result_path)
        print(f"Result image saved as: {result_path}")
        
        # Test detection with a specific label (e.g., 'person')
        print("\nTesting detection with label filter (person):")
        filtered_result = detector.detect(image_base64, target_label="person")
        print(f"Number of persons detected: {filtered_result['count']}")
        print("Detected persons:")
        for obj in filtered_result['objects']:
            print(f"- confidence={obj['confidence']:.2f}, "
                  f"position=({obj['x']}, {obj['y']}), "
                  f"size={obj['width']}x{obj['height']}")

if __name__ == "__main__":
    test_detector() 