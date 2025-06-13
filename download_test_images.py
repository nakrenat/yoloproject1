import os
import requests
from PIL import Image
from io import BytesIO

def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        print(f"Successfully downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

def main():
    # Create test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')

    # Sample images with different objects
    sample_images = [
        {
            'url': 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg',
            'name': 'sample1_people.jpg',
            'description': 'Image with multiple people'
        },
        {
            'url': 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg',
            'name': 'sample2_bus.jpg',
            'description': 'Image with a bus and people'
        },
        {
            'url': 'https://ultralytics.com/images/bus.jpg',
            'name': 'sample3_traffic.jpg',
            'description': 'Image with traffic scene'
        },
        {
            'url': 'https://ultralytics.com/images/coco.jpg',
            'name': 'sample4_coco.jpg',
            'description': 'COCO dataset sample (multiple objects)'
        },
        {
            'url': 'https://ultralytics.com/images/zidane.jpg',
            'name': 'sample5_zidane.jpg',
            'description': 'Another image with people (Zidane)'
        },
        {
            'url': 'https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg',
            'name': 'sample6_dog.jpg',
            'description': 'Dog in the park'
        },
        {
            'url': 'https://images.pexels.com/photos/1402787/pexels-photo-1402787.jpeg',
            'name': 'sample7_cars.jpg',
            'description': 'Street with cars'
        },
        {
            'url': 'https://images.pexels.com/photos/459225/pexels-photo-459225.jpeg',
            'name': 'sample8_bike.jpg',
            'description': 'Person riding a bike'
        },
        {
            'url': 'https://images.pexels.com/photos/45170/kittens-cat-cat-puppy-rush-45170.jpeg',
            'name': 'sample9_animals.jpg',
            'description': 'Kittens and puppy'
        },
        {
            'url': 'https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg',
            'name': 'sample10_street.jpg',
            'description': 'Busy street scene'
        },
        {
            'url': 'https://images.pexels.com/photos/3802510/pexels-photo-3802510.jpeg',
            'name': 'sample11_traffic_jam.jpg',
            'description': 'Complex traffic jam with multiple cars'
        }
    ]

    print("Downloading sample images...")
    for img in sample_images:
        save_path = os.path.join('test_images', img['name'])
        print(f"\nDownloading {img['description']}...")
        download_image(img['url'], save_path)

    print("\nAll sample images have been downloaded to the 'test_images' directory.")
    print("You can now run the test script with:")
    print("python test_detector.py")

if __name__ == "__main__":
    main() 