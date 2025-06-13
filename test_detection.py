import requests
import json
import base64
from datetime import datetime

def test_detection():
    # Görüntüyü dosya olarak gönder - tüm nesneler için confidence 0.1
    with open("test_images/sample9_animals.jpg", "rb") as image_file:
        files = {"file": ("sample9_animals.jpg", image_file, "image/jpeg")}
        # Çok düşük confidence ile tüm nesneleri ara
        params = {"confidence": 0.1}
        response = requests.post("http://localhost:8000/detect", files=files, params=params)
    
    # Sonucu yazdır
    print("Status Code:", response.status_code)
    
    if response.status_code == 200:
        result = response.json()
        
        # Tüm JSON yanıtını yazdır (image hariç)
        print("\n=== API Yanıtı (Tüm Nesneler) ===")
        response_copy = result.copy()
        if "image" in response_copy:
            response_copy["image"] = "[BASE64_IMAGE_DATA - GÖSTERILMIYOR]"
        print(json.dumps(response_copy, indent=2, ensure_ascii=False))
        
        # Tespit edilen nesneleri detaylı yazdır
        if "objects" in result:
            print("\n=== Tespit Edilen Tüm Nesneler ===")
            for i, det in enumerate(result["objects"]):
                print(f"{i+1}. {det['label']}")
                print(f"   - Confidence: {det['confidence']:.3f}")
                print(f"   - Konum: x={det['x']}, y={det['y']}, width={det['width']}, height={det['height']}")
        
        # İstatistikleri yazdır
        if "count" in result:
            print(f"\nToplam nesne sayısı: {result['count']}")
        
        # Sonuç görüntüsünü kaydet
        if "image" in result:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_images/result_all_{timestamp}.jpg"
            
            # Base64 görüntüyü decode et ve kaydet
            image_data = base64.b64decode(result["image"])
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"\nTüm nesneler sonuç görüntüsü kaydedildi: {output_path}")
    else:
        print("Hata:", response.text)

if __name__ == "__main__":
    test_detection() 