import cv2
import numpy as np
import os

def test_video_writer():
    width, height, fps = 640, 480, 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'test_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ VideoWriter açılamadı!")
        return False

    # 100 frame yaz
    for i in range(100):
        # Renkli bir arkaplan oluştur (daha iyi görünürlük için)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Koyu gri arkaplan
        
        # Ortada beyaz bir dikdörtgen çiz
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 255, 255), -1)
        
        # Frame numarasını yaz
        cv2.putText(frame, f'Frame {i}', (width//3, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        out.write(frame)
        
        if i % 20 == 0:  # Her 20 frame'de bir ilerleme göster
            print(f"⏳ Frame {i}/100 yazıldı...")

    out.release()
    
    # Dosya kontrolü
    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"\n✅ Video dosyası oluşturuldu: {output_path}")
        print(f"📦 Dosya boyutu: {size_kb:.1f} KB")
        return size_kb > 100  # En az 100 KB olmalı
    else:
        print("❌ Video dosyası oluşturulamadı!")
        return False

if __name__ == "__main__":
    print("🎥 Video yazma testi başlıyor...")
    success = test_video_writer()
    if success:
        print("✅ Test başarılı! Video dosyası oluşturuldu ve yeterli boyutta.")
    else:
        print("❌ Test başarısız! Video dosyası oluşturulamadı veya çok küçük.") 