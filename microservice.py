from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np
from typing import Optional, List, Dict
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import onnxruntime as ort
import onnx

app = FastAPI(
    title="YOLO Object Detection Microservice",
    description="A microservice for object detection using YOLO models",
    version="1.0.0"
)

class YOLODetectorService:
    def __init__(self, model_path: str = "yolov8n.pt", use_onnx: bool = True):
        """
        Initialize YOLO detector service.
        
        Args:
            model_path (str): Path to the YOLO model file
            use_onnx (bool): Whether to use ONNX format for better performance
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.model = None
        self.onnx_session = None
        self.class_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model and optionally convert to ONNX."""
        # Load PyTorch model
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        
        if self.use_onnx:
            onnx_path = self.model_path.replace('.pt', '.onnx')
            
            # Convert to ONNX if not exists
            if not os.path.exists(onnx_path):
                print(f"Converting {self.model_path} to ONNX format...")
                self.model.export(format='onnx', optimize=True)
                print(f"ONNX model saved to: {onnx_path}")
            
            # Load ONNX model for inference
            try:
                self.onnx_session = ort.InferenceSession(onnx_path)
                print(f"ONNX model loaded successfully: {onnx_path}")
            except Exception as e:
                print(f"Failed to load ONNX model, falling back to PyTorch: {e}")
                self.use_onnx = False
    
    def detect_objects(self, image: Image.Image, target_label: Optional[str] = None, confidence_threshold: float = 0.5) -> Dict:
        """
        Perform object detection on an image.
        
        Args:
            image (PIL.Image): Input image
            target_label (str, optional): Filter results by this label
            confidence_threshold (float): Minimum confidence threshold
        
        Returns:
            dict: Detection results with image, objects, and count
        """
        # Convert PIL image to format suitable for YOLO
        img_array = np.array(image)
        
        # Perform inference
        if self.use_onnx and self.onnx_session:
            results = self._onnx_inference(img_array)
        else:
            results = self.model(img_array)
        
        # Process results
        detections = []
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y = int(x1), int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Skip if confidence is below threshold
                    if confidence < confidence_threshold:
                        continue
                    
                    label = self.class_names[class_id]
                    
                    # Add detection if it matches target label or no target specified
                    if target_label is None or label.lower() == target_label.lower():
                        detection = {
                            "label": label,
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "confidence": round(confidence, 2)
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        self._draw_detection(draw, detection, font)
        
        # Convert result image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Count objects for the specific label if provided
        count = len(detections)
        
        return {
            "image": result_base64,
            "objects": detections,
            "count": count
        }
    
    def _onnx_inference(self, img_array: np.ndarray):
        """Perform inference using ONNX model (placeholder for actual implementation)."""
        # For now, fallback to PyTorch model
        # In a production environment, you would implement proper ONNX preprocessing and inference
        return self.model(img_array)
    
    def _draw_detection(self, draw: ImageDraw.Draw, detection: Dict, font):
        """Draw bounding box and label on the image."""
        x, y, width, height = detection["x"], detection["y"], detection["width"], detection["height"]
        label = detection["label"]
        confidence = detection["confidence"]
        
        # Draw rectangle
        draw.rectangle(
            [(x, y), (x + width, y + height)],
            outline="red",
            width=2
        )
        
        # Draw label with confidence
        label_text = f"{label} {confidence:.2f}"
        
        # Get text size for background
        try:
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width, text_height = 100, 20  # fallback
        
        # Draw background for text
        draw.rectangle(
            [(x, y - text_height - 2), (x + text_width + 4, y)],
            fill="red"
        )
        
        # Draw text
        draw.text(
            (x + 2, y - text_height - 2),
            label_text,
            fill="white",
            font=font
        )

# Initialize the detector service with YOLOv8x for highest accuracy
detector = YOLODetectorService(model_path="yolov8x.pt", use_onnx=True)

# Thread pool for handling concurrent requests
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "YOLO Object Detection Microservice is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "onnx_enabled": detector.use_onnx,
        "available_classes": len(detector.class_names) if detector.class_names else 0
    }

@app.post("/detect")
async def detect_objects_all(
    file: UploadFile = File(...),
    confidence: float = 0.1
):
    """
    Detect all objects in an uploaded image.
    
    Args:
        file: Image file to process (JPEG, PNG, WEBP)
        confidence: Confidence threshold (0.0-1.0, default: 0.1)
    
    Returns:
        JSON response with detection results:
        {
            "image": "base64_encoded_result_image_with_bounding_boxes",
            "objects": [
                {
                    "label": "person",
                    "x": 12,
                    "y": 453,
                    "width": 10,
                    "height": 40,
                    "confidence": 0.6
                }
            ],
            "count": 2
        }
    """
    return await _process_detection(file, None, confidence)

@app.post("/detect/{label}")
async def detect_objects_filtered(
    label: str = Path(..., description="Object label to filter (e.g., 'person', 'car')"),
    file: UploadFile = File(...),
    confidence: float = 0.1
):
    """
    Detect specific objects in an uploaded image.
    
    Args:
        label: Object label to filter results (e.g., 'person', 'car', 'dog')
        file: Image file to process (JPEG, PNG, WEBP)
        confidence: Confidence threshold (0.0-1.0, default: 0.1)
    
    Returns:
        JSON response with detection results filtered by label:
        {
            "image": "base64_encoded_result_image_with_bounding_boxes",
            "objects": [
                {
                    "label": "person",
                    "x": 12,
                    "y": 453,
                    "width": 10,
                    "height": 40,
                    "confidence": 0.6
                }
            ],
            "count": 2
        }
        
        Note: Only objects matching the specified label are returned.
        The count reflects the number of objects with the given label.
    """
    return await _process_detection(file, label, confidence)

async def _process_detection(file: UploadFile, target_label: Optional[str], confidence: float):
    """
    Process detection request asynchronously.
    
    Args:
        file: Uploaded image file
        target_label: Optional label to filter results
        confidence: Confidence threshold
    
    Returns:
        JSON response with detection results
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence threshold
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            detector.detect_objects, 
            image, 
            target_label, 
            confidence
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/classes")
async def get_available_classes():
    """Get list of available object classes."""
    if detector.class_names:
        return {
            "classes": list(detector.class_names.values()),
            "total_classes": len(detector.class_names)
        }
    else:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

if __name__ == "__main__":
    uvicorn.run(
        "microservice:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    ) 