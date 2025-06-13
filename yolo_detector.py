from ultralytics import YOLO
import base64
from PIL import Image
import io
import json
from typing import List, Dict, Optional, Union
import numpy as np

class YOLODetector:
    def __init__(self, model_size="s"):
        """
        Initialize YOLO detector with specified model size.
        
        Args:
            model_size (str): Model size to use ('s', 'm', or 'l')
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model based on the specified size."""
        model_name = f"yolov8{self.model_size}.pt"
        self.model = YOLO(model_name)
    
    def detect(self, image_base64, target_label=None):
        """
        Perform object detection on a base64 encoded image.
        
        Args:
            image_base64 (str): Base64 encoded image string
            target_label (str, optional): If specified, only return detections of this label
            
        Returns:
            dict: Detection results including:
                - image: Base64 encoded image with bounding boxes
                - objects: List of detected objects with their properties
                - count: Number of objects detected
        """
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Perform detection
        results = self.model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                label = result.names[class_id]
                
                # Add detection if it matches target label or no target specified
                if target_label is None or label == target_label:
                    detections.append({
                        "label": label,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "confidence": confidence
                    })
        
        # Draw bounding boxes on image
        result_image = image.copy()
        for det in detections:
            # Draw rectangle
            result_image = self._draw_box(result_image, det)
        
        # Convert result image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": result_base64,
            "objects": detections,
            "count": len(detections)
        }
    
    def _draw_box(self, image, detection):
        """Draw bounding box and label on the image."""
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(image)
        
        # Draw rectangle
        draw.rectangle(
            [(detection["x"], detection["y"]), 
             (detection["x"] + detection["width"], 
              detection["y"] + detection["height"])],
            outline="red",
            width=2
        )
        
        # Draw label with confidence
        label = f"{detection['label']} {detection['confidence']:.2f}"
        draw.text(
            (detection["x"], detection["y"] - 10),
            label,
            fill="red"
        )
        
        return image 