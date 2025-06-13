# Enhanced app.py with Confidence Recommendations & Data Persistence
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import pickle
from pathlib import Path
import subprocess  # Add this at the top with other imports

# Page Configuration
st.set_page_config(
    page_title="🎯 YOLO AI Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ultralytics/ultralytics',
        'Report a bug': None,
        'About': "# YOLO AI Object Detection\nAdvanced AI-powered object detection application!"
    }
)

# Data persistence functions
DATA_DIR = Path("detection_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "detection_history.json"
SETTINGS_FILE = DATA_DIR / "user_settings.json"

def save_detection_history():
    """Save detection history to file"""
    try:
        history_data = {
            'history': st.session_state.detection_history,
            'total_detections': st.session_state.total_detections,
            'processed_files': st.session_state.processed_files,
            'last_updated': datetime.now().isoformat()
        }
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_data, f, default=str, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def load_detection_history():
    """Load detection history from file"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
            
            # Convert timestamp strings back to datetime objects
            for item in data.get('history', []):
                if 'timestamp' in item:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
            
            return data
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def save_user_settings(settings):
    """Save user settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        return False

def load_user_settings():
    """Load user settings from file"""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading settings: {e}")
        return {}

# Confidence threshold recommendations
CONFIDENCE_RECOMMENDATIONS = {
    "person": {"optimal": 0.5, "description": "Good balance for human detection"},
    "car": {"optimal": 0.6, "description": "Higher confidence for vehicle accuracy"},
    "truck": {"optimal": 0.6, "description": "Higher confidence for large vehicles"},
    "bus": {"optimal": 0.6, "description": "Higher confidence for public transport"},
    "motorcycle": {"optimal": 0.4, "description": "Lower threshold for smaller vehicles"},
    "bicycle": {"optimal": 0.4, "description": "Lower threshold for two-wheelers"},
    "traffic light": {"optimal": 0.7, "description": "High confidence for traffic signals"},
    "stop sign": {"optimal": 0.7, "description": "High confidence for road signs"},
    "cat": {"optimal": 0.5, "description": "Standard threshold for pets"},
    "dog": {"optimal": 0.5, "description": "Standard threshold for pets"},
    "bird": {"optimal": 0.4, "description": "Lower threshold for small animals"},
    "general": {"optimal": 0.5, "description": "Recommended for mixed object detection"}
}

def get_confidence_recommendation(selected_classes):
    """Get confidence recommendation based on selected classes"""
    if not selected_classes:
        return CONFIDENCE_RECOMMENDATIONS["general"]
    
    if len(selected_classes) == 1:
        return CONFIDENCE_RECOMMENDATIONS.get(selected_classes[0], CONFIDENCE_RECOMMENDATIONS["general"])
    
    # For multiple classes, find average optimal confidence
    total_confidence = 0
    count = 0
    descriptions = []
    
    for class_name in selected_classes:
        if class_name in CONFIDENCE_RECOMMENDATIONS:
            total_confidence += CONFIDENCE_RECOMMENDATIONS[class_name]["optimal"]
            descriptions.append(f"{class_name}: {CONFIDENCE_RECOMMENDATIONS[class_name]['optimal']}")
            count += 1
    
    if count > 0:
        avg_confidence = total_confidence / count
        return {
            "optimal": round(avg_confidence, 2),
            "description": f"Average for selected classes: {', '.join(descriptions[:3])}"
        }
    
    return CONFIDENCE_RECOMMENDATIONS["general"]

# Initialize session state for dark mode and data
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Load saved data
if 'data_loaded' not in st.session_state:
    saved_data = load_detection_history()
    saved_settings = load_user_settings()
    
    if saved_data:
        st.session_state.detection_history = saved_data.get('history', [])
        st.session_state.total_detections = saved_data.get('total_detections', 0)
        st.session_state.processed_files = saved_data.get('processed_files', 0)
    else:
        st.session_state.detection_history = []
        st.session_state.total_detections = 0
        st.session_state.processed_files = 0
    
    # Load user settings including dark mode
    st.session_state.dark_mode = saved_settings.get('dark_mode', False)
    st.session_state.data_loaded = True

# HANDLE DARK MODE TOGGLE FIRST
# Create a placeholder for the sidebar to handle dark mode toggle early
sidebar_placeholder = st.sidebar.empty()

with sidebar_placeholder.container():
    st.markdown("## 🎨 Theme Settings")
    dark_mode_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
    
    # Handle dark mode change
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        # Save settings immediately
        save_user_settings({'dark_mode': dark_mode_toggle})
        st.rerun()

# Enhanced CSS Styles with Transitions and Dark Mode
def get_css_styles(dark_mode):
    if dark_mode:
        # Dark mode colors
        bg_primary = "#0e1117"
        bg_secondary = "#262730"
        text_primary = "#fafafa"
        text_secondary = "#a0a0a0"
        border_color = "#404040"
        card_bg = "#1e1e1e"
        gradient_primary = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        gradient_secondary = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        stApp_bg = "#0e1117"
    else:
        # Light mode colors
        bg_primary = "#ffffff"
        bg_secondary = "#f0f2f6"
        text_primary = "#262730"
        text_secondary = "#666666"
        border_color = "#e6e9ef"
        card_bg = "#ffffff"
        gradient_primary = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        gradient_secondary = "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)"
        stApp_bg = "#ffffff"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Main app background */
        .stApp {{
            background-color: {bg_primary};
            color: {text_primary};
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {bg_secondary};
            color: {text_primary} !important;
        }}
        
        .css-1cypcdb {{
            background-color: {bg_secondary};
            color: {text_primary} !important;
        }}
        
        /* Sidebar content styling - more specific selectors */
        .css-1d391kg .element-container {{
            color: {text_primary} !important;
        }}
        
        .css-1d391kg h1, 
        .css-1d391kg h2, 
        .css-1d391kg h3, 
        .css-1d391kg h4, 
        .css-1d391kg h5, 
        .css-1d391kg h6 {{
            color: {text_primary} !important;
        }}
        
        .css-1d391kg p, 
        .css-1d391kg span, 
        .css-1d391kg div {{
            color: {text_primary} !important;
        }}
        
        .css-1d391kg .stMarkdown {{
            color: {text_primary} !important;
        }}
        
        .css-1d391kg .stMarkdown h1,
        .css-1d391kg .stMarkdown h2,
        .css-1d391kg .stMarkdown h3,
        .css-1d391kg .stMarkdown h4 {{
            color: {text_primary} !important;
        }}
        
        .css-1d391kg .stSelectbox label,
        .css-1d391kg .stSlider label,
        .css-1d391kg .stMultiselect label,
        .css-1d391kg .stColorPicker label {{
            color: {text_primary} !important;
        }}
        
        /* Section within sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {bg_secondary} !important;
            color: {text_primary} !important;
        }}
        
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6 {{
            color: {text_primary} !important;
        }}
        
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label {{
            color: {text_primary} !important;
        }}
        
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {text_primary} !important;
        }}
        
        /* Text elements */
        .css-10trblm {{
            color: {text_primary};
        }}
        
        * {{
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }}
        
        /* Force sidebar text color in dark mode */
        .stSidebar {{
            background-color: {bg_secondary} !important;
        }}
        
        .stSidebar * {{
            color: {text_primary} !important;
        }}
        
        /* Additional sidebar selectors for different Streamlit versions */
        .css-6qob1r {{
            background-color: {bg_secondary} !important;
            color: {text_primary} !important;
        }}
        
        .css-6qob1r * {{
            color: {text_primary} !important;
        }}
        
        .css-17lntkn {{
            background-color: {bg_secondary} !important;
            color: {text_primary} !important;
        }}
        
        .css-17lntkn * {{
            color: {text_primary} !important;
        }}
        
        /* Sidebar form elements */
        .css-1d391kg .stSelectbox > div > div {{
            color: {text_primary} !important;
        }}
        
        .css-1d391kg .stSlider > div > div > div {{
            color: {text_primary} !important;
        }}
        
        /* All sidebar text elements */
        [data-testid="stSidebar"] {{
            background-color: {bg_secondary} !important;
        }}
        
        [data-testid="stSidebar"] * {{
            color: {text_primary} !important;
        }}
        
        [data-testid="stSidebar"] .element-container {{
            color: {text_primary} !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown {{
            color: {text_primary} !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown * {{
            color: {text_primary} !important;
        }}
        
        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton > button {{
            background: {gradient_primary} !important;
            color: white !important;
            border: none !important;
        }}
        
        /* Widget labels in sidebar */
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stMultiselect label,
        [data-testid="stSidebar"] .stColorPicker label,
        [data-testid="stSidebar"] .stButton label {{
            color: {text_primary} !important;
        }}
        
        .main-header {{
            background: {gradient_primary};
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
            animation: slideInDown 0.8s ease-out;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, #FFA726 0%, #FF7043 100%);
            border: none;
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: white;
            box-shadow: 0 3px 10px rgba(255, 167, 38, 0.3);
            font-size: 0.9rem;
        }}
        
        .data-management-section {{
            background: {card_bg};
            border: 2px solid #4CAF50;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.2);
        }}
        
        .history-stats {{
            background: {gradient_primary};
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: white;
            text-align: center;
        }}
        
        .metric-card {{
            background: {card_bg};
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid {border_color};
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            transform: translateY(0);
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }}
        
        .stTab [data-baseweb="tab-list"] {{
            gap: 20px;
            background: transparent;
            border-bottom: none;
        }}
        
        .stTab [data-baseweb="tab"] {{
            height: 60px;
            padding: 0px 30px;
            background: {card_bg};
            border-radius: 15px;
            border: 2px solid {border_color};
            color: {text_primary};
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .stTab [data-baseweb="tab"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }}
        
        .stTab [aria-selected="true"] {{
            background: {gradient_primary};
            color: white;
            border-color: transparent;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .upload-section {{
            border: 3px dashed #667eea;
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            background: {gradient_secondary};
            color: {text_primary};
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .upload-section:hover {{
            border-color: #764ba2;
            transform: scale(1.02);
        }}
        
        .success-box {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            border: none;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: white;
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            animation: slideInRight 0.5s ease-out;
        }}
        
        .info-box {{
            background: {gradient_primary};
            border: none;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            animation: slideInLeft 0.5s ease-out;
        }}
        
        .stButton > button {{
            background: {gradient_primary};
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }}
        
        /* Sidebar specific styling */
        .sidebar-section {{
            background: {card_bg};
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid {border_color};
        }}
        
        /* Animations */
        @keyframes slideInDown {{
            from {{
                opacity: 0;
                transform: translateY(-50px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes slideInRight {{
            from {{
                opacity: 0;
                transform: translateX(50px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-50px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                transform: scale(1);
            }}
            50% {{
                transform: scale(1.05);
            }}
        }}
        
        .fade-in {{
            animation: slideInDown 0.6s ease-out;
        }}
    </style>
    """

# NOW Apply CSS styles with current dark mode state
st.markdown(get_css_styles(st.session_state.dark_mode), unsafe_allow_html=True)

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1>🎯 YOLO AI Object Detection</h1>
    <p>Advanced AI-powered real-time object recognition and analysis</p>
</div>
""", unsafe_allow_html=True)

# Model Loading
@st.cache_resource
def load_model(model_size):
    return YOLO(f'yolov8{model_size}.pt')

# Initialize model selection in session state
if 'selected_model' not in st.session_state:
    saved_settings = load_user_settings()
    st.session_state.selected_model = saved_settings.get('selected_model', 'n')

model = load_model(st.session_state.selected_model)

# Clear the sidebar placeholder and rebuild the full sidebar
sidebar_placeholder.empty()

# Enhanced Sidebar with Data Management
with st.sidebar:
    # Theme Settings (rebuild with full content)
    st.markdown("## 🎨 Theme Settings")
    # Display current mode status (read-only display)
    mode_status = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
    st.markdown(f"**Current Mode:** {mode_status}")
    
    # Toggle button for mode switching
    if st.button("🔄 Toggle Theme", help="Switch between light and dark mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        save_user_settings({'dark_mode': st.session_state.dark_mode})
        st.rerun()
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Advanced Settings")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Settings with Recommendations
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Configuration")
    
    # Model Size Selection
    st.markdown("#### 🧠 Model Size")
    model_options = {
        'n': 'YOLOv8n (Nano) - Fastest',
        's': 'YOLOv8s (Small) - Fast',
        'm': 'YOLOv8m (Medium) - Balanced',
        'l': 'YOLOv8l (Large) - Accurate',
        'x': 'YOLOv8x (Extra Large) - Most Accurate'
    }
    
    selected_model = st.selectbox(
        "🎯 Select Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.selected_model),
        help="Larger models are more accurate but slower"
    )
    
    # Handle model change
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Update user settings
        current_settings = load_user_settings()
        current_settings['selected_model'] = selected_model
        save_user_settings(current_settings)
        # Clear cache and reload model
        st.cache_resource.clear()
        st.rerun()
    
    # Model info
    model_info = {
        'n': {'params': '3.2M', 'size': '6.4MB', 'speed': '~80ms'},
        's': {'params': '11.2M', 'size': '22MB', 'speed': '~90ms'},
        'm': {'params': '25.9M', 'size': '52MB', 'speed': '~110ms'},
        'l': {'params': '43.7M', 'size': '88MB', 'speed': '~140ms'},
        'x': {'params': '68.2M', 'size': '136MB', 'speed': '~180ms'}
    }
    
    current_info = model_info[st.session_state.selected_model]
    st.markdown(f"""
    <div class="recommendation-box">
        <strong>📊 Model Stats:</strong><br>
        Parameters: {current_info['params']}<br>
        Size: {current_info['size']}<br>
        Speed: {current_info['speed']}
    </div>
    """, unsafe_allow_html=True)
    
    # Class Filter (moved up to use for recommendations)
    st.markdown("#### 🔍 Class Filter")
    available_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                        'traffic light', 'stop sign', 'cat', 'dog', 'bird']
    selected_classes = st.multiselect("🏷️ Classes to Detect", 
                                     available_classes, 
                                     default=available_classes)
    
    # Get confidence recommendation
    recommendation = get_confidence_recommendation(selected_classes)
    
    st.markdown("#### 🎯 Confidence Settings")
    
    # Display recommendation
    st.markdown(f"""
    <div class="recommendation-box">
        <strong>💡 Recommended: {recommendation['optimal']}</strong><br>
        {recommendation['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # Quick apply buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Apply Rec.", help="Apply recommended confidence"):
            confidence = recommendation['optimal']
    with col2:
        if st.button("🔒 High Prec.", help="High precision (0.7)"):
            confidence = 0.7
    with col3:
        if st.button("🔍 High Sens.", help="High sensitivity (0.3)"):
            confidence = 0.3
    
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 
                          recommendation['optimal'], 0.01, 
                          help="Lower value: More detections, Higher value: More precise detections")
    
    iou_threshold = st.slider("🔄 IOU Threshold", 0.0, 1.0, 0.45, 0.01,
                             help="Threshold for overlapping boxes")
    
    max_detections = st.selectbox("📊 Maximum Detections", [10, 25, 50, 100, 1000], index=2)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🎨 Visualization")
    box_color = st.color_picker("📦 Box Color", "#00FF00")
    text_color = st.color_picker("📝 Text Color", "#FFFFFF")
    line_thickness = st.slider("📏 Line Thickness", 1, 5, 2)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Management Section
    st.markdown('<div class="data-management-section">', unsafe_allow_html=True)
    st.markdown("### 💾 Data Management")
    
    # Display current statistics
    if st.session_state.detection_history:
        last_update = max([item['timestamp'] for item in st.session_state.detection_history])
        st.markdown(f"""
        <div class="history-stats">
            <strong>📊 Saved Data</strong><br>
            Files: {st.session_state.processed_files}<br>
            Detections: {st.session_state.total_detections}<br>
            Last: {last_update.strftime('%Y-%m-%d %H:%M')}
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Data", help="Save current session data"):
            if save_detection_history():
                st.success("✅ Data saved!")
            else:
                st.error("❌ Save failed!")
    
    with col2:
        if st.button("📥 Load Data", help="Load saved data"):
            saved_data = load_detection_history()
            if saved_data:
                st.session_state.detection_history = saved_data.get('history', [])
                st.session_state.total_detections = saved_data.get('total_detections', 0)
                st.session_state.processed_files = saved_data.get('processed_files', 0)
                st.success("✅ Data loaded!")
                st.rerun()
            else:
                st.warning("⚠️ No saved data found!")
    
    # Export/Import options
    st.markdown("#### 📤 Export/Import")
    
    if st.session_state.detection_history:
        # Export to JSON
        export_data = {
            'detection_history': st.session_state.detection_history,
            'total_detections': st.session_state.total_detections,
            'processed_files': st.session_state.processed_files,
            'export_date': datetime.now().isoformat()
        }
        
        json_data = json.dumps(export_data, default=str, indent=2)
        st.download_button(
            label="📤 Export JSON",
            data=json_data,
            file_name=f"yolo_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Clear data option
    if st.button("🗑️ Clear All Data", help="Clear all detection history"):
        if st.session_state.detection_history:
            st.session_state.detection_history = []
            st.session_state.total_detections = 0
            st.session_state.processed_files = 0
            save_detection_history()  # Save empty state
            st.success("✅ All data cleared!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📈 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 Files Processed", st.session_state.processed_files)
    with col2:
        st.metric("🎯 Total Detections", st.session_state.total_detections)
    st.markdown('</div>', unsafe_allow_html=True)

# Auto-save data after each detection
def add_detection_to_history(detection_data):
    """Add detection to history and auto-save"""
    st.session_state.detection_history.append(detection_data)
    save_detection_history()  # Auto-save after each detection

# Main Content Area
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

# Dynamic model metrics based on selection
model_metrics = {
    'n': {'name': 'YOLOv8n', 'speed': '~80ms', 'accuracy': '~85%'},
    's': {'name': 'YOLOv8s', 'speed': '~90ms', 'accuracy': '~88%'},
    'm': {'name': 'YOLOv8m', 'speed': '~110ms', 'accuracy': '~90%'},
    'l': {'name': 'YOLOv8l', 'speed': '~140ms', 'accuracy': '~92%'},
    'x': {'name': 'YOLOv8x', 'speed': '~180ms', 'accuracy': '~94%'}
}

current_metrics = model_metrics[st.session_state.selected_model]

with col1:
    st.metric("🤖 Model", current_metrics['name'], "Active")
with col2:
    st.metric("⚡ Speed", current_metrics['speed'], "Average")
with col3:
    st.metric("🎯 Accuracy", current_metrics['accuracy'], "mAP50")
st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📷 Image Analysis", "🎥 Video Processing", "📹 Live Camera", "📊 Analytics", "📂 History Manager"])

# Detection Function
def process_detections(results, image, selected_classes, box_color, text_color, line_thickness):
    """Process YOLO detections and draw on image"""
    detection_count = 0
    detected_objects = []
    
    # Ensure image is in BGR format for OpenCV operations
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image, 0, []  # Return original image if format is invalid
    
    # Convert color codes to RGB
    try:
        box_rgb = tuple(int(box_color[i:i+2], 16) for i in (1, 3, 5))
        text_rgb = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))
        # Convert RGB to BGR for OpenCV
        box_color_bgr = (box_rgb[2], box_rgb[1], box_rgb[0])
        text_color_bgr = (text_rgb[2], text_rgb[1], text_rgb[0])
    except Exception:
        # Fallback to default colors if conversion fails
        box_color_bgr = (0, 255, 0)  # Green in BGR
        text_color_bgr = (255, 255, 255)  # White in BGR
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    if len(results.boxes.data) > 0:
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            class_name = results.names[int(class_id)]
            
            if class_name in selected_classes:
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color_bgr, line_thickness)
                
                # Draw label background
                text = f"{class_name} {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness)
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color_bgr, -1)
                
                # Draw text
                cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color_bgr, line_thickness)
                
                detection_count += 1
                detected_objects.append({
                    'class': class_name,
                    'confidence': score,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return image, detection_count, detected_objects

# Enhanced Image Analysis Tab
with tab1:
    st.markdown("## 📷 Image Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Upload Image", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            original_image = np.array(image)
            
            # YOLO detection
            with st.spinner("🔍 Analyzing..."):
                start_time = time.time()
                results = model(original_image, conf=confidence, iou=iou_threshold, max_det=max_detections)[0]
                process_time = time.time() - start_time
                
                processed_image, detection_count, detected_objects = process_detections(
                    results, original_image.copy(), selected_classes, box_color, text_color, line_thickness
                )
            
            # Display results
            col_original, col_processed = st.columns(2)
            
            with col_original:
                st.markdown("### 📋 Original")
                st.image(original_image, use_column_width=True)
            
            with col_processed:
                st.markdown("### 🎯 Detection Result")
                st.image(processed_image, channels="BGR", use_column_width=True)
            
            # Processing metrics
            st.markdown(f"⏱️ **Processing Time:** {process_time:.2f} seconds")
            
            # Update statistics
            st.session_state.total_detections += detection_count
            st.session_state.processed_files += 1
            
            # Add to detection history with auto-save
            detection_data = {
                'timestamp': datetime.now(),
                'type': 'image',
                'filename': uploaded_file.name,
                'detections': detection_count,
                'objects': detected_objects,
                'process_time': process_time,
                'model_used': st.session_state.selected_model,
                'settings': {
                    'confidence': confidence,
                    'iou_threshold': iou_threshold,
                    'selected_classes': selected_classes
                }
            }
            add_detection_to_history(detection_data)
            
            # Success message
            if detection_count > 0:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Analysis Complete!</h4>
                    <p>{detection_count} objects detected successfully</p>
                    <small>Data automatically saved to history</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <h4>ℹ️ No Objects Detected</h4>
                    <p>Try adjusting the confidence threshold or check if the image contains detectable objects</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None and 'detected_objects' in locals():
            st.markdown("### 📊 Detection Details")
            
            if detected_objects:
                # Class distribution
                class_counts = {}
                for obj in detected_objects:
                    class_counts[obj['class']] = class_counts.get(obj['class'], 0) + 1
                
                df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
                fig = px.bar(df, x='Class', y='Count', 
                           title="Detected Classes",
                           color='Count',
                           color_continuous_scale='viridis')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white' if st.session_state.dark_mode else 'black'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence scores
                confidences = [obj['confidence'] for obj in detected_objects]
                fig2 = go.Figure(data=go.Histogram(x=confidences, nbinsx=10))
                fig2.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Frequency",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white' if st.session_state.dark_mode else 'black'
                )
                st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Video Upload")
    video_file = st.file_uploader("Select a video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                temp_path = tfile.name
            
            # Video processing
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                st.error("Video could not be opened! Please try a different video file.")
            else:
                # Video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # FPS check
                if fps <= 0:
                    fps = 30  # Default FPS
                
                # Output video path
                output_path = "output_video.mp4"
                
                # Video codec and writer settings
                fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Video processing
                processed_frames = 0
                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # YOLO detection
                    results = model(frame, conf=confidence)[0]
                    
                    # Visualize results
                    if len(results.boxes.data) > 0:
                        for result in results.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = result
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{results.names[int(class_id)]} {score:.2f}", 
                                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    out.write(frame)
                    processed_frames += 1
                    progress = processed_frames / frame_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {processed_frames}/{frame_count} frame ({int(progress * 100)}%)")
                
                # Release resources
                cap.release()
                out.release()
                
                # Show processed video
                if processed_frames > 0:
                    st.success(f"Video processing completed! {processed_frames} frames processed.")
                    
                    # Check if video file exists
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        with open(output_path, 'rb') as video_file_output:
                            video_bytes = video_file_output.read()
                        st.video(video_bytes)
                    else:
                        st.error("Video file could not be created.")
                else:
                    st.error("No frames could be processed.")
            
            # Delete temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Clean up resources on error
            try:
                if 'cap' in locals():
                    cap.release()
                if 'out' in locals():
                    out.release()
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
                try:
                    if 'cap' in locals():
                        cap.release()
                    if 'out' in locals():
                        out.release()
                    if 'temp_path' in locals():
                        os.unlink(temp_path)
                except:
                    pass
    
    with col2:
        st.markdown("### 🎛️ Video Controls")
        st.markdown("""
        <div class="info-box">
            <h4>💡 Processing Tips</h4>
            <ul>
                <li>📹 Original video is shown first</li>
                <li>🎞️ Use frame skipping for large videos</li>
                <li>👁️ Enable preview to see processed frames</li>
                <li>📊 View detection statistics after processing</li>
                <li>💾 Download processed video when ready</li>
                <li>⚡ GPU acceleration improves speed</li>
                <li>💿 Results are automatically saved</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Video processing history
        if st.session_state.detection_history:
            video_sessions = [item for item in st.session_state.detection_history if item['type'] == 'video']
            if video_sessions:
                st.markdown("### 📊 Recent Videos")
                for session in video_sessions[-3:]:  # Show last 3 videos
                    st.markdown(f"""
                    <div class="history-stats">
                        <strong>📹 {session['filename']}</strong><br>
                        🎯 Detections: {session['detections']}<br>
                        ⏱️ Time: {session['process_time']:.1f}s<br>
                        🕒 {session['timestamp'].strftime('%H:%M')}
                    </div>
                    """, unsafe_allow_html=True)

with tab3:
    st.header("Camera Image")
    camera_input = st.camera_input("Camera")
    
    if camera_input is not None:
        # Read camera image
        image = Image.open(camera_input)
        image = np.array(image)
        
        # YOLO detection
        results = model(image, conf=confidence)[0]
        
        # Visualize results
        if len(results.boxes.data) > 0:
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{results.names[int(class_id)]} {score:.2f}", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        st.image(image, channels="BGR", use_container_width=True)

# Enhanced Analytics Dashboard
with tab4:
    st.markdown("## 📊 Analytics Dashboard")
    
    if st.session_state.detection_history:
        # Time series analysis
        df_history = pd.DataFrame(st.session_state.detection_history)
        df_history['hour'] = pd.to_datetime(df_history['timestamp']).dt.hour
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly detection distribution
            hourly_detections = df_history.groupby('hour')['detections'].sum().reset_index()
            fig = px.line(hourly_detections, x='hour', y='detections', 
                         title="Hourly Detection Distribution",
                         markers=True)
            fig.update_layout(
                xaxis_title="Hour", 
                yaxis_title="Detection Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white' if st.session_state.dark_mode else 'black'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # File type distribution
            type_counts = df_history['type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig = px.pie(type_counts, values='Count', names='Type', 
                        title="Processed File Types")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white' if st.session_state.dark_mode else 'black'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance analysis
        if 'process_time' in df_history.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_time = df_history['process_time'].mean()
                st.metric("⏱️ Average Processing Time", f"{avg_time:.2f}s")
            
            with col2:
                total_time = df_history['process_time'].sum()
                st.metric("🕒 Total Processing Time", f"{total_time:.2f}s")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📊 No Analytics Data Yet</h4>
            <p>Start processing images or videos to see analytics data here</p>
        </div>
        """, unsafe_allow_html=True)

# New History Manager Tab
with tab5:
    st.markdown("## 📂 Detection History Manager")
    
    if st.session_state.detection_history:
        # History overview
        total_sessions = len(st.session_state.detection_history)
        first_detection = min([item['timestamp'] for item in st.session_state.detection_history])
        last_detection = max([item['timestamp'] for item in st.session_state.detection_history])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Sessions", total_sessions)
        with col2:
            st.metric("📁 Files Processed", st.session_state.processed_files)
        with col3:
            st.metric("🎯 Total Detections", st.session_state.total_detections)
        with col4:
            avg_detections = st.session_state.total_detections / total_sessions if total_sessions > 0 else 0
            st.metric("📈 Avg per Session", f"{avg_detections:.1f}")
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📅 History Range</h4>
            <p><strong>First Detection:</strong> {first_detection.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Last Detection:</strong> {last_detection.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed history table
        st.markdown("### 📋 Detailed History")
        df_history = pd.DataFrame(st.session_state.detection_history)
        
        # Create display dataframe
        display_data = []
        for idx, item in enumerate(st.session_state.detection_history):
            display_data.append({
                'ID': idx + 1,
                'Time': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Type': item['type'].title(),
                'Model': item.get('model_used', 'n').upper(),
                'Filename': item.get('filename', 'N/A'),
                'Detections': item['detections'],
                'Process Time (s)': f"{item.get('process_time', 0):.2f}",
                'Confidence': item.get('settings', {}).get('confidence', 'N/A')
            })
        
        df_display = pd.DataFrame(display_data)
        
        # Add search and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search filename", "")
        with col2:
            filter_type = st.selectbox("📁 Filter by type", ["All", "Image", "Video", "Camera"])
        with col3:
            min_detections = st.number_input("🎯 Min detections", min_value=0, value=0)
        
        # Apply filters
        filtered_df = df_display.copy()
        if search_term:
            filtered_df = filtered_df[filtered_df['Filename'].str.contains(search_term, case=False, na=False)]
        if filter_type != "All":
            filtered_df = filtered_df[filtered_df['Type'] == filter_type]
        if min_detections > 0:
            filtered_df = filtered_df[filtered_df['Detections'] >= min_detections]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Bulk operations
        st.markdown("### 🛠️ Bulk Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"filtered_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🗑️ Clear Old Data (>30 days)"):
                cutoff_date = datetime.now() - pd.Timedelta(days=30)
                old_count = len(st.session_state.detection_history)
                st.session_state.detection_history = [
                    item for item in st.session_state.detection_history 
                    if item['timestamp'] > cutoff_date
                ]
                new_count = len(st.session_state.detection_history)
                removed = old_count - new_count
                
                if removed > 0:
                    save_detection_history()
                    st.success(f"✅ Removed {removed} old records")
                    st.rerun()
                else:
                    st.info("ℹ️ No old records to remove")
        
        with col3:
            if st.button("📊 Generate Report"):
                # Generate a comprehensive report
                report_data = {
                    'summary': {
                        'total_sessions': total_sessions,
                        'total_files': st.session_state.processed_files,
                        'total_detections': st.session_state.total_detections,
                        'date_range': f"{first_detection.strftime('%Y-%m-%d')} to {last_detection.strftime('%Y-%m-%d')}"
                    },
                    'details': st.session_state.detection_history
                }
                
                report_json = json.dumps(report_data, default=str, indent=2)
                st.download_button(
                    label="📊 Download Report",
                    data=report_json,
                    file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📂 No History Data</h4>
            <p>Start processing images or videos to build your detection history</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🚀 Powered by:**")
    st.markdown("- Ultralytics YOLOv8")
    st.markdown("- Streamlit")
    st.markdown("- OpenCV")
    st.markdown("- Plotly")

with col2:
    st.markdown("**📈 Performance:**")
    st.markdown("- Real-time processing")
    st.markdown("- GPU acceleration")
    st.markdown("- Batch processing")
    st.markdown("- Advanced analytics")

with col3:
    st.markdown("**🎯 Features:**")
    st.markdown("- 80+ object classes")
    st.markdown("- High accuracy detection")
    st.markdown("- Interactive visualizations")
    st.markdown("- Persistent data storage")

st.markdown('</div>', unsafe_allow_html=True)

def convert_video_with_ffmpeg(input_path, output_path):
    """Convert video to web-compatible format using FFMPEG"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            '-y',  # Overwrite output file if exists
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        st.error(f"❌ FFMPEG conversion error: {str(e)}")
        return False

def create_video_writer(path, fps, width, height):
    """Güvenli video yazıcı başlatıcı"""
    # Boyutların tam sayı olduğundan emin ol
    width = int(width)
    height = int(height)
    fps = int(fps)
    
    # En uyumlu codec ile dene
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"VideoWriter başlatılamadı! Yol: {path}, FPS: {fps}, Size: {width}x{height}")
    
    # Test frame yaz
    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
    success = out.write(test_frame)
    if not success:
        raise ValueError("Test frame yazılamadı! Video yazıcı düzgün çalışmıyor.")
    
    return out