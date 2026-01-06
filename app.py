from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from collections import Counter
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Load color dataset
colors_df = pd.read_csv('colors.csv', names=['color', 'color_name', 'hex', 'R', 'G', 'B'], header=None)

def get_color_name(R, G, B):
    """Find the closest color name from RGB values"""
    minimum = 10000
    cname = "Unknown"
    
    for i in range(len(colors_df)):
        d = abs(R - int(colors_df.loc[i, 'R'])) + \
            abs(G - int(colors_df.loc[i, 'G'])) + \
            abs(B - int(colors_df.loc[i, 'B']))
        
        if d <= minimum:
            minimum = d
            cname = colors_df.loc[i, 'color_name']
    
    return cname

def extract_dominant_colors(image, num_colors=3):
    """Extract dominant colors from image using k-means clustering"""
    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Remove very dark and very light pixels (likely shadows/highlights)
    pixels = pixels[
        (pixels.sum(axis=1) > 30) &  # Not too dark
        (pixels.sum(axis=1) < 700)    # Not too bright
    ]
    
    if len(pixels) == 0:
        return [(0, 0, 0)]
    
    # Use k-means to find dominant colors
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors)
    colors = kmeans.cluster_centers_
    
    # Count pixels in each cluster to rank by dominance
    labels = kmeans.labels_
    label_counts = Counter(labels)
    
    # Sort colors by frequency
    sorted_colors = [colors[i] for i, _ in label_counts.most_common()]
    
    return [(int(c[2]), int(c[1]), int(c[0])) for c in sorted_colors]  # BGR to RGB

def map_to_clothing_colors(color_names):
    """Map detected color names to standard clothing color categories"""
    color_mapping = {
        'black': 'black',
        'white': 'white',
        'gray': 'gray', 'grey': 'gray', 'silver': 'gray',
        'brown': 'brown', 'tan': 'brown', 'khaki': 'beige',
        'beige': 'beige', 'cream': 'beige', 'ivory': 'beige',
        'red': 'red', 'maroon': 'red', 'crimson': 'red',
        'pink': 'pink', 'rose': 'pink', 'salmon': 'pink',
        'orange': 'orange', 'coral': 'orange', 'peach': 'orange',
        'yellow': 'yellow', 'gold': 'yellow',
        'green': 'green', 'olive': 'green', 'lime': 'green',
        'blue': 'blue', 'navy': 'blue', 'cyan': 'blue', 'teal': 'blue',
        'purple': 'purple', 'violet': 'purple', 'lavender': 'purple',
    }
    
    mapped_colors = []
    for name in color_names:
        name_lower = name.lower()
        for key, value in color_mapping.items():
            if key in name_lower:
                mapped_colors.append(value)
                break
        else:
            # If no match, keep original
            mapped_colors.append('multicolor')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_colors = []
    for color in mapped_colors:
        if color not in seen:
            seen.add(color)
            unique_colors.append(color)
    
    return unique_colors[:3]  # Return top 3

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'color-detection'}), 200

@app.route('/api/detect-colors', methods=['POST'])
def detect_colors():
    """Detect dominant colors from image URL"""
    try:
        data = request.get_json()
        
        if not data or 'imageUrl' not in data:
            return jsonify({'error': 'Missing imageUrl parameter'}), 400
        
        image_url = data['imageUrl']
        
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image'}), 400
        
        # Convert to OpenCV image
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Resize for faster processing
        img = cv2.resize(img, (400, 400))
        
        # Extract dominant colors
        dominant_rgb_colors = extract_dominant_colors(img, num_colors=5)
        
        # Get color names
        color_names = [get_color_name(r, g, b) for r, g, b in dominant_rgb_colors]
        
        # Map to standard clothing colors
        clothing_colors = map_to_clothing_colors(color_names)
        
        # Convert RGB to hex for reference
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in dominant_rgb_colors[:3]]
        
        return jsonify({
            'colors': clothing_colors,
            'detailedColors': [
                {
                    'name': color_names[i],
                    'clothingColor': clothing_colors[i] if i < len(clothing_colors) else 'multicolor',
                    'hex': hex_colors[i] if i < len(hex_colors) else '#000000',
                    'rgb': {'r': dominant_rgb_colors[i][0], 'g': dominant_rgb_colors[i][1], 'b': dominant_rgb_colors[i][2]}
                }
                for i in range(min(3, len(dominant_rgb_colors)))
            ]
        }), 200
        
    except Exception as e:
        print(f"Error detecting colors: {str(e)}")
        return jsonify({'error': str(e), 'fallback': ['black', 'gray']}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)