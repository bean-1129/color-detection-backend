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
CORS(app)

# Load color dataset (fallback if missing)
try:
    colors_df = pd.read_csv('colors.csv', names=['color', 'color_name', 'hex', 'R', 'G', 'B'], header=None)
except FileNotFoundError:
    # Fallback color palette
    colors_df = pd.DataFrame({
        'R': [0,255,128,255,0,0,255,128,0,255],
        'G': [0,255,0,0,255,128,0,0,128,192],
        'B': [0,255,128,0,0,255,255,128,255,128],
        'color_name': ['black','white','gray','red','green','blue','yellow','orange','purple','pink']
    })

def get_color_name(R, G, B):
    """Find closest color name from RGB values"""
    minimum = 10000
    cname = "Unknown"
    for i in range(len(colors_df)):
        d = (abs(R - int(colors_df.iloc[i]['R'])) + 
             abs(G - int(colors_df.iloc[i]['G'])) + 
             abs(B - int(colors_df.iloc[i]['B'])))
        if d <= minimum:
            minimum = d
            cname = colors_df.iloc[i]['color_name']
    return cname

def extract_dominant_colors(image, num_colors=3):
    """Extract dominant colors using k-means"""
    pixels = image.reshape(-1, 3)
    pixels = pixels[(pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 700)]
    
    if len(pixels) == 0:
        return [(0, 0, 0)]
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    label_counts = Counter(labels)
    sorted_colors = [colors[i] for i, _ in label_counts.most_common()]
    
    return [(int(c[2]), int(c[1]), int(c[0])) for c in sorted_colors]  # BGR to RGB

def map_to_clothing_colors(color_names):
    """Map to standard clothing categories"""
    color_mapping = {
        'black': 'black', 'white': 'white', 'gray': 'gray', 'grey': 'gray',
        'brown': 'brown', 'beige': 'beige', 'tan': 'brown', 'khaki': 'beige',
        'red': 'red', 'maroon': 'red', 'pink': 'pink', 'orange': 'orange',
        'yellow': 'yellow', 'green': 'green', 'blue': 'blue', 'purple': 'purple'
    }
    
    mapped = []
    for name in color_names:
        name_lower = name.lower()
        for key, value in color_mapping.items():
            if key in name_lower:
                mapped.append(value)
                break
        else:
            mapped.append('multicolor')
    
    return list(dict.fromkeys(mapped))[:3]  # Unique, ordered

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'color-detection'}), 200

@app.route('/api/detect-colors', methods=['POST'])
def detect_colors():
    try:
        data = request.get_json()
        if not data or 'imageUrl' not in data:
            return jsonify({'error': 'Missing imageUrl'}), 400
        
        image_url = data['imageUrl']
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image'}), 400
        
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        img = cv2.resize(img, (400, 400))
        dominant_rgb = extract_dominant_colors(img, 5)
        color_names = [get_color_name(r, g, b) for r, g, b in dominant_rgb]
        clothing_colors = map_to_clothing_colors(color_names)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b).upper() for r, g, b in dominant_rgb[:3]]
        
        return jsonify({
            'colors': clothing_colors,
            'detailedColors': [{
                'name': color_names[i],
                'clothingColor': clothing_colors[i] if i < len(clothing_colors) else 'multicolor',
                'hex': hex_colors[i] if i < len(hex_colors) else '#000000',
                'rgb': {'r': dominant_rgb[i][0], 'g': dominant_rgb[i][1], 'b': dominant_rgb[i][2]}
            } for i in range(min(3, len(dominant_rgb)))]
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'fallback': ['black', 'gray']}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
