#!/usr/bin/env python3
"""
Simple Flask server to serve the test UI and proxy API calls.
Run this instead of app.py to serve both the UI and API.
"""

from flask import Flask, send_from_directory, request, jsonify
import requests
import os

app = Flask(__name__)

# API proxy target
API_BASE = "http://localhost:8084"

@app.route('/')
def serve_ui():
    """Serve the main UI"""
    return send_from_directory('public', 'index.html')

@app.route('/public/<path:filename>')
def serve_static(filename):
    """Serve static files from public directory"""
    return send_from_directory('public', filename)

# Health check proxy
@app.route('/api/')
def proxy_health():
    """Proxy health check"""
    try:
        resp = requests.get(f"{API_BASE}/", timeout=10)
        return resp.json(), resp.status_code
    except requests.exceptions.RequestException as e:
        return {"error": f"Backend unavailable: {str(e)}"}, 503

# Proxy all API calls to the main Flask app
@app.route('/api/themes/<path:path>', methods=['GET', 'POST'])
@app.route('/api/themes/import', methods=['POST'])
@app.route('/api/themes/available', methods=['GET'])
@app.route('/api/themes/reset', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
def proxy_api(path=None):
    """Proxy API calls to the main Flask application"""
    # Construct the target URL
    if path:
        if request.endpoint.startswith('proxy_api'):
            # Extract the actual path from the request
            actual_path = request.path
        else:
            actual_path = f"/{path}"
    else:
        actual_path = request.path
    
    target_url = f"{API_BASE}{actual_path}"
    
    try:
        # Forward the request with longer timeout for generation endpoints
        timeout = 120 if 'generate' in target_url else 30
        
        if request.method == 'GET':
            resp = requests.get(target_url, timeout=timeout)
        elif request.method == 'POST':
            resp = requests.post(
                target_url, 
                json=request.get_json() if request.is_json else None,
                data=request.form if not request.is_json else None,
                timeout=timeout
            )
        else:
            resp = requests.request(
                method=request.method,
                url=target_url,
                json=request.get_json() if request.is_json else None,
                timeout=timeout
            )
        
        # Return the response
        return jsonify(resp.json()), resp.status_code
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Backend connection failed: {str(e)}"}), 503

if __name__ == '__main__':
    print("🌐 Starting UI server on http://localhost:3000")
    print("📡 Proxying API calls to http://localhost:8084")
    print("💡 Make sure your main Flask app (app.py) is running on port 8084")
    print()
    print("Open http://localhost:3000 in your browser to use the test UI")
    
    app.run(host='0.0.0.0', port=3000, debug=True)
