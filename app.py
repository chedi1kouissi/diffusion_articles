# app.py
"""
Simplified Flask microservice for French "assurance emprunteur" content generation

Three simple models:
1. Article Generator - generates articles from CSV themes
2. Hashtag & Caption Generator - creates social media content
3. HTML Transformer - converts content to simple HTML

Endpoints:
- POST /themes/import    - Import themes from CSV
- GET /themes/available  - List available themes
- POST /themes/reset     - Reset consumed themes
- POST /generate         - Generate complete content package
"""

import os
import json
import random
from datetime import datetime, timezone
from threading import Lock
from flask import Flask, request, jsonify
from flask_cors import CORS
from slugify import slugify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini SDK
from google import genai
from google.genai import types

# Configuration
STORE_PATH = os.environ.get("STORE_PATH", "themes_store.json")
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

_pick_lock = Lock()

# ---------------------- UTILITIES ----------------------

def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _ensure_store_exists():
    if not os.path.exists(STORE_PATH):
        data = {"themes": []}
        _atomic_save(data)

def _load_store() -> dict:
    _ensure_store_exists()
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"themes": []}

def _atomic_save(data: dict):
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STORE_PATH)

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

# ---------------------- MODEL 1: ARTICLE GENERATOR ----------------------

class ArticleGenerator:
    def __init__(self):
        self.system_instruction = """
Tu es un rédacteur expert en assurance emprunteur pour le marché français.
- Écris en français avec un ton clair et pédagogique
- Crée des articles de 800-1200 mots
- Structure: titre H1, introduction, 3-4 sections principales, conclusion
- Utilise des exemples concrets et évite le jargon technique
- Reste factuel et informatif sans donner de conseils personnalisés
- Focus sur le marché français uniquement
"""

    def generate_article(self, theme_title: str, target_keyword: str) -> str:
        """Generate a complete article based on theme and keyword"""
        
        # Mock mode for testing
        if os.environ.get("MOCK_GEN") == "1":
            return f"""# {theme_title}

## Introduction
Découvrez tout ce qu'il faut savoir sur {target_keyword} dans le contexte de l'assurance emprunteur en France.

## Les points essentiels
L'assurance emprunteur est obligatoire pour obtenir un crédit immobilier. Voici ce que vous devez savoir sur {target_keyword}.

## Comment bien choisir
Pour faire le bon choix concernant {target_keyword}, plusieurs critères sont à prendre en compte.

## Conclusion
En résumé, {target_keyword} joue un rôle important dans votre assurance emprunteur."""

        client = get_gemini_client()
        
        user_prompt = f"""
Écris un article complet sur le sujet suivant:
- Thème: {theme_title}
- Mot-clé principal: {target_keyword}

L'article doit être informatif, bien structuré et adapté aux emprunteurs français.
Utilise un format Markdown avec des titres H1, H2 et du contenu détaillé.
"""

        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2000,
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=user_prompt,
            config=config
        )

        return response.text or ""

# ---------------------- MODEL 2: HASHTAG & CAPTION GENERATOR ----------------------

class HashtagCaptionGenerator:
    def __init__(self):
        self.system_instruction = """
Tu es un expert en marketing digital spécialisé dans la création de contenu pour les réseaux sociaux.
- Crée du contenu engageant en français
- Utilise des hashtags pertinents pour l'assurance et l'immobilier français
- Adapte le ton selon chaque plateforme sociale
- Reste informatif et professionnel
"""

    def generate_social_content(self, article_title: str, target_keyword: str) -> dict:
        """Generate hashtags and captions for social media"""
        
        # Mock mode for testing
        if os.environ.get("MOCK_GEN") == "1":
            return {
                "linkedin_caption": f"📋 {article_title}\n\nDécouvrez notre guide complet sur {target_keyword} pour bien choisir votre assurance emprunteur.\n\n#AssuranceEmprunteur #Immobilier #ConseilFinance",
                "instagram_caption": f"Guide complet: {target_keyword} 📝\n\n#assurance #emprunt #immobilier #france #conseil #finance #pret",
                "twitter_thread": [
                    f"🧵 Guide sur {target_keyword}",
                    f"Les points essentiels à retenir pour votre assurance emprunteur",
                    "Tout ce qu'il faut savoir dans notre article complet !"
                ],
                "hashtags": ["#AssuranceEmprunteur", "#Immobilier", "#CreditImmobilier", "#France", "#Conseil", "#Finance"]
            }

        client = get_gemini_client()
        
        user_prompt = f"""
Crée du contenu pour les réseaux sociaux basé sur:
- Titre d'article: {article_title}
- Mot-clé: {target_keyword}

Génère:
1. Un post LinkedIn (150-200 caractères)
2. Une légende Instagram (avec emojis)
3. Un thread Twitter (3 tweets)
4. Une liste de 6 hashtags pertinents

Format la réponse en JSON avec les clés: linkedin_caption, instagram_caption, twitter_thread, hashtags
"""

        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=0.8,
            max_output_tokens=800,
            response_mime_type="application/json"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=config
        )

        try:
            return json.loads(response.text or "{}")
        except:
            return {
                "linkedin_caption": f"Guide sur {target_keyword} #AssuranceEmprunteur",
                "instagram_caption": f"Nouveau guide: {target_keyword} 📝",
                "twitter_thread": [f"Guide sur {target_keyword}"],
                "hashtags": ["#AssuranceEmprunteur"]
            }

# ---------------------- MODEL 3: HTML TRANSFORMER ----------------------

class HTMLTransformer:
    def __init__(self):
        pass

    def markdown_to_html(self, markdown_content: str) -> str:
        """Simple Markdown to HTML conversion"""
        if not markdown_content:
            return ""
        
        lines = markdown_content.split('\n')
        html_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                html_lines.append('')
                continue
                
            # Headers
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            # Lists
            elif line.startswith('- '):
                html_lines.append(f'<li>{line[2:]}</li>')
            # Paragraphs
            else:
                if not any(line.startswith(tag) for tag in ['<h1>', '<h2>', '<h3>', '<li>']):
                    html_lines.append(f'<p>{line}</p>')
                else:
                    html_lines.append(line)
        
        return '\n'.join(html_lines)

    def create_complete_html(self, article_content: str, social_content: dict, title: str) -> str:
        """Transform article and social content into complete HTML"""
        
        article_html = self.markdown_to_html(article_content)
        
        social_html = f"""
        <div class="social-content">
            <h2>Contenu Social Media</h2>
            
            <div class="linkedin">
                <h3>LinkedIn</h3>
                <p>{social_content.get('linkedin_caption', '')}</p>
            </div>
            
            <div class="instagram">
                <h3>Instagram</h3>
                <p>{social_content.get('instagram_caption', '')}</p>
            </div>
            
            <div class="twitter">
                <h3>Twitter Thread</h3>
                {''.join([f'<p>Tweet {i+1}: {tweet}</p>' for i, tweet in enumerate(social_content.get('twitter_thread', []))])}
            </div>
            
            <div class="hashtags">
                <h3>Hashtags</h3>
                <p>{' '.join(social_content.get('hashtags', []))}</p>
            </div>
        </div>
        """
        
        complete_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .social-content {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #ccc; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                h3 {{ color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <article>
                {article_html}
            </article>
            {social_html}
        </body>
        </html>
        """
        
        return complete_html

# ---------------------- THEME MANAGEMENT ----------------------

@app.route('/themes/import', methods=['POST'])
def import_themes():
    """Import themes from CSV"""
    body = request.get_json() or {}
    csv_path = body.get("csv_path")
    
    if not csv_path:
        return jsonify({"error": "csv_path is required"}), 400
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        store = _load_store()
        existing_ids = {t.get("id") for t in store.get("themes", [])}
        
        imported = 0
        for _, row in df.iterrows():
            theme_id = int(row.get("id"))
            if theme_id not in existing_ids:
                theme = {
                    "id": theme_id,
                    "titre": str(row.get("titre", "")),
                    "mot_cle_principal": str(row.get("mot_cle_principal", "")),
                    "consumed": False,
                    "consumed_at": None
                }
                store["themes"].append(theme)
                imported += 1
        
        _atomic_save(store)
        return jsonify({"status": "ok", "imported": imported})
        
    except Exception as e:
        return jsonify({"error": f"Failed to import: {e}"}), 400

@app.route('/themes/available', methods=['GET'])
def list_available_themes():
    """List all available (non-consumed) themes"""
    store = _load_store()
    available = [
        {"id": t["id"], "titre": t["titre"], "mot_cle_principal": t["mot_cle_principal"]}
        for t in store.get("themes", [])
        if not t.get("consumed")
    ]
    return jsonify({"available": len(available), "themes": available})

@app.route('/themes/reset', methods=['POST'])
def reset_themes():
    """Reset all themes to unconsumed state"""
    store = _load_store()
    for theme in store.get("themes", []):
        theme["consumed"] = False
        theme["consumed_at"] = None
    _atomic_save(store)
    return jsonify({"status": "ok", "message": "All themes reset"})

# ---------------------- CONTENT GENERATION ----------------------

@app.route('/generate', methods=['POST'])
def generate_complete_content():
    """Generate complete content package: article + social media + HTML"""
    
    # Select random unconsumed theme
    with _pick_lock:
        store = _load_store()
        available_themes = [t for t in store.get("themes", []) if not t.get("consumed")]
        
        if not available_themes:
            return jsonify({"error": "No available themes. Import themes or reset."}), 400
        
        # Pick random theme
        selected_theme = random.choice(available_themes)
        
        # Mark as consumed
        for theme in store["themes"]:
            if theme["id"] == selected_theme["id"]:
                theme["consumed"] = True
                theme["consumed_at"] = _now_iso()
                break
        
        _atomic_save(store)
    
    try:
        # Initialize generators
        article_gen = ArticleGenerator()
        social_gen = HashtagCaptionGenerator()
        html_gen = HTMLTransformer()
        
        # Generate content
        article_content = article_gen.generate_article(
            selected_theme["titre"], 
            selected_theme["mot_cle_principal"]
        )
        
        social_content = social_gen.generate_social_content(
            selected_theme["titre"],
            selected_theme["mot_cle_principal"]
        )
        
        complete_html = html_gen.create_complete_html(
            article_content,
            social_content,
            selected_theme["titre"]
        )
        
        return jsonify({
            "theme": {
                "id": selected_theme["id"],
                "titre": selected_theme["titre"],
                "mot_cle_principal": selected_theme["mot_cle_principal"]
            },
            "article_markdown": article_content,
            "social_content": social_content,
            "complete_html": complete_html,
            "slug": slugify(selected_theme["titre"]),
            "generated_at": _now_iso()
        })
        
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

# ---------------------- HEALTH CHECK ----------------------

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    store = _load_store()
    total_themes = len(store.get("themes", []))
    consumed_themes = len([t for t in store.get("themes", []) if t.get("consumed")])
    
    return jsonify({
        "status": "ok",
        "service": "Simplified Content Generator",
        "version": "1.0.0",
        "themes": {
            "total": total_themes,
            "consumed": consumed_themes,
            "available": total_themes - consumed_themes
        }
    })

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    # Check dependencies
    try:
        import pandas
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pandas python-dotenv")
        exit(1)
    
    # Verify API key is loaded
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not os.environ.get("MOCK_GEN"):
        print("⚠️  Warning: GEMINI_API_KEY not found in environment variables")
        print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        print("Or set MOCK_GEN=1 for testing without API")
        exit(1)
    
    print("🚀 Starting Article Generator...")
    print(f"📁 Themes store: {STORE_PATH}")
    print(f"🤖 Mock mode: {'ON' if os.environ.get('MOCK_GEN') == '1' else 'OFF'}")
    print(f"🔑 API key: {'Loaded' if api_key else 'Not loaded (Mock mode)'}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8084)))