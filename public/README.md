# Test UI pour le Générateur de Contenus

Interface utilisateur minimaliste pour tester l'API Flask de génération de contenus d'assurance emprunteur.

## Utilisation

### Option 1: Serveur dédié (Recommandé)
```bash
# Terminal 1: Démarrer l'API principale
python app.py

# Terminal 2: Démarrer l'interface utilisateur
python serve_ui.py
```
Puis ouvrir http://localhost:3000

### Option 2: Via Flask principal avec fichiers statiques
```bash
# Ajouter à app.py:
from flask import send_from_directory

@app.route('/ui')
def serve_ui():
    return send_from_directory('public', 'index.html')

@app.route('/public/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)
```
Puis ouvrir http://localhost:8084/ui

### Option 3: Fichier statique direct
Ouvrir `public/index.html` directement dans le navigateur (nécessite CORS autorisé).

## Fonctionnalités

### 1. Import de thèmes
- Saisir le chemin du fichier CSV (ex: `test_themes.csv`)
- Cliquer sur "Importer"
- Vérifier le nombre de thèmes importés

### 2. Visualisation des thèmes
- Cliquer sur "Rafraîchir" pour charger la liste
- Voir les thèmes disponibles dans un tableau
- Bouton "Réinitialiser" pour remettre tous les thèmes comme non-consommés

### 3. Génération de contenu
- Cliquer sur "Générer" pour sélectionner un thème aléatoire
- Voir les détails du thème sélectionné
- Naviguer entre les onglets :
  - **Aperçu article** : Contenu HTML sans balises `<html>/<body>`
  - **Social** : LinkedIn, Instagram, Twitter thread, hashtags
  - **JSON** : Réponse API complète

### 4. Export et copie
- **Copier HTML article** : Place le HTML de l'article dans le presse-papiers
- **Télécharger HTML** : Sauvegarde un fichier `.html` avec le contenu article
- **Copier JSON** : Copie la réponse JSON complète

## Configuration

### URL du backend
Modifier l'URL dans le panneau "Paramètres" si le serveur Flask ne fonctionne pas sur `http://localhost:8084`.

### Mode mock
Pour tester sans clé API Gemini, démarrer le serveur avec :
```bash
MOCK_GEN=1 python app.py
```

## Tests d'acceptation

### Avec MOCK_GEN=1
1. ✅ Import CSV → `{ status: "ok", imported: N }`
2. ✅ Rafraîchir → Liste des thèmes disponibles
3. ✅ Générer → JSON avec `article_markdown`, `social_content`, `complete_html`
4. ✅ Aperçu article → Contenu HTML à partir de H2 (pas de H1)
5. ✅ Copier HTML → Contenu dans le presse-papiers
6. ✅ Télécharger HTML → Fichier téléchargé
7. ✅ Gestion d'erreurs → Toasts visibles

## Sécurité

- Utilisation de **DOMPurify** pour nettoyer tout HTML inséré
- Pas de rendu de balises `<html>`, `<head>`, `<body>` dans les aperçus
- Validation des entrées utilisateur
- Timeouts sur les requêtes API (30s)

## Support navigateurs

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Support mobile responsive

## Dépannage

### "Serveur hors ligne"
- Vérifier que `app.py` fonctionne sur le port 8084
- Vérifier l'URL du backend dans les paramètres
- Consulter la console du navigateur pour les erreurs CORS

### "Timeout : le serveur ne répond pas"
- Génération avec Gemini peut prendre 15-30 secondes
- Vérifier la clé API Gemini si pas en mode mock
- Augmenter le timeout si nécessaire

### Problèmes de CORS
- Utiliser `serve_ui.py` comme proxy
- Ou ajouter les headers CORS à `app.py` :
```python
from flask_cors import CORS
CORS(app, origins=["http://localhost:3000"])
```
