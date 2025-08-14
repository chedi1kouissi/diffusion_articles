# app.py
"""
Flask microservice for French "assurance emprunteur" content ops – FILE-BASED STORE (no DB)

Features
- Theme store in a JSON file (import from CSV, list, reset)
- Random non-repeating theme picker (state persisted in JSON)
- Article generation with Gemini 2.5 Pro + Google Search grounding
- Optional two-phase generation (Flash outline -> Pro full article)
- Structured JSON output for CMS/n8n

Endpoints
---------
POST  /themes/import    {"csv_path": "./themes_assurance_emprunteur_100_fr.csv"}
GET   /themes/unconsumed
POST  /themes/reset
POST  /article/next     body overrides optional, e.g. {"recency_days":365,"min_sources":6}
POST  /article          provide {topic,target_query,...} to generate directly (bypass picker)
"""
from __future__ import annotations

import os
import json
import random
from datetime import datetime, timezone
from threading import Lock
from typing import List, Optional

from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from slugify import slugify
import tldextract
import html

# Gemini SDK
from google import genai
from google.genai import types

STORE_PATH = os.environ.get("STORE_PATH", os.path.abspath("themes_store.json"))
ALLOW_DOMAINS_ENV = os.environ.get("ALLOW_DOMAINS", "")
DEFAULT_ALLOW_DOMAINS = [d.strip() for d in ALLOW_DOMAINS_ENV.split(",") if d.strip()]

app = Flask(__name__)
_pick_lock = Lock()

# ---------------------- SCHEMAS ----------------------

class FAQ(BaseModel):
    question: str
    answer_html: str

class Citation(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    publisher: Optional[str] = None
    accessed_iso: str

class ArticleOut(BaseModel):
    h1: str
    slug: str
    meta_description: str
    outline: List[str]
    html: str
    keywords: List[str]
    faqs: List[FAQ]
    sources: List[Citation]
    schema_jsonld: dict

class ArticleIn(BaseModel):
    topic: str = Field(..., description="Sujet/titre de travail (FR)")
    target_query: str = Field(..., description="Mot-clé principal (FR)")
    language: str = "fr-FR"
    min_words: int = 900
    max_words: int = 1300
    min_sources: int = 5
    recency_days: int = 365
    url_context: Optional[List[HttpUrl]] = None
    allow_domains: Optional[List[str]] = None
    persona: Optional[str] = "Rédacteur senior assurance (France), ton clair et pédagogique."
    audience: Optional[str] = "Emprunteurs particuliers en France"
    brand_tone: Optional[str] = "Fiable, précis, sans jargon inutile."
    include_faqs: int = 4
    two_phase: bool = True

# ---------------------- FILE STORE ----------------------

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

# ---------------------- UTILITIES ----------------------

def domain_from_url(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join(p for p in [ext.domain, ext.suffix] if p)

def gclient():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

# ---------------------- PROMPTS ----------------------

def build_system_prompt(company_style: str) -> str:
    return (
        "Tu es un rédacteur senior spécialisé en assurance emprunteur pour le marché français.\n"
        f"- Langue: français (fr-FR). Style: {company_style}.\n"
        "- Objectif: produire un article SEO de haute qualité, factuel, sourcé et durable (éviter les points de droit susceptibles d’évoluer).\n"
        "- Pas de conseil juridique ni fiscal personnalisé. Pas de chiffres précis non sourcés.\n"
        "- Structure attendue: H1, chapo court, sections H2/H3, tableaux/bullets si utile, conclusion actionnable, FAQ.\n"
        "- Lisibilité: phrases courtes, vocabulaire accessible, exemples concrets.\n"
        "- SEO: inclure ~5–8 mots-clés longue traîne naturels; éviter le bourrage.\n"
        "- France uniquement: terminologie locale (DC, PTIA, IPT, ITT, quotité, TAEG…), références FR.\n"
    )

def build_user_prompt(p: ArticleIn) -> str:
    domain_hint = ""
    if p.allow_domains:
        domain_hint = (
            "Privilégie des sources françaises officielles ou fiables, "
            f"notamment: {', '.join(p.allow_domains)}. "
        )
    url_hint = ""
    if p.url_context:
        url_hint = "Analyse aussi le contenu des URLs fournies (si accessibles). "
    return (
        f"Sujet: {p.topic}\n"
        f"Requête cible (mot-clé principal): \"{p.target_query}\"\n\n"
        "Contraintes:\n"
        f"- Longueur: {p.min_words}–{p.max_words} mots.\n"
        f"- Sources: au moins {p.min_sources} sources récentes (≤ {p.recency_days} jours) ET/OU reconnues comme références stables.\n"
        f"- {domain_hint}{url_hint}\n"
        "- Évite les promesses absolues et les détails juridiques susceptibles de changer. Reste pédagogique.\n\n"
        "Livrables (renvoie UNIQUEMENT le JSON demandé par le schéma):\n"
        "- h1, slug, meta_description (~155 caractères), outline (liste des intertitres),\n"
        "- html (corps complet en HTML propre, sans <script>), keywords (liste),\n"
        "- faqs (Q/R en HTML concis),\n"
        "- schema_jsonld (type Article + FAQPage, FR),\n"
        "- sources sera rempli par le système via le grounding : n’essaie pas de l’inventer.\n"
    )

# ---------------------- GEMINI CALLS ----------------------

def google_search_tool():
    return types.Tool(google_search=types.GoogleSearch())

def url_context_tool():
    return types.Tool(url_context=types.UrlContext())

def extract_citations(resp) -> List[dict]:
    out = []
    try:
        meta = resp.candidates[0].grounding_metadata
    except Exception:
        meta = None
    if not meta:
        return out
    chunks = getattr(meta, "grounding_chunks", None) or []
    seen = set()
    for ch in chunks:
        web = getattr(ch, "web", None)
        if not web or not web.uri:
            continue
        url = web.uri
        if url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "url": url,
                "title": getattr(web, "title", None),
                "publisher": domain_from_url(url),
                "accessed_iso": _now_iso(),
            }
        )
    return out

def build_jsonld(h1: str, meta_desc: str, url_slug: str, citations: List[dict]) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": h1,
        "inLanguage": "fr-FR",
        "description": meta_desc,
        "mainEntityOfPage": {"@type": "WebPage", "@id": f"https://www.example.com/{url_slug}"},
        "author": {"@type": "Organization", "name": "Votre Marque"},
        "publisher": {"@type": "Organization", "name": "Votre Marque"},
        "dateModified": _now_iso(),
        "citation": [c["url"] for c in citations] if citations else [],
    }

def gemini_generate_article(p: ArticleIn) -> str:
    sys_prompt = build_system_prompt(p.brand_tone or "") + (
        "\nConsigne de sortie: renvoie UNIQUEMENT l'article en HTML propre (pas de JSON, pas de texte autour)."
    )

    # Prompt orienté article direct, sans structure JSON
    domain_hint = ""
    if p.allow_domains:
        domain_hint = (
            "Privilégie des sources françaises officielles ou fiables, notamment: "
            + ", ".join(p.allow_domains)
            + ". "
        )
    url_hint = ""
    if p.url_context:
        url_hint = "Analyse aussi le contenu des URLs fournies (si accessibles). "

    user_prompt = (
        f"Sujet: {p.topic}\n"
        f"Requête cible (mot-clé principal): \"{p.target_query}\"\n\n"
        "Contraintes:\n"
        f"- Longueur: {p.min_words}–{p.max_words} mots.\n"
        f"- Sources: {p.min_sources}+ sources récentes (≤ {p.recency_days} jours) ET/OU reconnues comme références stables.\n"
        f"- {domain_hint}{url_hint}\n"
        "- Évite les promesses absolues et les détails juridiques susceptibles de changer. Reste pédagogique.\n\n"
        "Livrable: ARTICLE UNIQUEMENT en HTML propre (H1, chapo court, sections H2/H3, listes/tableaux si utile, conclusion)."
    )

    tools = [google_search_tool()]
    if p.url_context:
        tools.append(url_context_tool())

    cli = gclient()

    # Option: plan rapide
    outline_text = None
    if p.two_phase:
        outline_cfg = types.GenerateContentConfig(
            system_instruction=sys_prompt + "\nConcentre-toi uniquement sur un plan H2/H3 et un chapo (100 mots).",
            tools=tools,
            temperature=0.4,
            top_p=0.9,
            max_output_tokens=700,
        )
        prompt_outline = user_prompt + "\n\nNe renvoie qu'un plan (H2/H3) + chapo."
        outline_resp = cli.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_outline,
            config=outline_cfg,
        )
        outline_text = outline_resp.text or ""

    # Article complet
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        tools=tools,
        temperature=0.6,
        top_p=0.9,
        max_output_tokens=2400,
    )

    full_prompt = user_prompt
    if outline_text:
        full_prompt += "\n\nPlan suggéré à respecter et améliorer :\n" + outline_text
    if p.url_context:
        full_prompt += "\nURLs à considérer :\n" + "\n".join([str(u) for u in p.url_context])

    resp = cli.models.generate_content(
        model="gemini-2.5-pro",
        contents=full_prompt,
        config=config,
    )

    html_text = resp.text or ""
    # Nettoyage basique si le modèle entoure dans des fences
    if html_text.strip().startswith("```"):
        html_text = html_text.strip().strip("`")
        # Retire un éventuel indicateur de langage comme 'html'
        if html_text.startswith("html"):
            html_text = html_text[len("html"):].lstrip()

    return html_text.strip()

def _extract_first_json_object(text: str):
    """Return the first valid JSON object found in text, or None."""
    try:
        import json as _json
    except Exception:
        return None
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = idx
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : idx + 1]
                    try:
                        return _json.loads(candidate)
                    except Exception:
                        start = None
                        continue
    return None


def _structure_with_model(raw_text: str, p: ArticleIn) -> dict:
    """Second-pass: ask the model (no tools) to emit strict JSON matching ArticleOut."""
    cli = gclient()
    sys_msg = (
        "Transforme le contenu suivant en un JSON STRICT conforme au schéma ArticleOut. "
        "Renvoie UNIQUEMENT du JSON sans texte autour. N'invente pas 'sources' (laisse vide), "
        "et respecte la langue fr-FR."
    )
    cfg = types.GenerateContentConfig(
        system_instruction=sys_msg,
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=1800,
    )
    prompt = (
        f"Sujet: {p.topic}\n"
        f"Requête cible: {p.target_query}\n\n"
        "Contenu à structurer ci-dessous. Convertis-le en JSON ArticleOut (h1, slug, meta_description, outline, html, keywords, faqs, schema_jsonld, sources=[]):\n\n"
        f"{raw_text}"
    )
    resp = cli.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=cfg,
    )
    parsed = getattr(resp, "parsed", None)
    if parsed:
        return parsed
    text = resp.text or "{}"
    # Try direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fenced
    try:
        cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception:
        pass
    # Try first object extraction
    extracted = _extract_first_json_object(text)
    if extracted is not None:
        return extracted
    # Last resort: synthesize minimal valid structure using escaped raw text
    minimal = {
        "h1": p.topic,
        "slug": slugify(p.topic),
        "meta_description": p.target_query[:155],
        "outline": [],
        "html": f"<p>{html.escape(text or raw_text)}</p>",
        "keywords": [p.target_query],
        "faqs": [],
        "sources": [],
        "schema_jsonld": build_jsonld(p.topic, p.target_query[:155], slugify(p.topic), []),
    }
    return minimal

def _normalize_article_dict(data: dict, p: ArticleIn, citations: List[dict]) -> dict:
    result = dict(data or {})
    if not result.get("h1"):
        result["h1"] = p.topic
    if not result.get("meta_description"):
        fallback_meta = (p.target_query or p.topic or "")[:155]
        result["meta_description"] = fallback_meta
    if not isinstance(result.get("outline"), list):
        result["outline"] = []
    if not result.get("html"):
        result["html"] = f"<h1>{html.escape(result['h1'])}</h1><p>{html.escape(p.topic)}</p>"
    if not isinstance(result.get("faqs"), list):
        result["faqs"] = []
    if "keywords" not in result or not isinstance(result["keywords"], list) or not result["keywords"]:
        result["keywords"] = [p.target_query]
    if "sources" not in result or not isinstance(result["sources"], list):
        result["sources"] = citations or []
    if not result.get("slug"):
        result["slug"] = slugify(result["h1"])
    if not result.get("schema_jsonld"):
        result["schema_jsonld"] = build_jsonld(
            result["h1"],
            result["meta_description"],
            result["slug"],
            citations,
        )
    return result

# ---------------------- THEME OPS (FILE-BASED) ----------------------

@app.post("/themes/import")
def import_themes():
    """
    Import themes from CSV with columns:
    id,cluster,titre,mot_cle_principal,intention,type_evergreen,notes
    """
    body = request.get_json(force=True) or {}
    csv_path = body.get("csv_path")
    if not csv_path:
        return jsonify({"error": "csv_path is required"}), 400
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    store = _load_store()
    existing = {int(t.get("id")): t for t in store.get("themes", []) if "id" in t}

    imported = 0
    for _, r in df.iterrows():
        tid = int(r.get("id"))
        theme = {
            "id": tid,
            "cluster": str(r.get("cluster", "")),
            "titre": str(r.get("titre", "")),
            "mot_cle_principal": str(r.get("mot_cle_principal", "")),
            "intention": str(r.get("intention", "")),
            "type_evergreen": str(r.get("type_evergreen", "")),
            "notes": str(r.get("notes", "")),
            "consumed": existing.get(tid, {}).get("consumed", 0),
            "consumed_at": existing.get(tid, {}).get("consumed_at"),
        }
        existing[tid] = theme
        imported += 1

    store["themes"] = list(existing.values())
    _atomic_save(store)
    return jsonify({"status": "ok", "imported": imported})

@app.get("/themes/unconsumed")
def list_unconsumed():
    store = _load_store()
    rows = [
        {
            "id": t["id"],
            "cluster": t.get("cluster"),
            "titre": t.get("titre"),
            "mot_cle_principal": t.get("mot_cle_principal"),
        }
        for t in store.get("themes", [])
        if not t.get("consumed")
    ]
    rows.sort(key=lambda x: x["id"])
    return jsonify(rows)

@app.post("/themes/reset")
def reset_themes():
    store = _load_store()
    for t in store.get("themes", []):
        t["consumed"] = 0
        t["consumed_at"] = None
    _atomic_save(store)
    return jsonify({"status": "ok"})

# ---------------------- ARTICLE GENERATION ----------------------

@app.post("/article/next")
def generate_article_from_random_theme():
    """Pick a random unconsumed theme, mark consumed, generate article."""
    overrides = request.get_json(silent=True) or {}

    with _pick_lock:
        store = _load_store()
        candidates = [t for t in store.get("themes", []) if not t.get("consumed")]
        if not candidates:
            return jsonify({"error": "No unconsumed themes left. Reset or import more."}), 409
        chosen = random.choice(candidates)
        chosen["consumed"] = 1
        chosen["consumed_at"] = _now_iso()
        for i, t in enumerate(store["themes"]):
            if t.get("id") == chosen.get("id"):
                store["themes"][i] = chosen
                break
        _atomic_save(store)

    allow_domains = overrides.get("allow_domains") or DEFAULT_ALLOW_DOMAINS
    payload = ArticleIn(
        topic=chosen.get("titre"),
        target_query=chosen.get("mot_cle_principal") or chosen.get("titre"),
        min_words=int(overrides.get("min_words", 900)),
        max_words=int(overrides.get("max_words", 1300)),
        min_sources=int(overrides.get("min_sources", 5)),
        recency_days=int(overrides.get("recency_days", 365)),
        allow_domains=allow_domains,
        include_faqs=int(overrides.get("include_faqs", 4)),
        two_phase=bool(overrides.get("two_phase", True)),
    )

    try:
        article = gemini_generate_article(payload)
    except ValidationError as ve:
        return jsonify({"error": "Schema validation failed", "details": json.loads(ve.json())}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

    return jsonify(
        {
            "theme": {
                "id": chosen.get("id"),
                "titre": chosen.get("titre"),
                "mot_cle_principal": chosen.get("mot_cle_principal"),
            },
            "article": article,
        }
    )

@app.post("/article")
def generate_article_direct():
    """Generate article directly from explicit input (no theme pick)."""
    data = request.get_json(force=True)
    try:
        if not data.get("allow_domains") and DEFAULT_ALLOW_DOMAINS:
            data["allow_domains"] = DEFAULT_ALLOW_DOMAINS
        payload = ArticleIn(**data)
    except ValidationError as ve:
        return jsonify({"error": "Invalid payload", "details": json.loads(ve.json())}), 400

    try:
        article = gemini_generate_article(payload)
    except ValidationError as ve:
        return jsonify({"error": "Schema validation failed", "details": json.loads(ve.json())}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

    return jsonify(article)

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8084)))
