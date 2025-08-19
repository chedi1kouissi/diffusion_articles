# app.py
"""
Flask microservice for French "assurance emprunteur" content ops – FILE-BASED STORE (no DB)

Features
- Theme store in a JSON file (import from CSV, list, reset)
- Random non-repeating theme picker (state persisted in JSON)
- Article generation with Gemini 2.5 Pro + Google Search grounding
- Optional two-phase generation (Flash outline -> Pro full article)
- 2-step pipeline: Markdown text generation + HTML rendering
- Structured JSON output for CMS/n8n with social media content

Endpoints
---------
POST  /themes/import    {"csv_path": "./themes_assurance_emprunteur_100_fr.csv"}
GET   /themes/unconsumed
POST  /themes/reset
POST  /article/next     body overrides optional, e.g. {"recency_days":365,"min_sources":6}
POST  /article          provide {topic,target_query,...} to generate directly (bypass picker)
POST  /article/text     generate structured text with Markdown (new pipeline)
POST  /render/html      convert ArticleTextOut to clean WordPress HTML
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
import html

# Markdown rendering and sanitization
from markdown_it import MarkdownIt
import bleach

# Gemini SDK
from google import genai
from google.genai import types

STORE_PATH = os.environ.get("STORE_PATH", os.path.abspath("themes_store.json"))
ALLOW_DOMAINS_ENV = os.environ.get("ALLOW_DOMAINS", "")
DEFAULT_ALLOW_DOMAINS = [d.strip() for d in ALLOW_DOMAINS_ENV.split(",") if d.strip()]

app = Flask(__name__)
_pick_lock = Lock()

# Global Markdown and sanitization setup
_MD = MarkdownIt("commonmark").enable("table").enable("strikethrough").enable("linkify")
_BLEACH_TAGS = ["h1","h2","h3","p","ul","ol","li","a","strong","em","blockquote","code","pre","table","thead","tbody","tr","th","td","hr","br"]
_BLEACH_ATTRS = {"a": ["href","title","rel","target"]}

# ---------------------- SCHEMAS ----------------------

# Legacy schemas (kept for backward compatibility)
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

# New text-based schemas
class Section(BaseModel):
    title: str
    body_md: str

class FaqText(BaseModel):
    question: str
    answer_md: str

class SocialOut(BaseModel):
    linkedin_text: str
    x_thread: List[str]
    instagram_caption: str

class ArticleTextOut(BaseModel):
    h1: str
    slug: str
    meta_description: str
    outline: List[str]
    chapo_md: str
    sections: List[Section]
    keywords: List[str]
    faqs: List[FaqText]
    schema_jsonld: dict
    sources: List[Citation]
    social: SocialOut

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
    import tldextract
    ext = tldextract.extract(url)
    return ".".join(p for p in [ext.domain, ext.suffix] if p)

def gclient():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

# ---------------------- MARKDOWN RENDERING ----------------------

def render_markdown_to_html(md_text: str) -> str:
    """Convert Markdown to HTML and sanitize."""
    raw = _MD.render(md_text or "")
    cleaned = bleach.clean(raw, tags=_BLEACH_TAGS, attributes=_BLEACH_ATTRS, strip=True)
    cleaned = bleach.linkify(cleaned)
    return cleaned.strip()

# ---------------------- RESPONSE SCHEMA HELPERS ----------------------

def _article_text_response_schema() -> types.Schema:
    """Schema for strict JSON response from Gemini."""
    return types.Schema(
        type="object",
        properties={
            "h1": {"type": "string"},
            "slug": {"type": "string"},
            "meta_description": {"type": "string"},
            "outline": {"type": "array", "items": {"type": "string"}},
            "chapo_md": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body_md": {"type": "string"},
                    },
                    "required": ["title","body_md"],
                },
            },
            "keywords": {"type": "array", "items": {"type": "string"}},
            "faqs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer_md": {"type": "string"},
                    },
                    "required": ["question","answer_md"],
                },
            },
            "schema_jsonld": {"type": "object"},
            "social": {
                "type": "object",
                "properties": {
                    "linkedin_text": {"type": "string"},
                    "x_thread": {"type": "array", "items": {"type": "string"}},
                    "instagram_caption": {"type": "string"},
                },
                "required": ["linkedin_text","x_thread","instagram_caption"],
            },
        },
        required=["h1","slug","meta_description","outline","chapo_md","sections","keywords","faqs","schema_jsonld","social"],
    )

def _structure_with_schema(markdown_text: str, p: ArticleIn) -> dict:
    """Pass B: Transform Markdown into strict ArticleTextOut JSON using response schema."""
    cli = gclient()
    cfg = types.GenerateContentConfig(
        system_instruction=(
            "Transforme le CONTENU CI-DESSOUS en JSON STRICT conforme au schéma ArticleTextOut. "
            "Réponds UNIQUEMENT en JSON. Le contenu source est en Markdown. "
            "Conserve tout le contenu textuel en Markdown (chapo_md, sections[].body_md, faqs[].answer_md)."
        ),
        temperature=0.2,
        top_p=0.95,
        response_mime_type="application/json",
        response_schema=_article_text_response_schema(),
    )
    prompt = (
        f"Sujet: {p.topic}\n"
        f"Requête: {p.target_query}\n\n"
        "Contenu Markdown à structurer :\n\n"
        f"{markdown_text}"
    )
    resp = cli.models.generate_content(model="gemini-2.5-pro", contents=prompt, config=cfg)
    return json.loads(resp.text or "{}")

# ---------------------- PROMPTS ----------------------

def build_system_prompt_text(company_style: str) -> str:
    return (
        "Tu es un rédacteur senior spécialisé en assurance emprunteur pour le marché français.\n"
        f"- Langue: français (fr-FR). Style: {company_style}.\n"
        "- Objectif: produire un article SEO de haute qualité, factuel, sourcé et durable.\n"
        "- Format de sortie: MARKDOWN uniquement (pas de JSON, pas de HTML).\n"
        "- Pas de conseil juridique ni fiscal personnalisé. Pas de chiffres précis non sourcés.\n"
        "- Structure: H1, chapo court, sections avec titres, FAQ, contenu social.\n"
        "- Lisibilité: phrases courtes, vocabulaire accessible, exemples concrets.\n"
        "- SEO: inclure ~5–8 mots-clés longue traîne naturels; éviter le bourrage.\n"
        "- France uniquement: terminologie locale (DC, PTIA, IPT, ITT, quotité, TAEG…), références FR.\n"
    )

def build_user_prompt_text(p: ArticleIn) -> str:
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
        "- Évite les promesses absolues et les détails juridiques susceptibles de changer.\n\n"
        "Livrables (MARKDOWN uniquement):\n"
        "- H1 principal\n"
        "- Chapo introductif (100-150 mots)\n"
        "- Sections avec titres H2/H3 et contenu détaillé\n"
        f"- {p.include_faqs} FAQ avec questions et réponses\n"
        "- Contenu complet, structuré, prêt pour transformation JSON.\n"
        "\nTout le contenu textuel doit être en MARKDOWN (pas de HTML)."
    )

def build_system_prompt(company_style: str) -> str:
    """Legacy prompt for backward compatibility."""
    return (
        "Tu es un rédacteur senior spécialisé en assurance emprunteur pour le marché français.\n"
        f"- Langue: français (fr-FR). Style: {company_style}.\n"
        "- Objectif: produire un article SEO de haute qualité, factuel, sourcé et durable (éviter les points de droit susceptibles d'évoluer).\n"
        "- Pas de conseil juridique ni fiscal personnalisé. Pas de chiffres précis non sourcés.\n"
        "- Structure attendue: H1, chapo court, sections H2/H3, tableaux/bullets si utile, conclusion actionnable, FAQ.\n"
        "- Lisibilité: phrases courtes, vocabulaire accessible, exemples concrets.\n"
        "- SEO: inclure ~5–8 mots-clés longue traîne naturels; éviter le bourrage.\n"
        "- France uniquement: terminologie locale (DC, PTIA, IPT, ITT, quotité, TAEG…), références FR.\n"
    )

def build_user_prompt(p: ArticleIn) -> str:
    """Legacy prompt for backward compatibility."""
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
        "- sources sera rempli par le système via le grounding : n'essaie pas de l'inventer.\n"
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

def gemini_generate_article_text(p: ArticleIn) -> tuple[dict, list]:
    """Two-pass generation: Pass A (with tools) -> Markdown, Pass B (no tools) -> structured JSON."""
    
    # Check for mock mode
    if os.environ.get("MOCK_GEN") == "1":
        mock_data = {
            "h1": f"Guide complet : {p.topic}",
            "slug": slugify(p.topic),
            "meta_description": f"Découvrez tout sur {p.topic} et {p.target_query}. Guide complet pour les emprunteurs français.",
            "outline": ["Introduction", "Points clés", "FAQ"],
            "chapo_md": f"**Introduction** sur {p.topic}. Ce guide vous explique les essentiels du {p.target_query}.",
            "sections": [
                {"title": "Points clés", "body_md": f"Les éléments importants concernant {p.target_query}:\n\n- Point 1\n- Point 2\n- Point 3"}
            ],
            "keywords": [p.target_query, "assurance emprunteur", "France"],
            "faqs": [
                {"question": f"Qu'est-ce que {p.target_query} ?", "answer_md": f"Le {p.target_query} est un élément important de l'assurance emprunteur."}
            ],
            "schema_jsonld": build_jsonld(f"Guide complet : {p.topic}", f"Guide sur {p.topic}", slugify(p.topic), []),
            "social": {
                "linkedin_text": f"Découvrez notre guide complet sur {p.topic}. Tout ce qu'il faut savoir pour bien choisir votre assurance emprunteur.",
                "x_thread": [
                    f"🧵 Thread sur {p.topic}",
                    f"Les points essentiels à retenir sur {p.target_query}",
                    "Plus d'infos dans notre article complet !"
                ],
                "instagram_caption": f"Guide complet sur {p.topic} 📝 #assurance #emprunt #conseil #france"
            }
        }
        return mock_data, []

    sys_prompt = build_system_prompt_text(p.brand_tone or "")
    user_prompt = build_user_prompt_text(p)

    tools = [google_search_tool()]
    if p.url_context:
        tools.append(url_context_tool())

    cli = gclient()

    # Optional outline phase (Flash)
    outline_text = None
    if p.two_phase:
        outline_cfg = types.GenerateContentConfig(
            system_instruction=sys_prompt + "\nConcentre-toi uniquement sur un PLAN H2/H3 + CHAPO (100 mots). MARKDOWN UNIQUEMENT.",
            tools=tools,
            temperature=0.4,
            top_p=0.9,
            max_output_tokens=700,
        )
        prompt_outline = user_prompt + "\n\nNe renvoie qu'un plan (H2/H3) + chapo en Markdown."
        outline_resp = cli.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_outline,
            config=outline_cfg
        )
        outline_text = outline_resp.text or ""

    # Pass A: Generate Markdown content with tools
    config_A = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        tools=tools,
        temperature=0.6,
        top_p=0.9,
        max_output_tokens=3000,
    )

    full_prompt = user_prompt
    if outline_text:
        full_prompt += "\n\nPlan suggéré à respecter et améliorer :\n" + outline_text
    if p.url_context:
        full_prompt += "\nURLs à considérer :\n" + "\n".join([str(u) for u in p.url_context])

    resp_A = cli.models.generate_content(
        model="gemini-2.5-pro",
        contents=full_prompt,
        config=config_A,
    )

    citations = extract_citations(resp_A)
    markdown_body = resp_A.text or ""

    # Pass B: Structure into JSON
    try:
        structured = _structure_with_schema(markdown_body, p)
        return structured, citations
    except Exception as e:
        # Error transparency: return 502 with debug info
        raise RuntimeError(f"Pass B failed: {e}. First 500 chars: {markdown_body[:500]}")

def gemini_generate_article(p: ArticleIn) -> str:
    """Legacy pipeline: Generate HTML article."""
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

def _normalize_article_text_dict(data: dict, p: ArticleIn, citations: List[dict]) -> dict:
    """Normalize ArticleTextOut data."""
    result = dict(data or {})
    
    if not result.get("h1"):
        result["h1"] = p.topic
    if not result.get("meta_description"):
        fallback_meta = (p.target_query or p.topic or "")[:155]
        result["meta_description"] = fallback_meta
    if not isinstance(result.get("outline"), list):
        result["outline"] = []
    if not result.get("chapo_md"):
        result["chapo_md"] = f"Introduction sur {p.topic}."
    if not isinstance(result.get("sections"), list):
        result["sections"] = [{"title": "Contenu principal", "body_md": f"Contenu sur {p.topic}."}]
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
    if not result.get("social"):
        result["social"] = {
            "linkedin_text": f"Découvrez notre guide sur {p.topic}. Tout ce qu'il faut savoir pour bien choisir.",
            "x_thread": [
                f"🧵 Thread sur {p.topic}",
                f"Les points essentiels à retenir sur {p.target_query}",
                "Plus d'infos dans notre article complet !"
            ],
            "instagram_caption": f"Guide complet sur {p.topic} 📝 #assurance #emprunt #conseil #france"
        }
    
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

# ---------------------- NEW TEXT PIPELINE ----------------------

@app.post("/article/text")
def generate_article_text_endpoint():
    """Generate structured text article with Markdown content."""
    data = request.get_json(force=True)
    try:
        if not data.get("allow_domains") and DEFAULT_ALLOW_DOMAINS:
            data["allow_domains"] = DEFAULT_ALLOW_DOMAINS
        payload = ArticleIn(**data)
    except ValidationError as ve:
        return jsonify({"error": "Invalid payload", "details": json.loads(ve.json())}), 400

    try:
        article_data, citations = gemini_generate_article_text(payload)
        normalized = _normalize_article_text_dict(article_data, payload, citations)
        
        # Validate the result
        article_text_out = ArticleTextOut(**normalized)
        return jsonify(article_text_out.model_dump())
    except ValidationError as ve:
        return jsonify({"error": "Schema validation failed", "details": json.loads(ve.json())}), 500
    except RuntimeError as re:
        # Error transparency for Pass B failures
        error_msg = str(re)
        if "Pass B failed" in error_msg:
            return jsonify({"error": "Structuring failed", "debug": error_msg}), 502
        return jsonify({"error": f"Generation failed: {re}"}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

@app.post("/render/html")
def render_article_to_html():
    """Convert ArticleTextOut to WordPress-ready HTML."""
    data = request.get_json(force=True)
    try:
        article = ArticleTextOut(**data)
    except ValidationError as ve:
        return jsonify({"error": "Invalid ArticleTextOut", "details": json.loads(ve.json())}), 400

    try:
        # Build WordPress-ready HTML
        html_parts = []
        
        # H1 title
        html_parts.append(f"<h1>{html.escape(article.h1)}</h1>")
        
        # Chapo (no extra <p> wrapper since render_markdown_to_html already adds <p>)
        if article.chapo_md:
            chapo_html = render_markdown_to_html(article.chapo_md)
            html_parts.append(chapo_html)
        
        # Sections
        for section in article.sections:
            html_parts.append(f"<h2>{html.escape(section.title)}</h2>")
            section_html = render_markdown_to_html(section.body_md)
            html_parts.append(section_html)
        
        # FAQ
        if article.faqs:
            html_parts.append("<h2>FAQ</h2>")
            for faq in article.faqs:
                html_parts.append(f"<h3>{html.escape(faq.question)}</h3>")
                answer_html = render_markdown_to_html(faq.answer_md)
                html_parts.append(answer_html)
        
        final_html = "\n\n".join(html_parts)
        
        # Convert text FAQs to HTML FAQs for legacy compatibility
        html_faqs = []
        for faq in article.faqs:
            answer_html = render_markdown_to_html(faq.answer_md)
            html_faqs.append(FAQ(question=faq.question, answer_html=answer_html))
        
        # Build ArticleOut for compatibility
        article_out = ArticleOut(
            h1=article.h1,
            slug=article.slug,
            meta_description=article.meta_description,
            outline=article.outline,
            html=final_html,
            keywords=article.keywords,
            faqs=html_faqs,
            sources=article.sources,
            schema_jsonld=article.schema_jsonld
        )
        
        return jsonify({
            "html": final_html,
            "article": article_out.model_dump(),
            "social": article.social.model_dump()
        })
        
    except Exception as e:
        return jsonify({"error": f"Rendering failed: {e}"}), 500

# ---------------------- UPDATED ARTICLE GENERATION (using new pipeline) ----------------------

@app.post("/article/next")
def generate_article_from_random_theme():
    """Pick a random unconsumed theme, mark consumed, generate structured article."""
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
        article_data, citations = gemini_generate_article_text(payload)
        normalized = _normalize_article_text_dict(article_data, payload, citations)
        article_text_out = ArticleTextOut(**normalized)
        
        return jsonify(
            {
                "theme": {
                    "id": chosen.get("id"),
                    "titre": chosen.get("titre"),
                    "mot_cle_principal": chosen.get("mot_cle_principal"),
                },
                "article": article_text_out.model_dump(),
            }
        )
    except ValidationError as ve:
        return jsonify({"error": "Schema validation failed", "details": json.loads(ve.json())}), 500
    except RuntimeError as re:
        # Error transparency for Pass B failures
        error_msg = str(re)
        if "Pass B failed" in error_msg:
            return jsonify({"error": "Structuring failed", "debug": error_msg}), 502
        return jsonify({"error": f"Generation failed: {re}"}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

@app.post("/article")
def generate_article_direct():
    """Generate structured article directly from explicit input (no theme pick)."""
    data = request.get_json(force=True)
    try:
        if not data.get("allow_domains") and DEFAULT_ALLOW_DOMAINS:
            data["allow_domains"] = DEFAULT_ALLOW_DOMAINS
        payload = ArticleIn(**data)
    except ValidationError as ve:
        return jsonify({"error": "Invalid payload", "details": json.loads(ve.json())}), 400

    try:
        article_data, citations = gemini_generate_article_text(payload)
        normalized = _normalize_article_text_dict(article_data, payload, citations)
        article_text_out = ArticleTextOut(**normalized)
        
        return jsonify(article_text_out.model_dump())
    except ValidationError as ve:
        return jsonify({"error": "Schema validation failed", "details": json.loads(ve.json())}), 500
    except RuntimeError as re:
        # Error transparency for Pass B failures
        error_msg = str(re)
        if "Pass B failed" in error_msg:
            return jsonify({"error": "Structuring failed", "debug": error_msg}), 502
        return jsonify({"error": f"Generation failed: {re}"}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

# ---------------------- LEGACY ENDPOINTS (for backward compatibility) ----------------------

@app.post("/article/legacy")
def generate_article_legacy():
    """Legacy endpoint: Generate article with HTML output (backward compatibility)."""
    data = request.get_json(force=True)
    try:
        if not data.get("allow_domains") and DEFAULT_ALLOW_DOMAINS:
            data["allow_domains"] = DEFAULT_ALLOW_DOMAINS
        payload = ArticleIn(**data)
    except ValidationError as ve:
        return jsonify({"error": "Invalid payload", "details": json.loads(ve.json())}), 400

    try:
        article = gemini_generate_article(payload)
        return jsonify({"html": article})
    except ValidationError as ve:
        return jsonify({"error": "Schema validation failed", "details": json.loads(ve.json())}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500

# ---------------------- UTILITY ENDPOINTS ----------------------

@app.get("/")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "French Insurance Content Generator",
        "version": "2.0.0",
        "pipeline": "2-step (text + render)"
    })

@app.get("/endpoints")
def list_endpoints():
    """List available endpoints."""
    return jsonify({
        "themes": {
            "POST /themes/import": "Import themes from CSV",
            "GET /themes/unconsumed": "List unconsumed themes",
            "POST /themes/reset": "Reset all themes"
        },
        "content_generation": {
            "POST /article/text": "Generate structured text article (new pipeline)",
            "POST /render/html": "Convert ArticleTextOut to WordPress HTML",
            "POST /article/next": "Pick random theme and generate structured article",
            "POST /article": "Generate structured article from input",
            "POST /article/legacy": "Generate HTML article (legacy pipeline)"
        },
        "utilities": {
            "GET /": "Health check",
            "GET /endpoints": "List all endpoints"
        }
    })

# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    # Install required dependencies if missing
    try:
        import markdown_it
        import bleach
        import tldextract
        import pandas
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install markdown-it-py bleach tldextract pandas")
        exit(1)
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8084)))