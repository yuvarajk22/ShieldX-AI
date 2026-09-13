from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import os
import re
import html
import json
import pickle
import numpy as np
import sqlite3
import io
import csv
import base64
import threading
from datetime import datetime
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import easyocr
import nltk

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB max to prevent Denial of Service

# Enable CORS (Cross-Origin Resource Sharing) headers dynamically
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# Initialize NLTK VADER Sentiment
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize EasyOCR Reader (English, CPU mode)
easyocr_reader = easyocr.Reader(['en'], gpu=False)

DB_NAME = "history.db"

# Offensive and Cyberbullying lexicons for hybrid validation
CYBER_WORDS = [
    "hate", "kill", "die", "ugly", "loser", "worthless",
    "stupid", "idiot", "dumb", "fool", "useless", "go away",
    "nobody likes you", "leave", "shut up",
    "you are so stupid", "you are a loser", "please stop bullying me",
    "nobody likes you go away", "everyone hates you", "kill yourself",
    "suicide", "murder", "bitch", "whore", "slut", "cunt", "faggot"
]

OFFENSIVE_WORDS = [
    "stupid", "idiot", "dumb", "fool", "ugly", "loser", "shut up",
    "crap", "bastard", "asshole", "bitch"
]

# Model states (Single optimized model)
model = None
vectorizer = None
is_training = False
training_lock = threading.Lock()

# ================= MODEL LOADER & DYNAMIC INITIATOR =================
def load_models():
    global model, vectorizer
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/logistic_regression.pkl"
    vect_path = "models/tfidf_vectorizer.pkl"
    
    models_missing = not os.path.exists(model_path) or not os.path.exists(vect_path)
            
    if models_missing:
        print("Model binaries missing. Pre-training optimized model on Dataset.csv...")
        try:
            import train
            train.train_optimized_model()
        except Exception as e:
            print("Failed to auto-train model on startup:", e)
            
    # Load model and vectorizer
    try:
        if os.path.exists(vect_path) and os.path.exists(model_path):
            with open(vect_path, "rb") as f:
                vectorizer = pickle.load(f)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("Optimized ML model loaded successfully.")
    except Exception as e:
        print("Error loading model binaries:", e)

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            input_type TEXT,
            source_name TEXT,
            content TEXT,
            result TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def save_history(input_type, source_name, content, result, confidence):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history (time, input_type, source_name, content, result, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        input_type,
        source_name,
        content,
        result,
        confidence
    ))
    conn.commit()
    conn.close()


def get_stats():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM history")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE result LIKE '%SAFE%'")
    safe = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE result LIKE '%OFFENSIVE%'")
    offensive = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM history WHERE result LIKE '%CYBERBULLYING%'")
    cyber = cur.fetchone()[0]

    cur.execute("""
        SELECT time, input_type, source_name, result, confidence
        FROM history
        ORDER BY id DESC
        LIMIT 10
    """)
    recent = cur.fetchall()
    conn.close()

    return {
        "total": total,
        "safe": safe,
        "offensive": offensive,
        "cyber": cyber,
        "recent": recent
    }


def clear_all_history():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()


# ================= TEXT HELPERS =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text):
    cleaned = clean_text(text)
    found = []
    for phrase in CYBER_WORDS + OFFENSIVE_WORDS:
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, cleaned):
            if phrase not in found:
                found.append(phrase)
    return found


def highlight_keywords(text, keywords):
    # Prevent Cross-Site Scripting (XSS) by escaping raw input text before inserting mark tags
    escaped_text = html.escape(text)
    highlighted = escaped_text
    for word in sorted(keywords, key=len, reverse=True):
        escaped_word = html.escape(word)
        pattern = re.compile(rf"(\b{re.escape(escaped_word)}\b)", re.IGNORECASE)
        highlighted = pattern.sub(r"<mark class='highlight-word'>\1</mark>", highlighted)
    return highlighted


# ================= TEXT / ML PREDICTION =================
def predict_text_hybrid(text, sensitivity=50):
    cleaned = clean_text(text)
    if not cleaned:
        return "🟢 SAFE CONTENT", 90.0, "The text is empty.", [], "safe"

    # 1. VADER Sentiment Analysis
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    # 2. Fallback if vectorizer or model is missing
    if vectorizer is None or model is None:
        found_keywords = extract_keywords(text)
        if any(w in CYBER_WORDS for w in found_keywords):
            return "🔴 CYBERBULLYING", 85.0, "Lexicon match (ML model missing).", found_keywords, "danger"
        elif any(w in OFFENSIVE_WORDS for w in found_keywords):
            return "🟠 OFFENSIVE CONTENT", 80.0, "Lexicon match (ML model missing).", found_keywords, "warn"
        else:
            return "🟢 SAFE CONTENT", 75.0, "Lexicon analysis (ML model missing).", [], "safe"

    # 3. ML Prediction
    vect = vectorizer.transform([cleaned])
    
    # Calculate threshold based on sensitivity (slider: 10 to 90)
    # Sensitivity translates to probability threshold: threshold = 1.0 - (sensitivity / 100.0)
    threshold = 1.0 - (float(sensitivity) / 100.0)
    
    # Adjust threshold further if text contains high-threat indicators
    threat_terms = ["kill", "die", "suicide", "murder", "nigger", "faggot", "kill yourself", "worthless"]
    has_threat_term = any(t in cleaned for t in threat_terms)
    if has_threat_term:
        threshold = max(0.1, threshold - 0.15)  # Make it 15% more sensitive

    # Predict probability
    prob = model.predict_proba(vect)
    classes_list = list(model.classes_)
    
    bullying_idx = classes_list.index('bullying')
    bullying_prob = prob[0][bullying_idx]
    
    # Deterministic safety guardrail for extreme self-harm or hate-speech threats
    severe_threats = ["kill yourself", "suicide", "nigger", "faggot", "go kill yourself", "murder you", "die ugly"]
    if any(t in cleaned for t in severe_threats):
        bullying_prob = max(bullying_prob, 0.95)
    
    if bullying_prob >= threshold:
        pred = "bullying"
        confidence = round(float(bullying_prob) * 100, 2)
    else:
        pred = "not_bullying"
        not_bullying_idx = classes_list.index('not_bullying')
        confidence = round(float(prob[0][not_bullying_idx]) * 100, 2)

    # 4. Keyword/Profanity Matching
    found_keywords = extract_keywords(text)

    # 5. Hybrid Decision Pipeline
    if pred == "bullying":
        result = "🔴 CYBERBULLYING"
        explanation = f"ML model flagged bullying signatures (Bullying score {round(bullying_prob*100,1)}% is above the sensitivity threshold of {round(threshold*100,1)}%)."
        result_class = "danger"
    else:
        # Check if profanity or highly negative sentiment elevates it to offensive
        is_offensive = False
        offensive_reason = ""
        
        found_offensive = [w for w in OFFENSIVE_WORDS if w in found_keywords]
        if found_offensive:
            is_offensive = True
            offensive_reason = f"Contains flagged profanity: {', '.join(found_offensive)}."
            
        if compound <= -0.4:
            is_offensive = True
            if offensive_reason:
                offensive_reason += " Also shows negative sentiment."
            else:
                offensive_reason = "Sentiment analysis identified hostile vocabulary."
            
        if is_offensive:
            result = "🟠 OFFENSIVE CONTENT"
            explanation = f"AI classified text as non-bullying, but sentiment/profanity filter elevated it. Reason: {offensive_reason}"
            result_class = "warn"
        else:
            result = "🟢 SAFE CONTENT"
            explanation = "AI models and polarity checks classified this text as safe, positive, or neutral."
            result_class = "safe"

    return result, confidence, explanation, found_keywords, result_class


def calculate_toxicity_breakdown(text, confidence, result_class):
    cleaned = clean_text(text)
    
    threat_terms = ["kill", "die", "suicide", "murder", "hurt", "shoot", "hang"]
    insult_terms = ["ugly", "loser", "worthless", "stupid", "idiot", "dumb", "fool", "useless", "hate", "pathetic"]
    profanity_terms = ["fuck", "shit", "bitch", "asshole", "crap", "bastard", "whore", "slut", "cunt"]
    
    threat_score = 0
    insult_score = 0
    profanity_score = 0
    
    for w in threat_terms:
        if w in cleaned:
            threat_score += 40
    for w in insult_terms:
        if w in cleaned:
            insult_score += 30
    for w in profanity_terms:
        if w in cleaned:
            profanity_score += 45
            
    if result_class == "danger":
        threat_score += 35
        insult_score += 40
        profanity_score += 15
    elif result_class == "warn":
        insult_score += 25
        profanity_score += 30
        
    scores = sia.polarity_scores(text)
    neg = scores["neg"]
    agg_score = int(neg * 100)
    
    if result_class == "danger":
        agg_score = max(agg_score, 80)
    elif result_class == "warn":
        agg_score = max(agg_score, 50)
        
    return {
        "insult": min(insult_score, 100),
        "profanity": min(profanity_score, 100),
        "threat": min(threat_score, 100),
        "aggression": min(agg_score, 100)
    }


# ================= OCR HELPER (PURE EASYOCR) =================
def should_ignore_line(line):
    line = clean_text(line)
    ignore_words = [
        "online", "type a message", "message",
        "call", "video", "status", "archive", "active"
    ]
    for word in ignore_words:
        if word in line:
            return True
    if len(line) < 4:
        return True
    if re.fullmatch(r"\d+", line):
        return True
    return False


def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_easyocr_boxed_image(file_obj, sensitivity=50):
    img = Image.open(file_obj).convert("RGB")
    w, h = img.size
    
    # Scale up for higher OCR accuracy
    img_large = img.resize((w * 2, h * 2))
    img_np = np.array(img_large)
    
    results = easyocr_reader.readtext(img_np)
    
    draw_img = img_large.copy()
    draw = ImageDraw.Draw(draw_img)
    
    boxes = []
    extracted_lines = []
    
    for bbox, text, conf in results:
        cleaned = clean_text(text)
        if not cleaned or conf < 0.25:
            continue
            
        if should_ignore_line(cleaned):
            continue
            
        extracted_lines.append(text)
        pred, _, _, _, result_class = predict_text_hybrid(text, sensitivity)
        
        # Color styles
        if result_class == "danger":
            color = "#ff647e"
            outline_rgb = (255, 100, 126)
        elif result_class == "warn":
            color = "#ffd166"
            outline_rgb = (255, 209, 102)
        else:
            color = "#58b6ff"
            outline_rgb = (88, 182, 255)
            
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        xmin, ymin = min(xs), min(ys)
        xmax, ymax = max(xs), max(ys)
        
        # Draw box outline
        draw.rectangle([xmin, ymin, xmax, ymax], outline=outline_rgb, width=3)
        
        boxes.append({
            "x": int(xmin / 2),
            "y": int(ymin / 2),
            "w": int((xmax - xmin) / 2),
            "h": int((ymax - ymin) / 2),
            "text": text,
            "color": color,
            "label": "harmful" if result_class == "danger" else ("warning" if result_class == "warn" else "normal")
        })
        
    draw_img_original = draw_img.resize((w, h))
    base64_img = pil_to_base64(draw_img_original)
    
    combined_text = "\n".join(extracted_lines)
    return base64_img, boxes, combined_text


def analyze_ocr_text(extracted_text, sensitivity=50):
    raw_lines = extracted_text.splitlines()

    accepted_lines = []
    removed_lines = []
    line_results = []
    keywords = []

    for raw in raw_lines:
        cleaned = clean_text(raw)
        if not cleaned:
            continue

        if should_ignore_line(cleaned):
            removed_lines.append(raw)
            continue

        accepted_lines.append(raw)

    if not accepted_lines:
        return (
            "🟢 SAFE CONTENT",
            90.0,
            [],
            removed_lines,
            [],
            extracted_text,
            "The extracted content contains no readable text analysis lines.",
            "safe"
        )

    # Classify each line
    bully_count = 0
    warn_count = 0
    for line in accepted_lines:
        pred, conf, _, line_keys, r_class = predict_text_hybrid(line, sensitivity)
        line_results.append((line, pred, conf))
        keywords.extend(line_keys)
        if r_class == "danger":
            bully_count += 1
        elif r_class == "warn":
            warn_count += 1

    keywords = list(set(keywords))
    combined_text = " ".join(accepted_lines)
    result, confidence, explanation, _, result_class = predict_text_hybrid(combined_text, sensitivity)
    
    if bully_count > 0:
        result = "🔴 CYBERBULLYING"
        result_class = "danger"
        explanation = f"Detected {bully_count} highly abusive bullying patterns in the conversation."
    elif warn_count > 0 and result_class == "safe":
        result = "🟠 OFFENSIVE CONTENT"
        result_class = "warn"
        explanation = f"Conversation is mostly safe, but contains {warn_count} lines with offensive speech."

    highlighted_text = highlight_keywords(extracted_text, keywords) if keywords else extracted_text

    return (
        result,
        confidence,
        line_results,
        removed_lines,
        keywords,
        highlighted_text,
        explanation,
        result_class
    )


# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    explanation = None
    error = None
    user_text = ""
    extracted_text = ""
    line_results = []
    removed_lines = []
    uploaded_filename = ""
    keywords = []
    highlighted_text = ""
    result_class = ""
    ocr_boxed_image = ""
    ocr_boxes = []
    sensitivity = 50
    
    nlp_steps = {}
    vader_sentiment = {"compound": 0, "label": "Neutral"}
    toxicity_breakdown = {"insult": 0, "profanity": 0, "threat": 0, "aggression": 0}

    metrics_path = "static/model_metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()[:1000]
        sensitivity = int(request.form.get("sensitivity", 50))
        file = request.files.get("image")

        if file and file.filename:
            uploaded_filename = file.filename

        if user_text:
            result, confidence, explanation, keywords, result_class = predict_text_hybrid(user_text, sensitivity)
            save_history("Text Scan", "Manual Text Input", user_text, result, confidence)
            
            nlp_steps["step_1"] = user_text
            nlp_steps["step_2"] = clean_text(user_text)
            nlp_steps["step_3"] = " ".join([w for w in clean_text(user_text).split() if len(w) > 2])
            
            v_score = sia.polarity_scores(user_text)
            vader_sentiment["compound"] = v_score["compound"]
            if v_score["compound"] >= 0.05:
                vader_sentiment["label"] = "Positive"
            elif v_score["compound"] <= -0.05:
                vader_sentiment["label"] = "Negative"
            else:
                vader_sentiment["label"] = "Neutral"
                
            toxicity_breakdown = calculate_toxicity_breakdown(user_text, confidence, result_class)
            highlighted_text = highlight_keywords(user_text, keywords) if keywords else user_text

        elif file and file.filename != "":
            try:
                ocr_boxed_image, ocr_boxes, extracted_text = get_easyocr_boxed_image(file, sensitivity)

                if extracted_text:
                    (
                        result,
                        confidence,
                        line_results,
                        removed_lines,
                        keywords,
                        highlighted_text,
                        explanation,
                        result_class
                    ) = analyze_ocr_text(extracted_text, sensitivity)

                    save_history("Image OCR Scan", uploaded_filename, extracted_text, result, confidence)
                    
                    nlp_steps["step_1"] = extracted_text
                    nlp_steps["step_2"] = clean_text(extracted_text)
                    nlp_steps["step_3"] = " ".join([w for w in clean_text(extracted_text).split() if len(w) > 2])
                    
                    v_score = sia.polarity_scores(extracted_text)
                    vader_sentiment["compound"] = v_score["compound"]
                    if v_score["compound"] >= 0.05:
                        vader_sentiment["label"] = "Positive"
                    elif v_score["compound"] <= -0.05:
                        vader_sentiment["label"] = "Negative"
                    else:
                        vader_sentiment["label"] = "Neutral"
                        
                    toxicity_breakdown = calculate_toxicity_breakdown(extracted_text, confidence, result_class)
                else:
                    error = "No readable text found in the uploaded image."

            except Exception as e:
                import traceback
                traceback.print_exc()
                error = f"Unable to read text from image: {e}"

        else:
            error = "Please enter text or upload an image."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        explanation=explanation,
        error=error,
        user_text=user_text,
        extracted_text=extracted_text,
        line_results=line_results,
        removed_lines=removed_lines,
        uploaded_filename=uploaded_filename,
        stats=get_stats(),
        keywords=keywords,
        highlighted_text=highlighted_text,
        result_class=result_class,
        ocr_boxed_image=ocr_boxed_image,
        ocr_boxes=ocr_boxes,
        sensitivity=sensitivity,
        nlp_steps=nlp_steps,
        vader_sentiment=vader_sentiment,
        toxicity_breakdown=toxicity_breakdown,
        metrics=metrics,
        is_training=is_training
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():
    clear_all_history()
    return redirect(url_for("home"))


@app.route("/download-report")
def download_report():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT time, input_type, source_name, content, result, confidence
        FROM history
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "Input Type", "Source Name", "Content", "Result", "Confidence"])
    writer.writerows(rows)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    return send_file(
        mem,
        as_attachment=True,
        download_name="cyberbullying_report.csv",
        mimetype="text/csv"
    )


@app.route("/api/moderate", methods=["POST"])
def api_moderate():
    data = request.json or {}
    text = data.get("text", "").strip()[:1000]
    sensitivity = int(data.get("sensitivity", 50))
    
    if not text:
        return jsonify({"decision": "allow", "result_class": "safe"})
        
    result, confidence, explanation, keywords, result_class = predict_text_hybrid(text, sensitivity)
    
    if result_class == "danger":
        decision = "block"
    elif result_class == "warn":
        decision = "flag"
    else:
        decision = "allow"
        
    scores = sia.polarity_scores(text)
    
    return jsonify({
        "decision": decision,
        "result": result,
        "confidence": confidence,
        "explanation": explanation,
        "result_class": result_class,
        "sentiment_score": scores["compound"]
    })


@app.route("/api/moderate-image", methods=["POST"])
def api_moderate_image():
    file = request.files.get("image")
    sensitivity = int(request.form.get("sensitivity", 50))
    
    if not file or not file.filename:
        return jsonify({"error": "No image uploaded"}), 400
        
    try:
        ocr_boxed_image, ocr_boxes, extracted_text = get_easyocr_boxed_image(file, sensitivity)
        
        if extracted_text:
            (
                result,
                confidence,
                line_results,
                removed_lines,
                keywords,
                highlighted_text,
                explanation,
                result_class
            ) = analyze_ocr_text(extracted_text, sensitivity)
            
            save_history("Image OCR Scan", file.filename, extracted_text, result, confidence)
            
            v_score = sia.polarity_scores(extracted_text)
            vader_sentiment = {
                "compound": v_score["compound"],
                "label": "Positive" if v_score["compound"] >= 0.05 else ("Negative" if v_score["compound"] <= -0.05 else "Neutral")
            }
            
            toxicity_breakdown = calculate_toxicity_breakdown(extracted_text, confidence, result_class)
            
            return jsonify({
                "ocr_boxed_image": ocr_boxed_image,
                "ocr_boxes": ocr_boxes,
                "result": result,
                "confidence": confidence,
                "explanation": explanation,
                "result_class": result_class,
                "vader_sentiment": vader_sentiment,
                "toxicity_breakdown": toxicity_breakdown,
                "highlighted_text": highlighted_text,
                "line_results": [{"line": l, "pred": p, "conf": c} for l, p, c in line_results]
            })
        else:
            return jsonify({
                "result": "🟢 SAFE CONTENT",
                "confidence": 90.0,
                "explanation": "No readable text found in image.",
                "result_class": "safe",
                "ocr_boxed_image": ocr_boxed_image,
                "ocr_boxes": [],
                "highlighted_text": "",
                "vader_sentiment": {"compound": 0, "label": "Neutral"},
                "toxicity_breakdown": {"insult": 0, "profanity": 0, "threat": 0, "aggression": 0},
                "line_results": []
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats_json():
    stats = get_stats()
    recent_serialized = []
    for row in stats["recent"]:
        recent_serialized.append({
            "time": row[0],
            "input_type": row[1],
            "source_name": row[2],
            "result": row[3],
            "confidence": row[4]
        })
        
    metrics_path = "static/model_metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
            
    return jsonify({
        "total": stats["total"],
        "safe": stats["safe"],
        "offensive": stats["offensive"],
        "cyber": stats["cyber"],
        "recent": recent_serialized,
        "metrics": metrics
    })


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    clear_all_history()
    return jsonify({"status": "success"})


@app.route("/retrain", methods=["POST"])
def retrain():
    global is_training
    if is_training:
        return jsonify({"status": "error", "message": "Training is already in progress."}), 400
        
    def worker():
        global is_training
        with training_lock:
            is_training = True
            try:
                import train
                train.train_optimized_model()
                load_models()
            except Exception as e:
                print("Error during background retraining:", e)
            finally:
                is_training = False

    threading.Thread(target=worker).start()
    return jsonify({"status": "success", "message": "Retraining started."})


@app.route("/retrain-status")
def retrain_status():
    global is_training
    return jsonify({
        "is_training": is_training
    })


if __name__ == "__main__":
    init_db()
    load_models()
    app.run(debug=True)