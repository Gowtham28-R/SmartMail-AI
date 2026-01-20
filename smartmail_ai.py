# =====================================================
# SmartMail AI - COMPLETE EMAIL SECURITY SYSTEM
# High Accuracy + Explainable + Categories + Logging
# FINAL WORKING CYBER EDITION
# =====================================================

import pandas as pd
import numpy as np
import re, os, nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- Color Engine ----------------
class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    ORANGE = "\033[38;5;208m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# ---------------- UI Engine ----------------
def ui_line(char="─", n=78): return char * n

def ui_box(title):
    print(f"\n{C.BLUE}{ui_line('═')}")
    print(f"│ {title.center(74)} │")
    print(f"{ui_line('═')}{C.RESET}")

def ui_section(title):
    print(f"\n{C.PURPLE}{ui_line('─')}")
    print(f"▶ {title}")
    print(f"{ui_line('─')}{C.RESET}")

# ---------------- NLTK ----------------
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

# ---------------- Load Dataset ----------------
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['label','text']

# ---------------- Text Cleaning ----------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " link ", text)
    text = re.sub('[^a-z0-9₹$%]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# ---------------- Feature Extraction ----------------
word_vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2), min_df=2)
char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3)

X = hstack([
    word_vectorizer.fit_transform(df['clean_text']),
    char_vectorizer.fit_transform(df['clean_text'])
])

y = df['label'].map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Models ----------------
nb = MultinomialNB()
lr = LogisticRegression(max_iter=2000, class_weight='balanced')

ensemble = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr)],
    voting='soft',
    weights=[1,2]
)

# ---------------- Cross Validation ----------------
print("\nRunning cross-validation...")
scores = cross_val_score(ensemble, X, y, cv=5, scoring='f1')
print("Cross-validated F1 Score:", scores.mean())

# ---------------- Train ----------------
ensemble.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = ensemble.predict(X_test)
print("\n===== FINAL MODEL RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SmartMail AI")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------- Save ----------------
joblib.dump(ensemble, "smartmail_model.pkl")
joblib.dump(word_vectorizer, "word_vectorizer.pkl")
joblib.dump(char_vectorizer, "char_vectorizer.pkl")

# ---------------- Extract spam words ----------------
lr_model = ensemble.named_estimators_['lr']
feature_names = word_vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0][:len(feature_names)]
top_spam_idx = np.argsort(coefs)[-300:]

spam_feature_words = set()
for i in top_spam_idx:
    w = feature_names[i]
    if w.isalpha() and len(w) > 2:
        spam_feature_words.add(w)

# ---------------- Threat Risk ----------------
def spam_rule_score(text):
    score = 0
    t = text.lower()
    if any(w in t for w in ["win","won","free","prize","urgent","click","offer","money",
                            "lottery","reward","claim","verify","account","bank",
                            "bonus","credit","suspended","alert"]): score += 3
    if "http" in t or "www" in t: score += 3
    if re.search(r"(₹|\$|%|\b\d{4,}\b)", t): score += 2
    if t.count("!") >= 2: score += 2
    return min(score,10)

def threat_severity(risk):
    if risk >= 8: return "🔴 CRITICAL"
    elif risk >= 6: return "🟠 HIGH"
    elif risk >= 3: return "🟡 MEDIUM"
    else: return "🟢 LOW"

# ---------------- Phishing Detection ----------------
def detect_links(text):
    return re.findall(r"(https?://\S+|www\.\S+)", text)

def phishing_check(text):
    links = detect_links(text)
    suspicious = [l for l in links if any(x in l.lower() for x in ["login","verify","bank","secure","update","account"])]
    return links, suspicious

# ---------------- Explainability (YELLOW HIGHLIGHT) ----------------
def highlight_spam_words_model(text):
    highlighted = text
    found = set()
    for word in spam_feature_words:
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
        if re.search(pattern, highlighted):
            found.add(word)
            highlighted = re.sub(pattern, f"{C.YELLOW}[\\1]{C.RESET}", highlighted)
    return highlighted, list(found)

# ---------------- Confidence Fusion ----------------
def final_confidence(spam_conf, risk_score, final_pred):
    fused = 0.7*spam_conf + 0.3*(risk_score/10)
    if final_pred == 1 and fused < 0.6: fused = 0.6 + fused*0.4
    if final_pred == 0 and fused > 0.4: fused = fused*0.4
    return min(max(fused,0),1)

# ---------------- Category Engine ----------------
def classify_email_type(text, spam_words, phishing_links, risk_score):
    t = text.lower()
    promo_words = ["sale","offer","discount","deal","buy","shop","limited","promo"]
    scam_words = ["win","won","lottery","prize","reward","claim","money","cash","bitcoin"]
    phishing_words = ["verify","login","account","bank","secure","password","update","suspended"]

    if phishing_links or any(w in t for w in phishing_words): return "🎣 PHISHING"
    if any(w in t for w in scam_words) and re.search(r"(₹|\$|\b\d{4,}\b)", t): return "💰 SCAM"
    if any(w in t for w in promo_words): return "🟡 PROMOTION"
    if risk_score >= 4 or spam_words: return "🚨 SPAM"
    return "🟢 SAFE MAIL"

# ---------------- LOGGING SYSTEM (SAFE) ----------------
LOG_FILE = "email_logs.csv"

def log_email(text, label, category, confidence, risk, severity, reasons):
    try:
        log_data = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email": text.replace("\n"," "),
            "result": label,
            "category": category,
            "confidence": confidence,
            "risk_score": risk,
            "severity": severity,
            "reasons": " | ".join(reasons)
        }

        df_log = pd.DataFrame([log_data])

        if os.path.exists(LOG_FILE):
            df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            df_log.to_csv(LOG_FILE, index=False)

    except PermissionError:
        print(f"{C.RED}⚠ LOGGING FAILED: Close email_logs.csv to enable logging.{C.RESET}")

# ---------------- Prediction ----------------
def predict_email_advanced(text):

    clean = clean_text(text)
    Xf = hstack([word_vectorizer.transform([clean]), char_vectorizer.transform([clean])])

    probs = ensemble.predict_proba(Xf)[0]
    spam_conf = probs[1]

    risk = spam_rule_score(text)
    model_pred = 1 if spam_conf >= 0.5 else 0
    final_pred = 1 if risk >= 4 else model_pred

    final_conf = final_confidence(spam_conf, risk, final_pred)

    highlighted, spam_words = highlight_spam_words_model(text)
    links, suspicious_links = phishing_check(text)
    email_type = classify_email_type(text, spam_words, suspicious_links, risk)

    reasons = []
    if spam_words: reasons.append("Model detected spam-associated words")
    if links: reasons.append("Links found in email")
    if suspicious_links: reasons.append("Phishing-style links detected")
    if re.search(r"(₹|\$|%|\b\d{4,}\b)", text): reasons.append("Money-related patterns detected")
    if text.count("!") >= 2: reasons.append("Urgency patterns detected")
    if not reasons: reasons.append("Text structure matches legitimate emails")

    if final_pred == 1:
        label = "SPAM 🚫"
        confidence = round(final_conf*100,2)
    else:
        label = "HAM ✅"
        confidence = round((1-final_conf)*100,2)

    return label, email_type, confidence, risk, highlighted, spam_words, links, suspicious_links, reasons

# ---------------- Real-Time System ----------------
ui_box("🛡 SMARTMAIL AI – EMAIL SECURITY SYSTEM")

while True:
    print("\nPaste full email below. Type END on a new line to finish (or EXIT to quit):")

    lines = []
    while True:
        user_line = input()
        if user_line.strip().upper() == "END": break
        if user_line.strip().upper() == "EXIT": exit()
        lines.append(user_line)

    msg = "\n".join(lines)

    label, email_type, confidence, risk, highlighted, spam_words, links, phishing, reasons = predict_email_advanced(msg)
    severity = threat_severity(risk)

    print(f"\n{C.BLUE}{ui_line('═')}{C.RESET}")
    print(f"{C.RED if 'SPAM' in label else C.GREEN}📛 RESULT        : {label}{C.RESET}")
    print(f"{C.CYAN}📌 CATEGORY      : {email_type}{C.RESET}")
    print(f"{C.PURPLE}🎯 CONFIDENCE    : {confidence}%{C.RESET}")
    print(f"{C.ORANGE}🔥 SEVERITY      : {severity}{C.RESET}")
    print(f"{C.RED if risk>=6 else C.YELLOW if risk>=3 else C.GREEN}⚠ THREAT SCORE  : {risk}/10{C.RESET}")

    ui_section("🧠 AI ANALYSIS & REASONS")
    for r in reasons: print(f"{C.GREEN}✔ {r}{C.RESET}")

    if spam_words:
        ui_section("⚠ DETECTED SPAM TOKENS")
        print(f"{C.ORANGE}Words:{C.RESET}", spam_words)
        print("\n✉ HIGHLIGHTED EMAIL:")
        print(ui_line()); print(highlighted); print(ui_line())

    if links:
        ui_section("🔗 LINKS FOUND")
        for l in links: print("➜", l)

    if phishing:
        ui_section("🎣 POSSIBLE PHISHING LINKS")
        for p in phishing: print("⚠", p)

    # --------- LOG EMAIL ----------
    log_email(msg, label, email_type, confidence, risk, severity, reasons)

    print(f"\n{C.BLUE}{ui_line('═')}{C.RESET}")
