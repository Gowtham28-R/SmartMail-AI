# ============================================
# SmartMail AI - Threat Statistics & Analytics
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "email_logs.csv"

# ---------------- Load Logs ----------------
try:
    df = pd.read_csv(LOG_FILE)
except:
    print("❌ No logs found. Run smartmail_ai.py and scan some emails first.")
    exit()

print("\n===== SMARTMAIL AI – THREAT ANALYTICS =====")
print("Total emails scanned:", len(df))
print("\nCategory counts:\n", df['category'].value_counts())
print("\nSeverity counts:\n", df['severity'].value_counts())
print("\nSpam/Ham counts:\n", df['result'].value_counts())

# ---------------- Chart 1: Email Categories ----------------
plt.figure()
df['category'].value_counts().plot(kind='bar', title="Emails by Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------- Chart 2: Threat Severity ----------------
plt.figure()
df['severity'].value_counts().plot(kind='bar', title="Threat Severity Levels")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------- Chart 3: Spam vs Ham ----------------
plt.figure()
df['result'].value_counts().plot(kind='bar', title="Spam vs Ham")
plt.xlabel("Result")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------- Chart 4: Risk Score Trend ----------------
plt.figure()
plt.plot(df['risk_score'], marker='o')
plt.title("Threat Risk Score Trend")
plt.xlabel("Email Index")
plt.ylabel("Risk Score")
plt.tight_layout()
plt.show()
