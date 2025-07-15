# 🛡️ RAKSHak: Privacy-First AI Interaction

### 🔐 *"Your Personal Redaction Shield Before Talking to AI"*

---

## 📖 Project Overview

**RAKSHak** is a powerful, privacy-centric **text redaction and sanitation system** built to shield sensitive user data from exposure to **Large Language Models (LLMs)** 🤖.

It acts as a **smart intermediary** — identifying and redacting **Personally Identifiable Information (PII)** from your text so you can interact with AI **without compromising your privacy** 🔏.

> 🎯 Engineered using a multi-layered AI stack, **RAKSHak** achieves over **89% accuracy** in PII detection, making it a **reliable privacy protector** in digital communication.

---

## ✨ Features

- 🔍 **Intelligent PII Redaction**  
  Detects and replaces sensitive entities like names, phone numbers, addresses, emails, and more.

- 🇮🇳 **Support for Indian PII**  
  Recognizes Aadhaar, PAN, and other India-specific identifiers.

- 🧠 **Contextual Data Protection**  
  Uses NER models to identify sensitive data even without clear patterns (like uncommon names or cities).

- 🏷️ **Text Sensitivity Classification**  
  Labels input as `confidential`, `personal`, or `public` using LLM-based text classification.

- 🔗 **Seamless LLM Integration**  
  Ensures **only redacted & safe** text is passed to AI models.

---

## 🧬 How It Works: Layered Protection System

RAKSHak uses a **3-layer defense strategy** 🧱 for robust privacy:

### 1️⃣ Pattern-Based Scanning  
🛠️ Utilizes **Presidio** to scan and tag PII with recognizable formats:  
- Phone → `[PHONE_NUMBER]`  
- Email → `[EMAIL_ADDRESS]`  
- Aadhaar, PAN → `[ID_NUMBER]`

### 2️⃣ Contextual Detection (NER)  
🤖 Employs **`ai4bharat/IndicNER`** from HuggingFace for Named Entity Recognition:  
- Detects names, locations, organizations, etc. without pattern reliance.

### 3️⃣ Sensitivity Classification  
🧠 Leverages **`facebook/bart-large-mnli`** to assign sensitivity tags:  
- `Confidential` 🛡️  
- `Personal` 🙋‍♂️  
- `Public` 🌍  

Only after these steps is the **sanitized text released** for AI use.

---

## 💻 Local Setup & Run

Want to try it out? Here’s how you can run it locally 🧪:

### 🔧 Prerequisites

- Python 3.8+
- pip (Python package installer)
- Git

### 🛠️ Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/RAKSHak.git

# 2. Move into the project directory
cd RAKSHak

# 3. Install the required dependencies
pip install -r requirements.txt
