# ShieldX: AI-Based Cyberbullying Detection System

**ShieldX** is an ethical integrity security console that detects and moderates cyberbullying, toxic harassment, and offensive language in digital spaces. Founded and developed by **Yuvaraj Kumaran K** (III BSc AIML), this system uses Natural Language Processing (NLP), Machine Learning (ML), and Computer Vision/Optical Character Recognition (OCR) to intercept harmful interactions in text inputs and chat screenshots.

---

## 🌟 Key Features

1. **Optimized Threat Scanner**: Analyze manual text inputs or conversation screenshots using a single, highly tuned classifier.
2. **EasyOCR Image Bounding Box Visualization**: Extract dialogue directly from chat screenshots and visually overlay color-coded threat boxes (Red = Cyberbullying, Yellow = Offensive, Blue = Safe) directly on the image. *100% self-contained Python OCR (no external software/Tesseract installations required!)*
3. **Interactive Threat Sensitivity Slider**: Allow administrators to customize classification boundaries live (from 10% to 90% sensitivity) in both the scan console and simulator.
4. **Model Performance Hub**: Monitor accuracy, precision, recall, and F1-scores using a confusion matrix heatmap and trigger dynamic model retraining.
5. **Model Interpretation (Feature Weights)**: Displays the top 15 vocabulary coefficients indicating cyberbullying and safe classifications, making the model's decisions transparent.
6. **NLP Preprocessing Pipeline Visualizer**: Traces raw text normalization (lowercasing, cleaning, tokenization, and vectorization) step-by-step.
7. **SQLite Logging & Audits**: Persistent history logs in SQLite, with options to download CSV audits or purge DB archives.
8. **Print-Friendly PDF Reports**: Clean CSS print stylesheets for exporting neat PDF threat scan reports directly from the browser.

---

## 🛠️ Technology Stack & Role of Components

| Component | Technology | Description / Role |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript | Implements a responsive cyber-themed dark console utilizing glassmorphism, responsive grids, and micro-animations. |
| **Telemetry Visuals** | Chart.js | Renders live sentiment gauge speedometers, toxicity dimension radar charts, and comparative performance bar charts. |
| **Backend API** | Flask 2.3 (Python) | Micro-web framework handling routing, SQLite history interfaces, asynchronous retraining threads, and moderation API endpoints. |
| **Database** | SQLite 3 | Embedded database storing safety logs (time, source, raw content, AI verdict, and confidence scores). |
| **Computer Vision (OCR)**| EasyOCR (PyTorch-based) | Extracted text strings and coordinates from uploaded screenshot files. Coordinates are plotted using **Pillow (PIL)**. |
| **Sentiment Analytics** | NLTK VADER | Lexicon and rule-based sentiment engine computing compound polarity scores to flag hostile sentiment. |
| **Machine Learning** | Scikit-Learn | Handles text tokenization (TF-IDF) and runs the classification model (Logistic Regression). |
| **Data Pipelines** | Pandas & NumPy | Handled loading, partitioning, and cleaning the 47,000+ data rows in `Dataset.csv`. |
| **Plots & Metrics** | Matplotlib & Seaborn | Generated confusion matrix diagrams for the trained algorithm. |

---

## 🔬 Algorithms Explanation & Mathematical Backing

### 1. Feature Representation: TF-IDF Vectorization
Raw text cannot be processed directly by classifiers. It is vectorized using Term Frequency-Inverse Document Frequency (TF-IDF) to represent word importances:

$$\text{tf}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}$$

$$\text{idf}(t, D) = \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$

$$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$$

This maps sentences into a sparse matrix representing 5,000 unigram and bigram vocabularies.

### 2. Logistic Regression (LR)
A linear classifier that maps arbitrary inputs to probabilities between $0$ and $1$ using the Logistic/Sigmoid function:

$$P(y=\text{bullying} | x) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Weights ($\mathbf{w}$) are extracted to display "Feature Importances" in the dashboard. Regularization parameter $C=1.0$ is optimized alongside balanced class penalties to resolve data imbalance.

---

## 📥 Installation & Setup Instructions

Since `pytesseract` has been replaced by `easyocr` (pure Python), **you do not need to install Tesseract OCR or configure system PATH variables!** Everything is self-contained.

### Step 1: Clone or Copy the Repository
Place the folder in your directory of choice. Ensure `Dataset.csv` is present in the root folder.

### Step 2: Set Up Python Virtual Environment (Recommended)
Open Terminal/PowerShell inside the project directory:
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
Run pip to install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
Start the Flask development server:
```bash
python app.py
```
*Note: On the first startup, if models are not found in the `models/` directory, the backend will automatically call `train.py` to fit the classifier on `Dataset.csv`. This will take 10–20 seconds and pre-generate the model, confusion matrix, and metrics summary. EasyOCR will also download its English model weights on its first OCR scan.*

### Step 5: Access the Web Console
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## ☁️ Cloud Deployment Guidelines

This project can be deployed easily to cloud hosts since it runs on pure Python:

### Option A: Deploying to Render (Web Services)
1. Push the code to a private or public repository on **GitHub**.
2. Connect your GitHub account to **Render** (https://render.com).
3. Create a new **Web Service**:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py` (or use `gunicorn app:app`)
4. Set up an environment variable: `PYTHON_VERSION = 3.9` or above.
5. Render will automatically install dependencies, trigger `train.py` to compile the models, and host the web console.

### Option B: Deploying to Hugging Face Spaces (Streamlit/Docker or Gradio)
Hugging Face Spaces can host Flask applications using Docker:
1. Create a new space and select **Docker** as the SDK.
2. Provide a standard `Dockerfile` that installs python requirements and runs `EXPOSE 7860` (Hugging Face default port) and starts `app.py`.
3. Set the Flask app port to `7860` in `app.py` when running on Hugging Face:
   ```python
   app.run(host="0.0.0.0", port=7860)
   ```
