# Misinformation-detection-in-TikTok
# Detecting Misinformation in TikTok Videos Using Machine Learning

## 📌 Project Overview
This project applies **machine learning** techniques to predict whether a TikTok video presents a **"claim"** (potential misinformation) or an **"opinion"** (subjective statement). 

With misinformation being a growing challenge on social media, this model aims to support content moderation efforts by identifying claims that require fact-checking. The **ultimate goal** is to help mitigate misinformation while ensuring fair and ethical content regulation.

---

## 📝 Business Understanding
### 🎯 Business Need and Modeling Objective
TikTok users can report videos they believe violate the platform's terms of service. Given the millions of videos created daily, manually reviewing all reports is impractical. Analysis indicates that videos violating terms of service are more likely to present claims rather than opinions. 

TikTok aims to build a **machine learning model** to classify videos into **claims** and **opinions** to assist human moderators:
- **Opinion videos**: Less likely to be flagged for review.
- **Claim videos**: Prioritized for human moderation based on reporting frequency.

A machine learning model would help filter and rank reported videos, ensuring human moderators focus on the most relevant cases.

### 🎯 Modeling Design and Target Variable
The dataset contains a binary column, `claim_status`, which indicates whether a video presents a **claim (1)** or an **opinion (0)**. 

This is a **binary classification task** where the goal is to predict the `claim_status` for each video.

### 📊 Evaluation Metric Selection
The model's errors can take two forms:
- **False Positives:** An opinion is misclassified as a claim.
- **False Negatives:** A claim is misclassified as an opinion.

Since it is **more harmful** to miss actual claims (false negatives) than to mistakenly flag opinions (false positives), the primary evaluation metric will be **Recall** (minimizing false negatives). 

---

## ⚖️ Ethical Considerations
### 🔍 Key Issues
- **Bias & Fairness**: The model should not disproportionately flag content based on factors such as language, region, or user behavior.
- **False Negatives**: If a misinformation claim is missed, harmful content might remain on the platform, violating the terms of service.
- **False Positives**: If an opinion is flagged incorrectly, it might lead to unnecessary human review, reducing efficiency.

To **minimize false negatives**, the model should favor **higher recall**, ensuring potentially harmful claims are reviewed even at the expense of some false positives.

---

## 🏗️ Modeling Workflow & Selection Process
The dataset contains approximately **20,000 videos**, sufficient for robust model training. The workflow follows these steps:

1. **Data Preparation & Splitting**:  
   - Divide data into **training (60%)**, **validation (20%)**, and **test (20%)** sets.
2. **Feature Engineering**: 
   - Text processing (tokenization, TF-IDF, embedding techniques).
   - Metadata feature selection (engagement metrics, reporting history).
3. **Model Training & Hyperparameter Tuning**:  
   - Train multiple models (Logistic Regression, Decision Trees, Random Forests, SVM, Deep Learning models).
   - Tune hyperparameters using cross-validation.
4. **Model Evaluation & Selection**:  
   - Select the best-performing model based on **Recall**.
5. **Final Assessment**:  
   - Evaluate the champion model’s performance on the test set.

---

## 🗂️ Data Understanding & Feature Engineering
### 📂 Dataset
- **Features**: Video metadata, engagement metrics (likes, shares, comments), content features.
- **Target Variable**: `claim_status` (1 = Claim, 0 = Opinion).

### 🔎 Feature Engineering Steps
- **Text Processing**: Tokenization, stop-word removal, TF-IDF vectorization.
- **Metadata Processing**: Normalization of engagement metrics.
- **Categorical Encoding**: Encoding categorical attributes for model compatibility.

---

## 🤖 Modeling & Evaluation
### 🏆 Machine Learning Models Used
1. **Logistic Regression**
2. **Decision Trees**
3. **Random Forests**
4. **Support Vector Machines (SVM)**
5. **Deep Learning (optional)**

### 📊 Evaluation Metrics
- **Recall**: Primary metric to minimize false negatives.
- **Precision & F1-Score**: To balance recall and specificity.
- **ROC-AUC Score**: Overall model performance.

---

## 🎯 Conclusion & Recommendations
### ✅ Key Findings
- Certain keywords and linguistic structures correlate highly with misinformation claims.
- Engagement-based features help in predicting claim likelihood.
- The best-performing model balances recall and false positive rates.

### 🚀 Next Steps
- **Enhance training data**: Include more fact-checked videos.
- **Bias mitigation**: Ensure balanced representation across different video types.
- **Human-in-the-loop moderation**: Combine automated detection with human verification.



---

## 🙌 Acknowledgments
- **Libraries Used**: Pandas, NumPy, Scikit-learn, Matplotlib.


---


