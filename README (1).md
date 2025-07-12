
# 💼 Salary Prediction App

This is a **machine learning web application** built using **Streamlit** that predicts whether a person earns **more than $50K or less** based on demographic and work-related inputs. The model is trained on the **UCI Adult Census Dataset**.

---

## 🧠 Features

- Predicts income class: `<=50K` or `>50K`
- Uses a machine learning pipeline serialized via `pickle`
- Intuitive web interface for user input via **Streamlit**

---

## 🗂 Project Structure

```
.
├── app.py                               # Main Streamlit web app
├── final-pipeline-salary-prediction.pkl # Trained ML pipeline
├── requirements.txt                    # Required Python packages
├── Salary_Prediction_Project.ipynb     # Notebook for data analysis & model training
└── Salary_Prediction_Project_part_2.ipynb # Continuation notebook for further tuning/prediction
```

---

## 🔍 Concepts Used

### 1. **Data Preprocessing**
- Handled missing values (e.g., `?` placeholders)
- Categorical encoding using techniques like **OneHotEncoding**
- Feature scaling and selection

### 2. **Modeling**
- **Pipeline** approach using `sklearn.pipeline.Pipeline` for chaining preprocessing and model
- Model(s) used likely include **Logistic Regression**, **RandomForest**, or **XGBoost**

### 3. **Model Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score
- Used **train-test split** and **cross-validation**

### 4. **Web Deployment**
- User input fields (age, education, workclass, etc.) via **Streamlit widgets**
- Prediction triggered via a button click
- Model inference via deserialization (`pickle.load`)

---

## 🖥️ How to Run the App

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd <project-directory>
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

---

## 📥 Input Fields

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Hours per Week
- Native Country

> Note: Some columns like `fnlwgt`, `educational-num`, `capital-gain`, and `capital-loss` are placeholders or derived and thus auto-filled or ignored for prediction.

---

## 🧪 Example

```
Age: 39  
Workclass: Private  
Education: Bachelors  
Marital Status: Never-married  
Occupation: Tech-support  
Relationship: Not-in-family  
Race: White  
Gender: Male  
Hours-per-week: 40  
Native-country: United-States  
```

Predicted Output: `>50K` or `<=50K`

---

## 📦 Dependencies

From `requirements.txt`:

```
pandas==2.3.1  
numpy==2.3.1  
scikit-learn==1.6.1  
streamlit==1.46.1
```

---

## 📚 Dataset Reference

- UCI Adult Income Dataset  
  [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

---

## ✨ Future Improvements

- Add data visualizations of predictions
- Deploy to cloud (e.g., Streamlit Cloud)
- Improve model performance with advanced models or hyperparameter tuning
- Replace placeholder fields with meaningful inputs

---

## 📌 License

This project is for educational and demonstration purposes. Please attribute appropriately if used elsewhere.
