# 🌸 Flower Species Classifier

A beginner-friendly Machine Learning project that classifies Iris flower species using **Decision Tree** and **Random Forest** classifiers.

---

## 📌 Project Overview

This project uses the famous **Iris dataset** to predict the species of a flower based on its measurements. It covers the full ML pipeline — from data loading and exploration to model training, evaluation, and prediction.

**Flower Species Classified:**
- 🌸 Setosa
- 🌺 Versicolor
- 🌻 Virginica

---

## 🧠 What I Learned

- How to load and explore a real dataset using `pandas`
- How to visualize data with `matplotlib`
- How to build a **Decision Tree** and **Random Forest** classifier
- How to evaluate models using accuracy, classification report, and confusion matrix
- How to make predictions on new user inputs

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming Language |
| pandas | Data loading & exploration |
| scikit-learn | ML models & evaluation |
| matplotlib | Data visualization |

---

## 📁 Project Structure

```
flower-classifier/
│
├── flower_classifier.py    # Main Python script
├── requirements.txt        # Required libraries
├── .gitignore              # Files to ignore in Git
├── iris_visualization.png  # Generated: data exploration chart
├── confusion_matrix.png    # Generated: model evaluation chart
└── README.md               # Project documentation
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/flower-classifier.git
cd flower-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the classifier
```bash
python flower_classifier.py
```

### 4. Enter flower measurements when prompted
```
Sepal Length: 5.1
Sepal Width:  3.5
Petal Length: 1.4
Petal Width:  0.2

🌺 Predicted Species : Setosa
🎯 Confidence        : 100.0%
```

---

## 📊 Sample Results

| Model | Accuracy |
|-------|----------|
| Decision Tree | ~96% |
| Random Forest | ~97% |

Charts are automatically saved as PNG files when you run the script.

---

## 📈 Generated Visualizations

- **iris_visualization.png** — Scatter plot and bar chart of feature distributions
- **confusion_matrix.png** — Model performance for both classifiers

---

## 🔮 Future Improvements

- [ ] Add a web UI using Streamlit
- [ ] Try other classifiers (KNN, SVM, Logistic Regression)
- [ ] Use a custom / larger flower dataset
- [ ] Export the trained model using `joblib`

---

## 👩‍💻 Author

**[Your Name]**  
GitHub: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
