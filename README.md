# Proiect Titanic – PCLP3

## Autor
Ștefănescu Andrei-Cosmin, grupa 314CB

## Descriere generală
Acest proiect are ca scop analiza și modelarea datelor din celebrul set Titanic, cu accent pe clasificarea supraviețuirii pasagerilor. Am folosit Python și biblioteci precum pandas, scikit-learn, matplotlib și seaborn pentru preprocesare, analiză exploratorie și antrenarea unui model de clasificare.

## Argumentare alegeri
- **Sursa datelor:** Am folosit setul de date Titanic de pe Kaggle, deoarece este un benchmark clasic pentru probleme de clasificare binară și conține atât variabile numerice, cât și categorice, cu valori lipsă și distribuții dezechilibrate.
- **Tipul problemei:** Am ales să formulez problema ca una de clasificare (supraviețuire: da/nu), nu de regresie, deoarece variabila țintă este binară și interpretarea rezultatelor este mai relevantă pentru contextul istoric.
- **Model:** Am optat pentru Random Forest, un model robust pentru clasificare, capabil să gestioneze atât variabile numerice, cât și categorice (după encoding), tolerant la outlieri și valori lipsă (cu imputer). Am folosit și un pipeline cu SimpleImputer pentru a trata valorile lipsă.
- **Preprocesare:** Am tratat valorile lipsă cu mediană (pentru numeric) și modă (pentru categoric), am făcut encoding pentru variabilele categorice și am aliniat coloanele între seturile de antrenament și test.

## Structura proiectului
- `data_prep.py` – Preprocesează și generează seturile de date de antrenament și test.
- `train_eda_analis.py` – Analiză exploratorie a datelor pentru setul de antrenament.
- `test_eda_analis.py` – Analiză exploratorie a datelor pentru setul de test.
- `train_model.py` – Antrenează și evaluează modelul Random Forest, salvează metrici și grafice relevante.
- `plots/` – Conține graficele generate automat (distribuții, boxplot-uri, matrice de confuzie etc).
- `titanic/` – Seturile de date brute originale.
- `train_processed.csv`, `test_processed.csv` – Seturi de date preprocesate.

## Pași de rulare
1. Preprocesare date:
   ```bash
   python3 data_prep.py
   ```
2. Analiză exploratorie:
   ```bash
   python3 train_eda_analis.py
   python3 test_eda_analis.py
   ```
3. Antrenare și evaluare model:
   ```bash
   python3 train_model.py
   ```
4. Vizualizare rezultate: Graficele și rezultatele se găsesc în directorul `plots/`.

## Analiza exploratorie a datelor (EDA) și rezultate model
Am analizat distribuțiile, valorile lipsă, relațiile dintre variabile și corelațiile. Mai jos sunt TOATE ploturile generate, cu scurte comentarii:

### Vizualizare valori lipsă
![train_missing_values](plots/train_missing_values.png)
![test_missing_values](plots/test_missing_values.png)
  - **Comentariu:** Se observă că variabilele 'Age' și 'Cabin' au cele mai multe valori lipsă. 'Cabin' a fost eliminată, iar 'Age' a fost completată cu mediana.
                    De asemenea s-au completat cu cea mai frecventa varianta campurile 'Fare' respectiv 'Embarked' in cazul campurilor goale.   

### Distribuții variabile numerice
![train_numerical_distributions](plots/train_numerical_distributions.png)
![test_numerical_distributions](plots/test_numerical_distributions.png)
  - **Comentariu:** 'Age' și 'Fare' au distribuții asimetrice, cu outlieri. 'SibSp' și 'Parch' sunt concentrate spre 0.

### Boxplot-uri și violin plot-uri (train)
![train_Age_boxplot](plots/train_Age_boxplot.png)
![train_Fare_boxplot](plots/train_Fare_boxplot.png)
![train_Parch_boxplot](plots/train_Parch_boxplot.png)
![train_SibSp_boxplot](plots/train_SibSp_boxplot.png)
![train_Age_violin_by_survival](plots/train_Age_violin_by_survival.png)
![train_Fare_violin_by_survival](plots/train_Fare_violin_by_survival.png)
![train_Parch_violin_by_survival](plots/train_Parch_violin_by_survival.png)
![train_SibSp_violin_by_survival](plots/train_SibSp_violin_by_survival.png)
  - **Comentariu:** Supraviețuitorii tind să fie mai tineri și să aibă bilete mai scumpe. Violin plot-urile arată diferențe de distribuție între clasele țintă.

### Boxplot-uri și violin plot-uri (test)
![test_Age_boxplot](plots/test_Age_boxplot.png)
![test_Fare_boxplot](plots/test_Fare_boxplot.png)
![test_Parch_boxplot](plots/test_Parch_boxplot.png)
![test_SibSp_boxplot](plots/test_SibSp_boxplot.png)
![test_Age_violin_by_survival](plots/test_Age_violin_by_survival.png)
![test_Fare_violin_by_survival](plots/test_Fare_violin_by_survival.png)
![test_Parch_violin_by_survival](plots/test_Parch_violin_by_survival.png)
![test_SibSp_violin_by_survival](plots/test_SibSp_violin_by_survival.png)
  - **Comentariu:** Distribuțiile sunt similare cu cele din setul de antrenament, dar cu unele variații în forme și valori extreme.

### Countplot-uri pentru variabile categorice (train)
![train_Sex_by_survival](plots/train_Sex_by_survival.png)
![train_Pclass_by_survival](plots/train_Pclass_by_survival.png)
![train_Embarked_by_survival](plots/train_Embarked_by_survival.png)
![train_Sex_count_by_survival](plots/train_Sex_count_by_survival.png)
![train_Pclass_count_by_survival](plots/train_Pclass_count_by_survival.png)
![train_Embarked_count_by_survival](plots/train_Embarked_count_by_survival.png)
  - **Comentariu:** Femeile și pasagerii din clasa I au avut șanse mai mari de supraviețuire. Portul de îmbarcare influențează ușor distribuția.

### Countplot-uri pentru variabile categorice (test)
![test_Sex_by_survival](plots/test_Sex_by_survival.png)
![test_Pclass_by_survival](plots/test_Pclass_by_survival.png)
![test_Embarked_by_survival](plots/test_Embarked_by_survival.png)
![test_Sex_count_by_survival](plots/test_Sex_count_by_survival.png)
![test_Pclass_count_by_survival](plots/test_Pclass_count_by_survival.png)
![test_Embarked_count_by_survival](plots/test_Embarked_count_by_survival.png)
  - **Comentariu:** Distribuțiile sunt consistente cu cele din setul de antrenament, validând astfel alegerea variabilelor explicative.

### Scatter plot și corelații
![train_age_fare_scatter_by_survival](plots/train_age_fare_scatter_by_survival.png)
![test_age_fare_scatter_by_survival](plots/test_age_fare_scatter_by_survival.png)
![train_correlation_heatmap](plots/train_correlation_heatmap.png)
![test_correlation_heatmap](plots/test_correlation_heatmap.png)
  - **Comentariu:** Nu există o corelație puternică între 'Age' și 'Fare', dar corelația cu supraviețuirea este vizibilă pentru unele variabile.

### Matrice de confuzie și erori model
![confusion_matrix_baseline](plots/confusion_matrix_baseline.png)
![prediction_errors](plots/prediction_errors.png)
  - **Comentariu:** Matricea de confuzie arată performanța modelului Random Forest. Graficul de erori evidențiază unde modelul greșește predicțiile.

## Rezultate model Random Forest
- **Acuratețe:** ~0.8 (variază în funcție de random seed și preprocesare)
- **Precizie, recall, F1-score:** Raportate în consolă și în `train_model.py`.
- **Interpretare:** Modelul se descurcă bine pe datele de test, dar există încă confuzii între clase, mai ales la pasagerii din clasa a III-a și bărbați.

## Concluzii
- Alegerea Random Forest a fost justificată de robustetea sa și de capacitatea de a lucra cu date eterogene.
- Preprocesarea corectă (imputare, encoding, aliniere coloane) este esențială pentru rezultate bune.
- Analiza exploratorie ajută la înțelegerea datelor și la alegerea strategiilor de preprocesare și modelare.


## Notă
Acest proiect a fost realizat pentru disciplina PCLP3, anul universitar 2024-2025.

