
## Task 1: Multi-Class Benchmark Classification Pipeline

**Goal:** Build a supervised classification pipeline on the Iris dataset and benchmark two models.

**Approach**
1. Load the dataset with Pandas and explore it with Seaborn pairplots.
2. Split the data 80/20 (stratified) and scale features with `StandardScaler`, fit on the training set only.
3. Train `LogisticRegression` and `DecisionTreeClassifier`.
4. Evaluate with accuracy, precision, recall and F1-score.
5. Plot confusion-matrix heatmaps and decision boundaries for both models.

**Deliverables:** classification reports, confusion-matrix heatmaps, decision-boundary plots.

---

## Task 2: Real Estate Housing Price Prediction (Ridge & Lasso)

**Goal:** Predict median house value on the California Housing dataset and compare OLS, Ridge (L2) and Lasso (L1) regression.

**Approach**
1. Load the data and impute missing `total_bedrooms` values with the median.
2. Plot a correlation heatmap.
3. Engineer interaction features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
4. Detect multicollinearity with Variance Inflation Factor (VIF) and drop highly collinear features (`total_bedrooms`, `households`, `total_rooms`).
5. One-hot encode `ocean_proximity`, split 80/20, and standardize features (fit on training data only).
6. Train OLS, plus Ridge and Lasso tuned with `GridSearchCV` (5-fold CV) over multiple alpha values.
7. Evaluate with RMSE, MAE and R², and plot predicted vs. actual values and residual distributions.

**Deliverables:** model comparison table, predicted-vs-actual plots, residual distribution plots.

**Findings:** OLS, Ridge and Lasso perform almost identically. The dataset has many rows and relatively few features, so regularization has little to correct. Residuals show larger errors at high prices, consistent with the dataset's capped target value.

> Update the results below with your final numbers after re-running with the multicollinearity fix.

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| OLS | _fill in_ | _fill in_ | _fill in_ |
| Ridge | _fill in_ | _fill in_ | _fill in_ |
| Lasso | _fill in_ | _fill in_ | _fill in_ |

---

## Task 3: Predictive Customer Churn Classifier

**Goal:** Predict telecom customer churn on an imbalanced dataset (about 26.5% churn) and reach ROC-AUC above 0.80.

**Approach**
1. Load the Telco Customer Churn dataset, convert `TotalCharges` to numeric, and fill the 11 blank values.
2. Drop the `customerID` identifier and encode the target (`Yes` = 1, `No` = 0).
3. Split 80/20 with stratification **before** any balancing, to avoid leakage.
4. Preprocess with `ColumnTransformer` (`OneHotEncoder` for categorical features, `StandardScaler` for numeric ones).
5. Handle class imbalance:
   - Random Forest: **SMOTE** inside an `imblearn` pipeline, so synthetic samples are only created from training folds.
   - XGBoost: **`scale_pos_weight`** set to the negative/positive class ratio.
6. Evaluate with 5-fold stratified cross-validation (ROC-AUC), then on the held-out test set.
7. Plot ROC curves and confusion matrices, and extract the top 10 feature importances.

**Deliverables:** ROC curves, confusion matrices, feature importance bar chart.

| Model | Imbalance method | Test ROC-AUC |
|-------|------------------|--------------|
| Random Forest | SMOTE | 0.830 |
| XGBoost | scale_pos_weight | 0.821 |

Both models exceed the 0.80 target. XGBoost has higher recall on the churn class (about 0.67 vs. 0.55) at the cost of lower precision, which can be preferable when missing a churner costs more than a false alarm.

> Re-run after dropping `customerID` and update these numbers if they change.

---

## Task 4: NLP Sentiment Classification (TF-IDF)

**Goal:** Classify movie reviews as positive or negative using TF-IDF features.

**Note on data:** The provided starter file contained only 10 rows, which is too small for a meaningful train/test evaluation. The full **IMDB 50K Movie Reviews** dataset (loaded from Hugging Face) was used instead.

**Approach**
1. Load the dataset and remove duplicate reviews.
2. Clean the text: lowercase, strip HTML tags, remove punctuation and digits, remove stopwords (keeping negation words such as "not", "no", "never", "nor"), and lemmatize with NLTK's WordNet lemmatizer.
3. Split 80/20, then fit `TfidfVectorizer(max_features=5000)` on the training data only.
4. Train `MultinomialNB` and `LogisticRegression`.
5. Evaluate with classification reports, an accuracy table and confusion matrices.
6. Inspect the most influential positive and negative words.
7. Provide a `predict_sentiment()` function and an interactive input loop for custom phrases.

**Deliverables:** accuracy benchmark, confusion matrices, interactive `predict_sentiment()` helper.

| Model | Accuracy |
|-------|----------|
| Multinomial Naive Bayes | 0.8523 |
| Logistic Regression | 0.8923 |

Logistic Regression outperforms Naive Bayes because it learns a weight for each word instead of assuming word independence. Known limitation: TF-IDF with single words ignores word order, so negated phrases such as "not good" can be misclassified. Adding bigrams (`ngram_range=(1, 2)`) helps with this.

---

## How to Run

All notebooks were developed in **Google Colab**.

1. Upload the notebook to Colab (or open it from GitHub).
2. For Tasks 2 and 3, upload the required CSV to `/content/` and match the filename in the load cell.
3. Run **Runtime → Restart session and run all**.

Task 4 ends with an interactive `input()` loop. Type `quit` to exit it, or comment it out for a non-interactive run.

### Dependencies

```
pandas, numpy, matplotlib, seaborn
scikit-learn
statsmodels            # Task 2 (VIF)
imbalanced-learn       # Task 3 (SMOTE)
xgboost                # Task 3
nltk, datasets         # Task 4
```

Install the extra packages in Colab with:

```python
!pip install -q statsmodels imbalanced-learn xgboost datasets
```

## Key Practices Used Across All Tasks

- Train/test split performed **before** scaling, vectorizing or balancing, so no information leaks from the test set.
- Preprocessing objects (`StandardScaler`, `TfidfVectorizer`) fitted on training data only.
- Class imbalance handled inside cross-validation pipelines, not before the split.
- Multiple metrics reported instead of accuracy alone, with confusion matrices to show where models fail.
