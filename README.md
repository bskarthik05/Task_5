# Heart Disease Prediction with Tree-Based Models

## Objective
This project aims to explore and apply tree-based machine learning models (Decision Trees and Random Forests) for classifying heart disease based on a given dataset. It covers training, visualization, overfitting analysis, feature importance interpretation, and model evaluation using cross-validation.

## Dataset
The analysis uses the `heart.csv` dataset. This dataset contains 1025 entries with 13 clinical features and a target variable indicating the presence (1) or absence (0) of heart disease.
* **Features**: `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar > 120 mg/dl), `restecg` (resting electrocardiographic results), `thalach` (maximum heart rate achieved), `exang` (exercise induced angina), `oldpeak` (ST depression induced by exercise relative to rest), `slope` (the slope of the peak exercise ST segment), `ca` (number of major vessels colored by fluoroscopy), `thal` (thalassemia type).
* **Target Variable**: `target` (0 = no disease, 1 = disease).

The dataset was found to have no missing values.

##  Process & Methodology
The project follows these key steps:

1.  **Data Loading and Inspection**:
    * The `heart.csv` dataset is loaded using pandas.
    * Initial exploration includes checking the data structure (`.head()`, `.info()`), summary statistics (`.describe()`), and missing values (`.isnull().sum()`).

2.  **Data Preparation**:
    * Features (X) and the target variable (y) are separated.
    * The data is split into training (80%) and testing (20%) sets, using stratification to maintain the proportion of target classes in both splits.

3.  **Decision Tree Classifier**:
    * A Decision Tree Classifier is trained on the training data.
    * An initial tree with a limited `max_depth` (e.g., 3) is trained for visualization purposes.
    * **Visualization**: The code includes functionality to visualize the decision tree using `Graphviz`, saving it as `decision_tree_heart_viz.png`. *Note: This requires Graphviz to be installed on the system and the Python `graphviz` library. If issues occur, ensure these dependencies are correctly set up and potentially restart your kernel/runtime.*

4.  **Overfitting Analysis and Tree Depth Control**:
    * The impact of varying `max_depth` on the Decision Tree's performance is analyzed.
    * Training and testing accuracies are plotted against different tree depths to identify potential overfitting and find an optimal depth. This plot is saved as `dt_accuracy_vs_depth.png`.
    * An optimized Decision Tree is trained using the best `max_depth` identified from this analysis.

5.  **Random Forest Classifier**:
    * A Random Forest Classifier (an ensemble of decision trees) is trained.
    * Parameters like `n_estimators` (number of trees) and `max_depth` (often guided by the Decision Tree analysis) are used.
    * The accuracy of the Random Forest model is compared with the optimized Decision Tree.

6.  **Feature Importance Interpretation**:
    * Feature importances are extracted from the trained Random Forest model.
    * These importances (typically Gini importance) indicate the relative contribution of each feature to the model's predictions.
    * A bar plot of feature importances is generated and saved as `rf_feature_importances.png`.

7.  **Cross-Validation**:
    * Both the optimized Decision Tree and the Random Forest models are evaluated using 5-fold cross-validation.
    * This provides a more robust estimate of model performance by training and testing on different subsets of the data. Mean cross-validation accuracy and standard deviation are reported.

## Key Findings (Example from a typical run)
* **Data**: The dataset was clean with no missing values.
* **Decision Tree (max_depth=3)**: Achieved an initial test accuracy of ~85.37%.
* **Optimal Decision Tree Depth**: The analysis of accuracy vs. depth indicated an optimal `max_depth` (e.g., 9 in the provided run), leading to a significantly higher test accuracy (e.g., ~98.54%).
* **Random Forest**: The Random Forest model (e.g., with 100 estimators and `max_depth` set to the optimal DT depth) often outperformed the single Decision Tree, achieving very high accuracy (e.g., ~100% on the test set in the provided run).
* **Feature Importances**: Features like `cp` (chest pain type), `ca` (number of major vessels), `thalach` (max heart rate), and `oldpeak` were identified as highly important by the Random Forest model.
* **Cross-Validation**: Both models showed strong and stable performance in 5-fold cross-validation, with mean accuracies typically above 99%. The extremely high scores suggest the dataset version used might be highly separable or contain redundancies.


##  Requirements & Setup
The analysis is performed using Python and requires the following libraries:
* pandas
* scikit-learn
* matplotlib
* numpy
* graphviz (for decision tree visualization)

1.  **Install Python libraries**:
    You can install these using pip:
    ```bash
    pip install pandas scikit-learn matplotlib numpy graphviz
    ```
    Alternatively, if a `requirements.txt` file is provided:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Graphviz (System Dependency)**:
    For decision tree visualization, `Graphviz` (the software, not just the Python library) must be installed on your system and its executables (like `dot`) must be in your system's PATH.
    * **Linux (Debian/Ubuntu)**: `sudo apt-get install graphviz`
    * **macOS (using Homebrew)**: `brew install graphviz`
    * **Windows**: Download an installer from the [official Graphviz website](https://graphviz.org/download/) and ensure you add it to your PATH.

## How to Run
1.  Ensure all required libraries and Graphviz (if visualization is desired) are installed.
2.  Place the `heart.csv` dataset in the same directory as the Python script, or update the file path in the script.
3.  Run the code

##  Generated Files
The script will generate the following output files in the same directory:
* `decision_tree_heart_viz.png`: Visualization of the initial Decision Tree (if Graphviz is set up correctly).
* `dt_accuracy_vs_depth.png`: Plot showing Decision Tree training and testing accuracy vs. `max_depth`.
* `rf_feature_importances.png`: Bar plot of feature importances from the Random Forest model.
