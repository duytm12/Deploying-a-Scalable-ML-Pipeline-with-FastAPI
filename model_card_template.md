# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Name**: Census Income Prediction Model

**Model Type**: Random Forest Classifier

**Model Version**: 1.0.0

**Model Architecture**: 
- Algorithm: Random Forest Classifier from scikit-learn
- Number of estimators: 100
- Random state: 42 for reproducibility
- Criterion: Gini impurity (default)
- Max depth: None (unlimited)
- Min samples split: 2 (default)
- Min samples leaf: 1 (default)

**Training Data**: UCI Census Income Dataset (Adult dataset)
- Source: https://archive.ics.uci.edu/dataset/20/census+income
- Training set size: 80% of total data (approximately 26,048 samples)
- Test set size: 20% of total data (approximately 6,512 samples)

**Features**:
- Categorical features (8): workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Numerical features (6): age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- Target variable: salary (binary classification: <=50K vs >50K)

## Intended Use

**Primary Use Case**: Income prediction based on demographic and employment characteristics

**Intended Users**: 
- Researchers studying income inequality and demographic factors
- Policy makers analyzing socioeconomic patterns
- Educational institutions for academic research

**Out-of-Scope Uses**:
- Direct employment decisions
- Credit scoring
- Insurance underwriting
- Any commercial applications without proper validation

## Training Data

**Dataset**: UCI Census Income Dataset (Adult dataset)

**Data Collection**: 
- Collected from the 1994 Census database
- Contains demographic and employment information
- Target variable indicates whether income exceeds $50K annually

**Data Preprocessing**:
- Categorical features encoded using One-Hot Encoding
- Label binarization for the target variable
- No scaling applied to numerical features (Random Forest is scale-invariant)
- Missing values handled by the OneHotEncoder with handle_unknown="ignore"

**Data Quality**:
- Contains 32,561 total samples
- 14 features per sample
- Binary classification task
- Class distribution: ~76% <=50K, ~24% >50K

## Evaluation Data

**Test Set**: 20% of the original dataset (approximately 6,512 samples)

**Validation Strategy**: Train-test split with random_state=42 for reproducibility

**Cross-Validation**: Not used in this implementation, but could be added for more robust evaluation

## Metrics

**Primary Metrics Used**:
- Precision: Measures the accuracy of positive predictions
- Recall: Measures the ability to find all positive cases
- F1-Score: Harmonic mean of precision and recall (beta=1)

**Model Performance**:
- Precision: 0.7376
- Recall: 0.6288
- F1-Score: 0.6789

**Slice Performance Analysis**:
The model performance varies across different demographic slices. For example:
- Workclass "Federal-gov": Precision: 0.8197, Recall: 0.7353, F1: 0.7752
- Workclass "Private": Precision: 0.7381, Recall: 0.6245, F1: 0.6766
- Education "Bachelors": Precision: 0.7500, Recall: 0.7000, F1: 0.7241

**Performance Interpretation**:
- The model shows moderate performance with room for improvement
- Higher precision indicates fewer false positives
- Lower recall suggests the model misses some high-income individuals
- F1-score provides a balanced view of overall performance

## Ethical Considerations

**Potential Biases**:
- Historical biases in the 1994 Census data may be reflected in the model
- Demographic features (race, sex, native-country) could introduce unfair bias
- The model may perpetuate existing socioeconomic inequalities

**Fairness Concerns**:
- Performance varies across different demographic groups
- The model should not be used for discriminatory purposes
- Regular fairness audits should be conducted

**Privacy Considerations**:
- The dataset contains sensitive demographic information
- Model predictions should be used responsibly
- Compliance with data protection regulations is essential

**Mitigation Strategies**:
- Regular monitoring of model performance across demographic slices
- Implementation of fairness constraints in model training
- Transparent documentation of model limitations
- Regular retraining with updated, more representative data

## Caveats and Recommendations

**Model Limitations**:
- Based on 1994 data, may not reflect current economic conditions
- Limited to US Census data, not generalizable to other countries
- Binary classification may oversimplify income distribution
- No temporal dynamics considered

**Recommendations for Use**:
- Use only for research and educational purposes
- Validate results with current data sources
- Consider ensemble methods for improved performance
- Implement feature importance analysis for interpretability
- Regular model retraining with updated data

**Deployment Considerations**:
- Monitor model drift over time
- Implement A/B testing for model updates
- Set up automated retraining pipelines
- Establish clear model governance policies

**Future Improvements**:
- Collect more recent data for training
- Implement cross-validation for more robust evaluation
- Add feature engineering for better performance
- Consider alternative algorithms (XGBoost, Neural Networks)
- Implement explainable AI techniques for interpretability
