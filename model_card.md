# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classifier that predicts whether a person's annual income is above 50,000 US dollars (">50K") or at most 50,000 US dollars ("<=50K"), based on demographic and employment attributes from US Census Bureau data. It was built as a course project on deploying a scalable machine learning pipeline.

The model is a scikit-learn `RandomForestClassifier` (scikit-learn 1.7.2, Python 3.13) with 100 trees, `min_samples_leaf=5` and `random_state=42` (plus `n_jobs=-1` to train in parallel); all other hyperparameters keep their scikit-learn defaults. The `min_samples_leaf` setting keeps the saved model at about 10 MB instead of about 70 MB for fully grown trees, and slightly improved the F1 score on the test set in an informal comparison. No further hyperparameter tuning was done.

The eight categorical features are one hot encoded with a `OneHotEncoder` that ignores unseen categories, the label is binarized with a `LabelBinarizer`, and the six numeric features are used unscaled (tree models do not need scaling). The fitted model, encoder and label binarizer are stored in `model/model.joblib`, `model/encoder.joblib` and `model/lb.joblib`. This is version 1.0, trained in October 2026.

## Intended Use

The model is intended for learning and demonstration: it shows how to train, test, version, deploy and monitor a classifier behind a REST API. It can be used to explore how census attributes relate to a binary income label and to study model performance on slices of a population.

It is not intended for any real decision about a person, such as hiring, lending, insurance, housing or benefits eligibility. The data is decades old and the model has known performance gaps between groups, as described below.

## Training Data

The data is the Census Income data set (also known as the "Adult" data set) from the UCI Machine Learning Repository, extracted from the 1994 US Census database. The file `data/census.csv` has 32,561 rows, 14 input features and the label `salary`. The label is "<=50K" for 24,720 rows and ">50K" for 7,841 rows, so about 24% of rows are in the positive class.

The eight categorical features are `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex` and `native-country`. The six numeric features are `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss` and `hours-per-week`. The `fnlgt` column is a census sampling weight and not a personal attribute.

The raw file contained a space after every comma. These spaces were removed and the column names were left unchanged. Missing values are recorded as "?" in `workclass` (1,836 rows), `occupation` (1,843 rows) and `native-country` (583 rows); they were kept and are treated as their own category. The 24 exact duplicate rows were not removed.

The data was split once into 80% for training (26,048 rows) and 20% for evaluation (6,513 rows), using a random split stratified by the label with `random_state=42`.

## Evaluation Data

The model is evaluated on the 6,513 rows held out from the split described above. These rows were not used to fit the model or the encoder. They have the same label balance as the training data: 1,568 rows (about 24%) are ">50K". The evaluation data is processed with the encoder and label binarizer that were fitted on the training data only.

## Metrics

The model is evaluated with precision, recall and the F1 score (the harmonic mean of precision and recall, `fbeta` with beta equal to 1), treating ">50K" as the positive class. Accuracy is not used as the main metric because about 76% of rows are "<=50K", so a model that always predicts "<=50K" would already be about 76% accurate.

On the held-out test set the model reaches a precision of 0.7847, a recall of 0.6135 and an F1 score of 0.6886. In other words, when the model predicts ">50K" it is right about 78% of the time, and it finds about 61% of the people whose true label is ">50K".

Performance on slices of the data, where the value of one categorical feature is held fixed, is written to `slice_output.txt` for every distinct value of every categorical feature. Some examples from that file:

- By sex, precision is 0.8693 and recall is 0.5429 for women (2,158 test rows), and precision is 0.7726 and recall is 0.6266 for men (4,355 test rows). The F1 scores are 0.6683 and 0.6920.
- By education, the F1 score is high for Masters (0.8723), Doctorate (0.8689) and Prof-school (0.8475) and much lower for HS-grad (0.4097) and for the lowest school levels such as 10th (0.2353).
- By relationship, the F1 score is 0.7156 for Husband and 0.7713 for Wife, but only 0.1176 for Own-child, a group that has just 15 positive cases among 1,032 rows.

## Ethical Considerations

The data comes from the 1994 US Census and reflects the social and economic conditions of that time, including historical inequality between groups. The features include sensitive attributes (`race`, `sex` and `native-country`) and strong proxies for them, such as `relationship` and `marital-status`. A model trained on such data can reproduce or amplify those inequalities. The slice results show uneven performance: for example, recall is lower for women than for men, so women with a true income above 50K are less likely to be identified by the model.

Using this model to make decisions about real people could cause unfair treatment, and it should not be used that way. If a similar model were ever used in a real setting, the training data would need to be current and representative, the choice of sensitive features would need to be reviewed, and fairness would need to be measured across groups before and after deployment. The API does not store the data that is sent to it.

## Caveats and Recommendations

The 50,000 US dollar threshold refers to 1994 dollars and is not comparable to today's incomes, and the population, occupations and wages have changed since. The model would need to be retrained on current data before it could say anything about the present.

The model is conservative: its precision is higher than its recall in most slices, so it misses many people whose true label is ">50K". It was evaluated on a single random split and was not cross-validated or extensively tuned, so the reported numbers carry some sampling uncertainty.

The slice metrics must be read together with the number of rows in each slice. Several slices are very small, for example `native-country` values with a handful of rows, and their metrics are unreliable. The metric functions use `zero_division=1`, so when a slice has no predicted (or no actual) ">50K" rows, precision (or recall) is reported as 1.0 by convention and not because the model did well. The `n` and `positives` columns in `slice_output.txt` should be checked first. The value `Holand-Netherlands` appears only once in the full data, is not in the test split, and is therefore listed with no metrics.

Categories that the encoder did not see during training are ignored, so the API still returns a prediction for them, but that prediction does not use the unknown category. Recommended next steps are cross-validation and hyperparameter tuning, a probability threshold chosen to balance precision and recall, fairness checks across `sex` and `race`, and monitoring of the slice metrics whenever the model is retrained.
