# Deep Learning PR 2 - Adult Income Prediction with an ANN

Red & White Skill Education | Deep Learning | PR 2

## Project Objective
Predict whether a person earns more than 50K per year (class 1) or 50K or less (class 0) using an ANN (MLP) built with TensorFlow/Keras. The project studies how activation functions, weight initialization, loss functions, batch normalization and optimizers change the result, with a special focus on the minority class (>50K).

## Dataset
- Name: Adult Income Dataset (Census Income)
- Source: UCI Machine Learning Repository (also on Kaggle)
- UCI URL: https://archive.ics.uci.edu/dataset/2/adult
- Kaggle URL: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
- File used: `adult.csv` (48,842 rows, 14 input features + `income` target)
- Class balance: about 76% <=50K and 24% >50K (imbalanced)

The dataset file is not included in this repository. Download `adult.csv` from the Kaggle link and place it next to the notebook.

## Preprocessing Steps
1. Strip whitespace from all text columns.
2. Replace `?` with `NaN` (missing values are in `workclass`, `occupation`, `native.country`) and drop those rows. Dropping is used instead of mode imputation because the columns are categorical with many categories and the mode would inflate the biggest category.
3. Drop `fnlwgt` (census sampling weight) and `education` (same information as `education.num`).
4. Encode the target: `>50K` = 1, `<=50K` = 0.
5. Group rare `native.country` values into `Other`.
6. One-hot encode all categorical columns with `pd.get_dummies(drop_first=True)`.
7. Stratified 80/20 split (`test_size=0.2`, `random_state=42`, `stratify=y`).
8. StandardScaler on the numeric columns only (age, education.num, capital.gain, capital.loss, hours.per.week), fitted on the training set.

## The Six Main Concepts
1. **Preprocessing and imbalance:** Categorical columns are one-hot encoded and numeric columns are scaled. Because about 75% of people are in class 0, accuracy is misleading, so Precision, Recall and F1 of class 1 are tracked.
2. **Activation functions:** ReLU, Tanh, Sigmoid and ELU are compared. A dead neuron check (ReLU) and a gradient flow check with `tf.GradientTape` (Sigmoid) explain dead neurons and vanishing gradients.
3. **Weight initialization:** Glorot Uniform, Glorot Normal, He Uniform, He Normal and Zeros are compared. Zeros shows the symmetry problem. Weight histograms are drawn before and after training.
4. **Loss functions:** BCE, MSE, Weighted BCE (`compute_class_weight`) and a custom Focal Loss are compared using class-1 F1, Recall and Precision.
5. **Batch Normalization:** A BN network is compared with the baseline. The position of BN (before or after activation) is tested and the learned gamma and beta values are inspected.
6. **Optimizers:** SGD, SGD with momentum, RMSprop and Adam are compared, followed by a learning rate sensitivity test for SGD and Adam and a final combined model trained for 80 epochs.

## build_ann() Parameters
| Parameter | Default | Description |
|---|---|---|
| input_dim | required | Number of input features after preprocessing |
| hidden_units | [128, 64] | Number of neurons in each hidden Dense layer |
| activation | 'relu' | Activation used in the hidden layers |
| initializer | 'glorot_uniform' | Kernel initializer of the Dense layers |
| use_batch_norm | False | If True, adds BatchNormalization between Dense and Activation |
| optimizer | 'adam' | Optimizer name or Keras optimizer object |
| loss | 'binary_crossentropy' | Loss name or custom loss function |

The output layer is always one sigmoid neuron. The model is compiled with accuracy, Precision and Recall metrics.

## Important Plots
Add the plots after running the notebook. The notebook saves them in the `plots/` folder.

| Plot | File | Caption |
|---|---|---|
| Class balance | `plots/eda_class_balance.png` | Class distribution of income (about 75/25) |
| Activation comparison | `plots/activation_comparison.png` | 2x2 comparison of ReLU, Tanh, Sigmoid and ELU |
| Initialiser convergence | `plots/initialiser_convergence.png` | Validation accuracy for the 5 initializers |
| Weight distributions | `plots/weight_distributions.png` | He Normal and Glorot Uniform weights before and after training |
| Loss comparison | `plots/loss_f1_comparison.png` | Validation F1 (class 1) of BCE vs MSE |
| BatchNorm dynamics | `plots/batchnorm_dynamics.png` | Loss and accuracy with and without BatchNorm |
| Optimiser convergence | `plots/optimiser_convergence.png` | Validation accuracy for the 5 optimizers |
| ROC curves | `plots/roc_curves.png` | Baseline vs Weighted BCE vs Final Combined |
| Results table | `plots/results_table.png` | Test metrics of all models |

Example of how to show a plot in this README:

```
![Activation comparison](plots/activation_comparison.png)
```

## Final Results Table
Run the notebook and paste the `df_results` table here (it is also saved as `plots/results_table.png`). The values must come from your own run.

| Model | Activation | Init | Loss | BatchNorm | Optimizer | Test Acc | Prec(1) | Recall(1) | F1(1) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| (paste results here) | | | | | | | | | | |

## Video
Video explanation: (paste your Google Drive or YouTube unlisted link here)

## Tools Used
Python 3, Jupyter Notebook, TensorFlow 2.x / Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn
