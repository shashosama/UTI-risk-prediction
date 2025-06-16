#  UTI Risk Prediction 

This project builds a machine learning model to predict the risk of **urinary tract infection (UTI)** using patient symptoms and temperature. The goal is to help identify patients at high risk of UTI-related conditions, such as **inflammation of the urinary bladder** and **nephritis of renal pelvis origin**, using interpretable models.

---

##  Dataset

The dataset includes **120 patient records** with features such as:
- `Temperature of patient` (numeric)
- `Occurrence of nausea` (boolean)
- `Lumbar pain` (boolean)
- `Urine pushing (continuous need for urination)` (boolean)
- `Micturition pains` (boolean)
- `Burning of urethra, itch, swelling of urethra outlet` (boolean)

Targets:
- `Inflammation of urinary bladder` (boolean, main target)
- `Nephritis of renal pelvis origin` (boolean, optional target)

---

## Features
 Data cleaning and feature engineering:
- Converted boolean columns to numeric (0/1)
- Created `symptom_score` = sum of key symptoms

 Modeling:
- Logistic regression, decision tree, and random forest models (with `caret` package)
- ROC-AUC, confusion matrix, precision-recall metrics

 Model explainability:
- Feature importance plots
- Partial dependence plots

 Visualization:
- ggplot2 and plotly for EDA and results

---

##  How to Run

```r
# Load required packages
library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(plotly)

# Read dataset
df <- read.csv("data/uti_real_data.csv")

# Preprocess
df <- df %>%
  mutate(
    Occurrence.of.nausea = as.integer(Occurrence.of.nausea),
    Lumbar.pain = as.integer(Lumbar.pain),
    Urine.pushing..continuous.need.for.urination. = as.integer(Urine.pushing..continuous.need.for.urination.),
    Micturition.pains = as.integer(Micturition.pains),
    Burning.of.urethra..itch..swelling.of.urethra.outlet = as.integer(Burning.of.urethra..itch..swelling.of.urethra.outlet),
    Inflammation.of.urinary.bladder = as.factor(Inflammation.of.urinary.bladder)
  ) %>%
  mutate(symptom_score = Occurrence.of.nausea + Lumbar.pain +
         Urine.pushing..continuous.need.for.urination. +
         Micturition.pains +
         Burning.of.urethra..itch..swelling.of.urethra.outlet)

# Split
set.seed(42)
trainIndex <- createDataPartition(df$Inflammation.of.urinary.bladder, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Train model
model <- train(Inflammation.of.urinary.bladder ~ Temperature.of.patient + symptom_score,
               data = trainData, method = "glm", family = "binomial")

# Predict + evaluate
pred <- predict(model, testData)
confusionMatrix(pred, testData$Inflammation.of.urinary.bladder)

# ROC
roc_obj <- roc(testData$Inflammation.of.urinary.bladder, as.numeric(pred))
plot(roc_obj)
