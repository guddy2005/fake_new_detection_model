# Fake News Detection using RoBERTa

## Overview
This project builds a **Fake News Detection system** using the **RoBERTa transformer model**.  
The model classifies news articles as:

- **0 → Fake News**
- **1 → Real News**

The pipeline includes:
- Data collection and cleaning
- Dataset balancing
- Tokenization using RoBERTa tokenizer
- Model training using PyTorch
- Model evaluation using validation accuracy

---

## Project Structure
fake-news-detection/
│
├── dataset.py
├── training.py
├── final_dataset.csv
├── README.md

| File | Description |
|-----|-------------|
| dataset.py | Data cleaning and dataset creation |
| training.py | RoBERTa training pipeline |
| final_dataset.csv | Final processed dataset |

---

## Dataset Preparation

The dataset is created by combining **global fake news data** and **Indian news data**.

### Load datasets

```python
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


Assign labels:
df_fake["label"] = 0
df_true["label"] = 1

Combine datasets:
df_global = pd.concat([df_fake, df_true])

Keep required columns:

text
label


Data Cleaning

News articles often contain source bias patterns such as:
(Reuters) - WASHINGTON -
These patterns can leak labels into the model.

Cleaning steps:
text = re.sub(r'\(Reuters\)', '', text)
text = re.sub(r'^[A-Z\s]+\s-\s', '', text)

Additional preprocessing:

Remove duplicates
Remove null values
Normalize whitespace


Dataset Sampling

To maintain dataset balance:
10,000 samples from Indian dataset
10,000 samples from Global dataset

20,000 news articles
```
## Model Architecture

The project uses the **RoBERTa-base transformer model**.

### Model Configuration

| Parameter | Value |
|-----------|------|
| Model | roberta-base |
| Max sequence length | 384 |
| Batch size | 8 |
| Epochs | 6 |
| Learning rate | 1e-5 |

---

## Data Split

The dataset is split using **stratified sampling**.

- **80% Training**
- **20% Validation**

```python
train_test_split(... stratify=df["label"])
```

This ensures both classes remain balanced.

---

## Tokenization

Text is tokenized using the **RoBERTa tokenizer**.

```python
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
```

### Parameters

- `padding = True`
- `truncation = True`
- `max_length = 384`

---

## Custom PyTorch Dataset

A custom dataset class feeds tokenized text into the model.

```python
class NewsDataset(torch.utils.data.Dataset):
```

Each sample contains:

- `input_ids`
- `attention_mask`
- `labels`

---

## Handling Class Imbalance

Class weights are calculated using:

```python
compute_class_weight(class_weight="balanced")
```

Loss function:

```python
CrossEntropyLoss(weight=class_weights)
```

This improves performance on minority classes.

---

## Training Strategy

### Optimizer

```
AdamW
```

Learning rate:

```
1e-5
```

---

### Learning Rate Scheduler

A **linear scheduler with warm-up** stabilizes training.

```python
get_linear_schedule_with_warmup()
```

Benefits:

- smoother learning
- prevents early overfitting
- faster convergence

---

## Model Evaluation

Validation is performed after every epoch.

Prediction:

```python
argmax(logits)
```

Accuracy calculation:

```python
accuracy_score()
```

### Example Output

```
Epoch 1
Validation Accuracy: 0.79

Epoch 2
Validation Accuracy: 0.83

Epoch 3
Validation Accuracy: 0.86
```

---

## Expected Performance

Typical performance with this configuration:

```
Validation Accuracy: 85% – 90%
```

---

## Installation

Install dependencies:

```bash
pip install torch transformers pandas scikit-learn tqdm
```

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas

---

## Future Improvements

Possible improvements:

- RoBERTa-large
- DeBERTa model
- Data augmentation
- Attention pooling

These techniques can improve accuracy up to **92–95%**.
