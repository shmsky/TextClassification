# !pip install transformers[torch] datasets

import pandas as pd
import numpy as np
import re # работа с регулярными выражениями (очистка текста)
import string

# Импорт библиотеки NLTK для работы с текстом
import nltk
from nltk.corpus import stopwords

from datasets import Dataset

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Ноутбук запускался в среде kaggle, поэтому данные загружались напрямую с датасета

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

"""### Данные

* **Clothing ID** — целочисленная категориальная переменная, указывающая на конкретный предмет одежды, к которому относится отзыв.
* **Age** — положительное целое число, возраст автора отзыва.
* **Title** — строковая переменная, заголовок отзыва.
* **Review Text** — строковая переменная, основная часть отзыва.
* **Rating** — порядковая целочисленная переменная от 1 (хуже всего) до 5 (лучше всего), отражающая оценку товара, выставленную покупателем.
* **Recommended IND** — бинарная переменная: 1 — товар рекомендован, 0 — не рекомендован.
* **Positive Feedback Count** — положительное целое число, количество других пользователей, которые нашли отзыв полезным.
* **Division Name** — категориальное наименование верхнего уровня товарного раздела.
* **Department Name** — категориальное наименование товарного отдела.
* **Class Name** — категориальное наименование товарной категории.

### Предобработка корпуса текста
"""

# Очистка
def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Удаление строк с пустыми отзывами
df.dropna(subset=['Review Text'], inplace=True)

# Стоп-слова
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df['Cleaned Review'] = df['Review Text'].apply(clean_text)

df

df[['Review Text', 'Cleaned Review']].head()

# Дублированные функции: токенизация и подсчет метрик

def tokenize_function(examples):
    return tokenizer(examples['Cleaned Review'], padding='max_length', truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'f1': f1_score(labels, predictions)}

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Recommended IND'], random_state=42)

"""### BERT (bert-base-uncased)

"""

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['Review Text', 'Title', 'Cleaned Review', '__index_level_0__'])
val_dataset = val_dataset.remove_columns(['Review Text', 'Title', 'Cleaned Review', '__index_level_0__'])

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Recommended IND'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Recommended IND'])

# Меняем название, поскольку модель ожидает название "labels"
train_dataset = train_dataset.rename_column('Recommended IND', 'labels')
val_dataset = val_dataset.rename_column('Recommended IND', 'labels')

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir='./results/BaseBert',
    num_train_epochs=3, # количество эпох - полных проходов по тренировочным данным
    per_device_train_batch_size=8, # размер батча на одно устройство (GPU)
    per_device_eval_batch_size=8,
    warmup_steps=500, # количество шагов "разогрева" - постепенное увеличение learning rate в начале обучения
    weight_decay=0.01, # L2-регуляризация для весов модели - для борьбы с переобучением
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

"""### RoBERTa (roberta-base)"""

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['Review Text', 'Title', 'Cleaned Review', '__index_level_0__'])
val_dataset = val_dataset.remove_columns(['Review Text', 'Title', 'Cleaned Review', '__index_level_0__'])

train_dataset = train_dataset.rename_column('Recommended IND', 'labels')
val_dataset = val_dataset.rename_column('Recommended IND', 'labels')

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir='./results/RoBERT',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

"""### DistilBERT (distilbert-base-uncased)"""

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Загрузка токенизатора и модели DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column('Recommended IND', 'labels')
val_dataset = val_dataset.rename_column('Recommended IND', 'labels')

columns_to_remove = ['Review Text', 'Title', 'Cleaned Review', '__index_level_0__']
train_dataset = train_dataset.remove_columns([col for col in columns_to_remove if col in train_dataset.column_names])
val_dataset = val_dataset.remove_columns([col for col in columns_to_remove if col in val_dataset.column_names])

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir='./results/DistilBERT',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

"""### Общий пайплайн - пример"""

from transformers import (
    AutoTokenizer,         # общий класс-фабрика для всех токенизаторов
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True # mixed-precision (FP16) ускоряет и экономит память

data_collator = DataCollatorWithPadding(tokenizer=None, return_tensors="pt")  # токенизатор задам позже


model_names = {
    "BERT-base": "bert-base-uncased",
    "RoBERTa-base": "roberta-base",
    "DistilBERT-base": "distilbert-base-uncased",
}

results = {}
for label, model_id in model_names.items():
    print(f"\n {label} ")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_train = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    data_collator.tokenizer = tokenizer  # обновим паддинг к актуальному токенизатору

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_res = trainer.evaluate()
    results[label] = eval_res["eval_f1"]

print(pd.Series(results, name="F1-score").sort_values(ascending=False))

"""### Предсказываем рейтинг"""

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # используем 'weighted' для многоклассовой задачи
    return {'f1': f1_score(labels, predictions, average='weighted')}

# Преобразование рейтинга в диапазон 0-4, поскольку модель работает с метками,
# которые начинаются с нуля

df['Rating'] = df['Rating'] - 1

# Работать будем с базовой Бертой для удобства
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Rating'], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['Review Text', 'Title', 'Cleaned Review', '__index_level_0__'])
val_dataset = val_dataset.remove_columns(['Review Text', 'Title', 'Cleaned Review', '__index_level_0__'])

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Rating'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Rating'])

train_dataset = train_dataset.rename_column('Rating', 'labels')
val_dataset = val_dataset.rename_column('Rating', 'labels')

training_args = TrainingArguments(
    output_dir='./results/BaseBert_2',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
