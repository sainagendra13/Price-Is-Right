from pathlib import Path

import chromadb
import joblib
import numpy as np

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline


OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


# 1. Load dataset
print("Loading product dataset...")

dataset = load_dataset("ed-donner/items_lite")
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

print(f"Training records: {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Testing records: {len(test)}")

x_train = [row["summary"] for row in train]
y_train = np.array([row["price"] for row in train], dtype=float)

x_test = [row["summary"] for row in test]
y_test = np.array([row["price"] for row in test], dtype=float)


# 2. Train Deep Neural Network
print("Training DNN...")

dnn = make_pipeline(
    TfidfVectorizer(
        max_features=1500,
        stop_words="english"
    ),
    MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        early_stopping=True,
        max_iter=100,
        random_state=42
    )
)

# Log transformation helps with skewed prices
dnn.fit(x_train, np.log1p(y_train))

joblib.dump(dnn, OUT / "dnn.joblib")

predictions = np.maximum(
    0,
    np.expm1(dnn.predict(x_test))
)

print(
    f"DNN test MAE: "
    f"${mean_absolute_error(y_test, predictions):.2f}"
)


# 3. Create RAG vector database
print("Building Chroma database...")

client = chromadb.PersistentClient(
    path=str(OUT / "chroma")
)

# Check whether the collection exists before deleting it
existing_collections = [
    collection.name
    for collection in client.list_collections()
]

if "products" in existing_collections:
    client.delete_collection(name="products")

# Create a fresh collection
collection = client.create_collection(
    name="products"
)

# Insert all 20,000 training products
for start in range(0, len(train), 200):
    end = min(start + 200, len(train))

    collection.add(
        ids=[str(i) for i in range(start, end)],
        documents=x_train[start:end],
        metadatas=[
            {"price": float(p)}
            for p in y_train[start:end]
        ]
    )

    print(f"Indexed {end}/{len(train)}")

print("Training and indexing finished.")
print("Total products in ChromaDB:", collection.count())