# The Price Is Right — Multi-Agent AI Deal Hunter

An AI-powered deal discovery and price prediction system built using **LangGraph, Retrieval-Augmented Generation (RAG), Scikit-learn, ChromaDB, and OpenAI/Ollama**.

The application automatically discovers product deals, estimates their market prices using a Deep Neural Network (DNN) and RAG, identifies potential savings, and sends notifications through Pushover.

## Project Overview

This project demonstrates a multi-agent AI workflow in which specialized agents collaborate to discover and evaluate product deals.

**Workflow:**

Scanner Agent → Preprocessor Agent → DNN Agent → RAG Agent → Ensemble Agent → Planning Agent → Messaging Agent

### Agents and Responsibilities

| Agent | Responsibility |
|---|---|
| Scanner Agent | Retrieves product deals from RSS feeds |
| Preprocessor Agent | Cleans and standardizes product descriptions |
| DNN Agent | Predicts product prices using a trained Scikit-learn neural network |
| RAG Agent | Retrieves similar products from ChromaDB and estimates prices using an LLM |
| Ensemble Agent | Combines DNN and RAG predictions |
| Planning Agent | Ranks deals based on estimated savings |
| Messaging Agent | Sends deal notifications using Pushover |

**LangGraph** orchestrates the complete workflow and handles conditional execution.


## Multi-Agent Workflow Architecture

The application uses LangGraph to orchestrate specialized AI agents responsible for discovering, evaluating, and ranking product deals.

flowchart TD
    A([User clicks Find Deals]) --> B[Scanner Agent]
    B --> C{New deals found?}

    C -->|No| K[Planning Agent]
    C -->|Yes| D[Select Next Product]

    D --> E[Preprocessor Agent]
    E --> F[DNN Agent]
    F --> G[RAG Agent]
    G --> H[Ensemble Agent]
    H --> I[Calculate Estimated Savings]

    I --> J{More products?}
    J -->|Yes| D
    J -->|No| K

    K --> L{Savings exceed threshold?}
    L -->|Yes| M[Messaging Agent: Pushover Alert]
    L -->|No| N[No qualifying deal]

    M --> O[Save Processed URLs]
    N --> O
    O --> P([Display Results in Gradio])

    subgraph Machine Learning
        F
    end

    subgraph Retrieval Augmented Generation
        G
    end

Workflow Explanation

Scanner Agent: Fetches product offers from RSS feeds and extracts product descriptions and advertised prices.

Preprocessor Agent: Uses an LLM to clean product descriptions and remove promotional information.

DNN Agent: Predicts product prices using a Scikit-learn neural network trained on 20,000 product records.

RAG Agent: Retrieves five similar products from ChromaDB and uses an OpenAI/Ollama model to estimate the product's price.

Ensemble Agent: Combines DNN and RAG estimates using a weighted average.

Planning Agent: Ranks evaluated products by potential savings and selects the best qualifying deal.

Messaging Agent: Generates an alert and optionally sends a Pushover notification.

Memory: Stores previously processed product URLs to avoid evaluating the same deals repeatedly.

Price Estimation

The ensemble combines both model predictions:

Final Estimated Price = 0.3 × DNN Prediction + 0.7 × RAG Prediction

Potential Savings = Final Estimated Price − Advertised Price

A deal qualifies for notification when its estimated savings meet the configured MIN_SAVINGS threshold.

Note: Estimated savings are based on model predictions, not verified resale values or guaranteed profits.

## Technology Stack

- **Python** — Core programming language
- **LangGraph** — Multi-agent workflow orchestration
- **Scikit-learn** — Deep Neural Network price prediction
- **ChromaDB** — Vector database for product similarity search
- **OpenAI / Ollama** — LLM-based preprocessing and RAG
- **Gradio** — Web interface
- **Pushover** — Push notifications
- **Hugging Face Datasets** — Product training dataset
- **uv** — Python environment and dependency management

## Dataset

The project uses the Hugging Face dataset `ed-donner/items_lite`.

| Split | Records |
|---|---:|
| Training | 20,000 |
| Validation | 1,000 |
| Testing | 1,000 |

The complete training split is used to train the neural network and build the ChromaDB vector database.

The validation and test splits are kept separate for model evaluation.

## Project Structure

```text
Price-Is-Right/
├── train.py
├── llm.py
├── agents.py
├── workflow.py
├── app.py
├── pyproject.toml
├── uv.lock
├── .env
├── .gitignore
└── artifacts/
    ├── dnn.joblib
    ├── chroma/
    └── seen_urls.json
```

The `artifacts/` directory is generated locally and is not committed to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Price-Is-Right.git
cd Price-Is-Right
```

### 2. Install uv

If uv is not installed, follow the instructions at:

https://docs.astral.sh/uv/getting-started/installation/

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

OLLAMA_MODEL=qwen2.5:3b

MIN_SAVINGS=50

PUSHOVER_USER=your_pushover_user_key
PUSHOVER_TOKEN=your_pushover_application_token
```

For a local LLM, set `LLM_PROVIDER=ollama`, install Ollama, and download the model:

```bash
ollama pull qwen2.5:3b
```

An OpenAI API key is not required when using Ollama.

## Pushover Notification Setup

Pushover is used to send notifications when the application identifies a potentially profitable deal.

### Step 1 — Create a Pushover account

1. Visit https://pushover.net/
2. Register or sign in.
3. Install the Pushover application on your phone and register your device.

### Step 2 — Get your User Key

1. Sign in to your Pushover dashboard.
2. Locate **Your User Key**.
3. Copy the key.
4. Add it to `.env`:

```dotenv
PUSHOVER_USER=your_user_key
```

### Step 3 — Generate an Application API Token

1. On the Pushover dashboard, find **Your Applications**.
2. Click **Create an Application/API Token**.
3. Enter an application name, such as `Price Is Right Deal Hunter`.
4. Complete the application registration.
5. Open the newly created application.
6. Copy its **API Token/Key**.
7. Add it to `.env`:

```dotenv
PUSHOVER_TOKEN=your_application_api_token
```

Both credentials are required to send notifications. Keep them private.

Pushover notifications are optional; the application can still display deal results without them.

## Running the Project

### Step 1 — Train the DNN and create ChromaDB

```bash
uv run train.py
```

This downloads the dataset, trains the Scikit-learn DNN, evaluates its predictions, and indexes the training products in ChromaDB.

Run this step initially or whenever rebuilding the model and vector database.

### Step 2 — Start the application

```bash
uv run app.py
```

Open the local URL displayed in the terminal, normally:

http://127.0.0.1:7860

### Step 3 — Find deals

1. Click **Find Deals** in the Gradio interface.
2. The Scanner Agent retrieves new product offers.
3. The DNN and RAG Agents estimate product prices.
4. The Ensemble Agent combines their predictions.
5. The Planning Agent ranks potential deals.
6. The Messaging Agent displays the result and sends a Pushover notification if configured and the savings threshold is met.

For subsequent runs, only `uv run app.py` is required unless the model or database needs rebuilding.

## Model Evaluation

The Scikit-learn neural network uses a multilayer perceptron (MLP) with three hidden layers.

- **Architecture:** 128 → 64 → 32 neurons
- **Text features:** TF-IDF
- **Target transformation:** Log-transformed product prices
- **Evaluation metric:** Mean Absolute Error (MAE)

**Initial DNN Test MAE: $73.97**

This means the model's predictions differ from actual test prices by approximately $73.97 on average.

The RAG Agent retrieves similar products from ChromaDB and uses their descriptions and prices as context for an LLM-generated price estimate.

The ensemble combines the two predictions using a weighted average.

## Limitations

- Predicted prices are estimates, not verified current market prices.
- Deal discovery depends on external RSS feed availability.
- Price prediction accuracy varies by product category.
- OpenAI API usage may incur costs.
- The workflow runs when triggered through the Gradio interface rather than continuously in the background.

## Future Improvements

- Evaluate and optimize the ensemble weights.
- Add RAG evaluation metrics and model comparison.
- Implement scheduled deal discovery.
- Add more product data sources.
- Deploy the Gradio application publicly.

## Project Summary

**The Price Is Right** is a multi-agent AI deal-hunting application that combines machine learning, retrieval-augmented generation, and LLM-powered agents to discover product offers and estimate their potential value. Built with LangGraph, Scikit-learn, ChromaDB, and Gradio, the system automates deal discovery, price prediction, ranking, and push notifications through Pushover.