## Description
We'll build an **AI fashion assistant**, using a fashion shop dataset from HuggingFace for indexing, and set up a RAG chain to process user queries and generate responses.

### Retrieval-Augmented Generation (RAG)
RAG a technique that enhances the knowledge of language models by integrating additional data. A RAG application has two main components:

#### 1. Indexing:
Ingest and index data from a specific source, typically done offline.

#### 2. Retrieval and Generation:
During runtime, process the user's query, retrieve relevant data, and generate a response.

The data used in this project is the Fashion Shop Dataset from HuggingFace. The data consists of questions and answers related to fashion. The data is stored in MongoDB Atlas and the embeddings are generated using LangChain. The user query is passed to LangChain to generate an embedding and then the most similar data is retrieved from MongoDB Atlas.

- Project: Shopping Assistant
- Dataset: [Fashion Shop Dataset](https://huggingface.co/datasets/Quangnguyen711/Fashion_Shop_Consultant) from HuggingFace


## Prerequisites
* [MongoDB Atlas Subscription](https://cloud.mongodb.com/) (Free Tier is fine)
* Open AI [API key](https://platform.openai.com/account/api-keys)
* LangChain [API key](https://docs.smith.langchain.com/)


## Quick Start Steps
- Setup env var for OpenAI API key and MongoDB connection string
```zsh
export OPENAI_API_KEY=<your-api-key>
export MONGODB_CONN_STRING=<your-conn-string>
```

- Setup env var for LangChain API key and tracing, required for: `hub.pull(...)` in the script
```zsh
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
```

- Create a new Python environment
```zsh
python3 -m venv env
```

- Activate the new Python environment
```zsh
source env/bin/activate
```

- Install the requirements
```zsh
pip3 install -r requirements.txt
```

### 1. Indexing:

Embedding is generated from the combination of `question` + `answer` fields. This is based on the assumption that we want to match user query against both the question and answer fields in the database. The embedding is stored in MongoDB and index is created for vector search.

#### Load, Transform, Embed and Store

Run the below script:

```zsh
python3 ingest.py
```

### 2. Retrieval and Generation:

Run the below script, by passing your prompt as an argument.

```zsh
python3 query.py  -q "What is the store policy for returns?"
```


## Google Colab Notebook
You can also run the code in a Google Colab notebook. The notebook is available [here](RAG_fashion_langchain_openai_mongodb.ipynb).



## Resources
* [MongoDB Atlas](https://cloud.mongodb.com/)
* [Open AI API key](https://platform.openai.com/account/api-keys)
* [LangChain Doc](https://python.langchain.com)
* [LangChain API key](https://docs.smith.langchain.com/)
* [MongoDB Atlas module](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas)  
* [Open AI module](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/openai)


## pgVector

* [pgVector](https://github.com/pgvector/pgvector)

```
CREATE EXTENSION vector;
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```


## Dataset

* [sales-transaction-dataset-with-product-details](https://www.kaggle.com/datasets/ishanshrivastava28/sales-transaction-dataset-with-product-details)

* [fashion-trend-dataset](https://www.kaggle.com/datasets/fashionworldda/fashion-trend-dataset)

* [fashion-shop-consultant](https://huggingface.co/datasets/Quangnguyen711/Fashion_Shop_Consultant)



## postgres

1. disable pager
```sh
\pset pager off
```


2. SQL

```sql
\c sales_data

-- products table: add description column
ALTER TABLE products ADD COLUMN description text;

UPDATE products
SET description = CONCAT(base_colour, ' color ', article_type, ' for ', gender, ', ', usage, ', ', season);

-- add pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- new table to store descriptions embedding
CREATE TABLE product_descriptions (
    id SERIAL PRIMARY KEY,
    description TEXT NOT NULL UNIQUE,
    embedding VECTOR(512)
);

-- populate table with unique descriptions
INSERT INTO product_descriptions (description)
SELECT DISTINCT description
FROM products
WHERE description IS NOT NULL
ON CONFLICT (description) DO NOTHING;

-- create FK in products table
ALTER TABLE products ADD COLUMN description_id INTEGER;

UPDATE products
SET description_id = pd.id
FROM product_descriptions pd
WHERE products.description = pd.description;

ALTER TABLE products
ADD CONSTRAINT fk_product_description
FOREIGN KEY (description_id) REFERENCES product_descriptions(id);
```



3. NLP: download en_core_web_sm model

```sh
python -m spacy download en_core_web_sm
```

The en_core_web_sm model gets installed as a package in your Python environment. It is typically downloaded and stored in the site-packages directory of your Python environment. This package can be loaded directly using spacy.load('en_core_web_sm').


## start python web server

```sh
python app.py
```

Then open the browser and go to http://localhost:5000/


## Managing Image Files with Git LFS

We use Git LFS to manage a large number of small image files in this repository.

### Initial Setup
- Install Git LFS:

```bash
# for macOS
brew install git-lfs

# for Ubuntu
sudo apt install git-lfs
```

- Set up Git LFS to track image files:

```bash
# Initialize Git LFS
git lfs install

# Track image files
git lfs track "*.jpg" "*.png" "*.gif"

# Add and commit
git add .gitattributes
git add .
git commit -m "Add images and track with Git LFS"

# Push changes
git push origin main
```

### For Cloning the Repo
Anyone cloning this repo must install Git LFS first, then clone:

```bash
git clone <repo-url>
```
