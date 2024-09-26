# Prerequisites

* [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

We use it to manage products images from the products dataset. Anyone cloning this repo must install Git LFS first, then clone the repo.

* Open AI [API key](https://platform.openai.com/account/api-keys)

We use OpenAI model to generate embedding and as LLM.


# Fashion AI Assistant Chat App

This is an AI-powered Fashion Assistant chat app that allows users to ask questions about shop policies or request product recommendations. The assistant leverages previous fashion-related Q&A data and intelligently suggests fashion products based on user queries.

The app operates in three main stages:

1. User Query: The user submits a question related to fashion products or shop policies.

2. Contextual Response Generation: The AI processes the question and retrieves relevant information from past Q&A data, providing a contextual answer to the user.

3. Fashion Item Recommendations: If the user's query relates to fashion products, the assistant analyzes the response and suggests relevant fashion items. These recommendations are drawn from a separate product dataset that is matched to the query context.

To illustrate this archicture, please refer the [flow diagram](flow_diagram/flow_diagram.html).

By combining a large Q&A dataset with a general fashion product dataset, the assistant provides informative answers and personalized product suggestions. For more details on the data structure, refer to the subsequent sections.

# Quick Start Steps

## First time set up

1. Set up environment variable for OpenAI API key
```sh
export OPENAI_API_KEY=<your-api-key>
```

2. bring up PostgreSQL docker container
```sh
docker compose up -d
```

3. Run the Database Setup Script

To set up the database for the first time (creating the `fashion_data` database, installing the `pgvector` extension, and restoring the dump file), run the following command:

```sh
docker compose exec db /bin/bash /init-scripts/01_restore_dump.sh
```

4. Create python virtual environment and install dependencies
```sh
# create python virtual env
python3 -m venv env

# activate the env
source env/bin/activate

# install dependencies
pip3 install -r requirements.txt
```

5. start python backend server, and access it at: http://localhost:5000/
```sh
python app.py
```

## Subsequent run

```sh
docker compose up -d

source env/bin/activate

python app.py
```

# Detailed Description

## Dataset

We use these 2 datasets for the app:

1. [Fashion Shop's Q&A](https://huggingface.co/datasets/Quangnguyen711/Fashion_Shop_Consultant)

This contains a fashion shop Q&A.


2. [Fashion Products Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

This contains fashion products details with images.


In the following sections, for SQL scripts, we use `\i` to run the scripts in the PostgreSQL database. These scripts are executed within the db container.

To connect to the PostgreSQL database, use the following command:

```sh
docker compose exec db bash -c "psql -U postgres fashion_data"
```

We have performed these steps to set up the datasets in PostgreSQL database (DB):

### Create database and enable [pgvector](https://github.com/pgvector/pgvector) extension.

We use the pgvector extension in PostgreSQL to store embedding vectors and perform efficient similarity searches.

```sql
\i /init-scripts/setup_database.sql
```


### Set up Fashion Shop Q&A dataset

The dataset is stored in `fashion_qa` table, which contains questions and answers related to fashion. We add a new column `qa` to store the concatenated question and answer. We then create a new column `qa_embedding` to store the embeddings of the `qa` column. We also remove redundant rows with the same question and answer.

```sql
\i /init-scripts/load_fashion_qa.sql
```

```sh
python generate_fashion_qa_embedding.py
```


### Set up Products dataset

1. CSV data

For some rows, the `productDisplayName` field contains `,`, which causes error while trying to ingest the csv into DB. We run this script to detect the occurence, then manually fix the data by replacing `,` with `-`.

We have also manually fixed the data where there is mismatch between productDisplayName and other fields, eg: subCategory, acticleType, etc.

```sh
python detect_malformed_csv.py
```

The dataset is stored in `products` table. We add a new column `description` to the `products` table, and populate it with the concatenation of `base_colour`, `article_type`, `usage`, and `season` columns.

We then create a new table `product_descriptions` to store unique descriptions and their embeddings. We then populate the `product_descriptions` table with unique descriptions from the `products` table and create a foreign key in the `products` table to link the `description_id` to the `product_descriptions` table.

```sql
\i /init-scripts/load_products.sql
```

Generate embeddings for the product descriptions.

```sh
python generate_products_embedding.py
```

2. Images

The dataset contains all images for the products. We placed these images in `static/images` dir, so the python backend is able to serve these images to the frontend.


### Set up PostgreSQL dump file

We run this command to create a dump file for the `fashion_data` database. This dump file is used to restore the database when setting up the app for the first time.

```sh
docker compose exec db bash -c "pg_dump -U postgres -Fc fashion_data > /init-scripts/fashion_data.dump"
```

## Managing Image Files with Git LFS

We use Git LFS to manage product image files in this repository.

### Initial Setup
- Install Git LFS:

```sh
# for macOS
brew install git-lfs

# for Ubuntu
sudo apt install git-lfs
```

- Set up Git LFS to track image files:

```sh
# Initialize Git LFS
git lfs install

# Track image files
git lfs track "*.jpg" "*.png" "*.gif"

# Add and commit
git add .gitattributes
git add .
git commit -m "Add images and track with Git LFS"
git push origin main
```
