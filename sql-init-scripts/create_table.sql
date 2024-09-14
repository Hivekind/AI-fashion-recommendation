CREATE DATABASE sales_data;

\c sales_data;

-- https://huggingface.co/datasets/Quangnguyen711/Fashion_Shop_Consultant
CREATE TABLE fashion_qa (
    id SERIAL PRIMARY KEY,
    qa_id INT NOT NULL,
    question TEXT,
    answer TEXT
);

-- https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_id INT NOT NULL,
    gender VARCHAR(50),
    master_category VARCHAR(100),
    sub_category VARCHAR(100),
    article_type VARCHAR(100),
    base_colour VARCHAR(100),
    season VARCHAR(50),
    year INT,
    usage VARCHAR(100),
    product_display_name TEXT
);
