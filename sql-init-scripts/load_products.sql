\c fashion_data;

-- Products dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

-- Table to store the products dataset
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

-- Table to store product descriptions embedding
CREATE TABLE product_descriptions (
    id SERIAL PRIMARY KEY,
    description TEXT NOT NULL UNIQUE,
    embedding VECTOR(512)
);

-- Load the products dataset into the `products` table
COPY products(product_id, gender, master_category, sub_category, article_type, base_colour, season, year, usage, product_display_name)
FROM '/init-scripts/dataset/products.csv'
DELIMITER ','
CSV HEADER;


-- Add a new column `description` to the `products` table, which will be a concatenation of `base_colour`, `article_type`, `gender`, `usage`, and `season` columns.
ALTER TABLE products ADD COLUMN description text;

-- Populate the `description` column with the concatenated values
UPDATE products
SET description = CONCAT(base_colour, ' color ', article_type, ' for ', gender, ', ', usage, ', ', season);


-- Populate the `product_descriptions` table with unique descriptions from the `products` table
INSERT INTO product_descriptions (description)
SELECT DISTINCT description
FROM products
WHERE description IS NOT NULL
ON CONFLICT (description) DO NOTHING;


-- Create a foreign key in the `products` table to link the `description_id` to the `product_descriptions` table.
ALTER TABLE products ADD COLUMN description_id INTEGER;

UPDATE products
SET description_id = pd.id
FROM product_descriptions pd
WHERE products.description = pd.description;

ALTER TABLE products
ADD CONSTRAINT fk_product_description
FOREIGN KEY (description_id) REFERENCES product_descriptions(id);