CREATE DATABASE sales_data;

\c sales_data;

-- https://www.kaggle.com/datasets/ishanshrivastava28/sales-transaction-dataset-with-product-details?resource=download
CREATE TABLE sales_product_details (
    id SERIAL PRIMARY KEY,  -- Auto-incrementing primary key
    date DATE,  -- Assuming 'Date' is in YYYYMMDD format, you might want to use TEXT if not properly formatted
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price NUMERIC(10, 2),
    sales_revenue NUMERIC(12, 2),
    product_description VARCHAR(255),
    product_category VARCHAR(255),
    product_line VARCHAR(255),
    raw_material VARCHAR(255),
    region VARCHAR(255),
    latitude NUMERIC(9, 6),  -- Latitude with precision up to 6 decimal places
    longitude NUMERIC(9, 6)  -- Longitude with precision up to 6 decimal places
);


-- https://www.kaggle.com/datasets/fashionworldda/fashion-trend-dataset
CREATE TABLE fashion_data (
    id SERIAL PRIMARY KEY,  -- Auto-incrementing primary key
    product_id INT,
    product_name VARCHAR(255),
    gender VARCHAR(50),
    category VARCHAR(100),
    pattern VARCHAR(100),
    color VARCHAR(100),
    age_group VARCHAR(50),
    season VARCHAR(50),
    price NUMERIC(10, 2),
    material VARCHAR(100),
    sales_count INT,
    reviews_count INT,
    average_rating NUMERIC(2, 1),
    out_of_stock_times INT,
    brand VARCHAR(100),
    discount VARCHAR(10),
    last_stock_date DATE,
    wish_list_count INT,
    month_of_sale INT,
    year_of_sale INT
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

