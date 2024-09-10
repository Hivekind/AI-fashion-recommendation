\c sales_data;

-- Create a temporary table with last_stock_date as TEXT to allow raw data import
CREATE TEMP TABLE fashion_data_temp (
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
    last_stock_date TEXT,  -- Use TEXT to load as-is
    wish_list_count INT,
    month_of_sale INT,
    year_of_sale INT
);


COPY fashion_data_temp
FROM '/init-scripts/dataset/fashion_data_2018_2022.csv'
DELIMITER ','
CSV HEADER;


INSERT INTO fashion_data (
    product_id, product_name, gender, category, pattern, color, age_group, season, price,
    material, sales_count, reviews_count, average_rating, out_of_stock_times, brand,
    discount, last_stock_date, wish_list_count, month_of_sale, year_of_sale
)
SELECT
    product_id, product_name, gender, category, pattern, color, age_group, season, price,
    material, sales_count, reviews_count, average_rating, out_of_stock_times, brand,
    discount, TO_DATE(last_stock_date, 'DD/MM/YYYY'), wish_list_count, month_of_sale, year_of_sale
FROM fashion_data_temp;


-- Drop the temporary table
DROP TABLE fashion_data_temp;
