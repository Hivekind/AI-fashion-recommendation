\c sales_data;

-- Create a temporary table with Date as TEXT to allow raw data import
CREATE TEMP TABLE temp_sales_product_details (
    date TEXT,  -- Store date as TEXT to allow for conversion
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
    latitude NUMERIC(9, 6),
    longitude NUMERIC(9, 6)
);

-- Load data into the temporary table
COPY temp_sales_product_details
FROM '/init-scripts/dataset/Sales_Product_Details.csv'
DELIMITER ','
CSV HEADER;


-- Create the final table
CREATE TABLE IF NOT EXISTS sales_product_details (
    date DATE,
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
    latitude NUMERIC(9, 6),
    longitude NUMERIC(9, 6)
);

-- Insert data from the temporary table into the final table with date conversion
INSERT INTO sales_product_details (
    date, customer_id, product_id, quantity, unit_price, sales_revenue, product_description, 
    product_category, product_line, raw_material, region, latitude, longitude
)
SELECT
    TO_DATE(date, 'YYYYMMDD'),  -- Convert the date format
    customer_id, product_id, quantity, unit_price, sales_revenue, product_description, 
    product_category, product_line, raw_material, region, latitude, longitude
FROM temp_sales_product_details;


-- Drop the temporary table
DROP TABLE temp_sales_product_details;
