\c sales_data;

COPY products(product_id, gender, master_category, sub_category, article_type, base_colour, season, year, usage, product_display_name)
FROM '/init-scripts/dataset/products.csv'
DELIMITER ','
CSV HEADER;