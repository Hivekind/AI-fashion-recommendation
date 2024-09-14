\c sales_data;

COPY fashion_qa(qa_id, question, answer)
FROM '/init-scripts/dataset/fashion_qa.csv'
DELIMITER ','
CSV HEADER;