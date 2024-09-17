\c fashion_data;

-- Fashion Shop Q&A dataset: https://huggingface.co/datasets/Quangnguyen711/Fashion_Shop_Consultant

-- Table to store the fashion Q&A dataset
CREATE TABLE fashion_qa (
    id SERIAL PRIMARY KEY,
    qa_id INT NOT NULL,
    question TEXT,
    answer TEXT
);

-- Load the fashion Q&A dataset into the `fashion_qa` table
COPY fashion_qa(qa_id, question, answer)
FROM '/init-scripts/dataset/fashion_qa.csv'
DELIMITER ','
CSV HEADER;


-- Add a new column `qa` to store the concatenated question and answer and populate it.
ALTER TABLE fashion_qa
ADD COLUMN qa TEXT;

UPDATE fashion_qa
SET qa = '[Question]' || question || '[Answer]' || answer;


-- Create a new column `qa_embedding` to store the embeddings of the `qa` column.
ALTER TABLE fashion_qa
ADD COLUMN qa_embedding VECTOR(512);


-- Remove redundant rows with the same question and answer
DELETE FROM fashion_qa
WHERE id IN (
  SELECT id
  FROM (
    SELECT id, ROW_NUMBER() OVER (PARTITION BY qa ORDER BY id) AS rn
    FROM fashion_qa
  ) t
  WHERE t.rn > 1
);
