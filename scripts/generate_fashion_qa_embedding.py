import psycopg
from langchain_openai import OpenAIEmbeddings
import tiktoken

# Configuration
ai_model = "text-embedding-3-small"
vector_dimension = 512

# Initialize the OpenAI embedding model and tokenizer
embedding_model = OpenAIEmbeddings(model=ai_model, dimensions=vector_dimension)
tokenizer = tiktoken.encoding_for_model(ai_model)

def generate_and_store_embeddings_for_fashion_qa():
    total_tokens_used = 0

    try:
        # Connect to PostgreSQL using psycopg3
        with psycopg.connect(
            dbname="sales_data",
            user="postgres",
            password="example",
            host="localhost",
            port="5432"
        ) as conn:
            with conn.cursor() as cur:
                # Select rows where the embedding in 'qa_embedding' is NULL (missing)
                cur.execute("""
                    SELECT id, qa FROM fashion_qa
                    WHERE qa_embedding IS NULL
                    ORDER BY id;
                """)
                rows = cur.fetchall()

                # Iterate through each row and generate embeddings for 'qa'
                for row in rows:
                    id, qa = row

                    # Generate the number of tokens used for the 'qa' field
                    tokens_used = len(tokenizer.encode(qa))
                    print(f"Tokens used for 'qa' {id}: {tokens_used}")

                    # Add to the total token count
                    total_tokens_used += tokens_used

                    # Generate embedding for the 'qa' field
                    embedding = embedding_model.embed_query(qa)

                    # Update the 'qa_embedding' field in the table for the corresponding id
                    cur.execute("""
                        UPDATE fashion_qa
                        SET qa_embedding = %s
                        WHERE id = %s;
                    """, (embedding, id))

                    # Commit the update for each row
                    conn.commit()

                # Output the total tokens used after the loop completes
                print(f"Total tokens used: {total_tokens_used}")

    except Exception as e:
        print(f"Error: {e}")

# Call the function to generate and store embeddings
generate_and_store_embeddings_for_fashion_qa()
