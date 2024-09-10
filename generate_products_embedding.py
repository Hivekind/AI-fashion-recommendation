import psycopg
from langchain_openai import OpenAIEmbeddings
import tiktoken


ai_model="text-embedding-3-small"
vector_dimension = 512

embedding_model = OpenAIEmbeddings(model=ai_model, dimensions=vector_dimension)
tokenizer = tiktoken.encoding_for_model(ai_model)


def generate_and_store_embeddings():
    i = 0
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
                # Select descriptions where embeddings are NULL (missing)
                cur.execute("""
                    SELECT id, description FROM product_descriptions
                    WHERE embedding IS NULL
                    ORDER BY id;
                """)
                rows = cur.fetchall()

                # Iterate through each description, generate embeddings, and update the table
                for row in rows:
                    id, description = row

                    # Generate the number of tokens used for the description
                    tokens_used = len(tokenizer.encode(description))
                    print(f"Tokens used for description {id}: {tokens_used}")

                    # Add to the total token count
                    total_tokens_used += tokens_used

                    # Generate embedding for the description
                    embedding = embedding_model.embed_query(description)

                    # Update the embedding in the table for the corresponding id
                    cur.execute("""
                        UPDATE product_descriptions
                        SET embedding = %s
                        WHERE id = %s;
                    """, (embedding, id))

                    # Commit the update for each row
                    conn.commit()


                    # SY: testing .....
                    # i += 1
                    # if i == 10:
                    #     print("Processed 3 records, exiting.")
                    #     break  # Exit after processing 3 rows
                      
                  # Output the total tokens used after the loop completes
                print(f"Total tokens used: {total_tokens_used}")


    except Exception as e:
        print(f"Error: {e}")



generate_and_store_embeddings()
