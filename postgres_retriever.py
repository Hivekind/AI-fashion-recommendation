from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
import psycopg


class FashionQARetriever(BaseRetriever):
    conn: psycopg.Connection
    table: str
    embedding_field: str
    text_field: str
    embedding_model: Embeddings
    top_k: int

    def _get_relevant_documents(self, query: str):
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)

        # Convert the embedding to a string in PostgreSQL array format
        query_embedding_str = '[' + ', '.join(map(str, query_embedding)) + ']'

        try:
            with self.conn.cursor() as cur:
                # Perform similarity search using pgvector's <=> operator
                cur.execute(f"""
                    SELECT id, {self.text_field}, 1 - ({self.embedding_field} <=> %s) AS similarity
                    FROM {self.table}
                    ORDER BY {self.embedding_field} <=> %s
                    LIMIT %s;
                """, (query_embedding_str, query_embedding_str, self.top_k))

                # Fetch the results
                rows = cur.fetchall()

            # Convert results to LangChain Document objects
            documents = [
                Document(page_content=row[1], metadata={"similarity": row[2], "id": row[0]})
                for row in rows
            ]

            return documents

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
