from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from openai import OpenAI
import warnings
import argparse
import os
import psycopg
import json
import params

# Global debug mode flag
debug_mode = False  # Set to False to disable debug printing

def debug_print(*args, **kwargs):
    """Custom debug print function that behaves like print() but only prints when debug_mode is True."""
    if debug_mode:
        print(*args, **kwargs)


mongodb_conn_string = os.getenv("MONGODB_CONN_STRING")

# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Process arguments
parser = argparse.ArgumentParser(description='Fashion Shop Assistant')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

query = args.question

debug_print(f"User question: {query}\n")

# Connect to MongoDB Atlas
mongo_client = MongoClient(mongodb_conn_string)
db = mongo_client[params.db_name]
collection = db[params.collection_name]

ai_model = params.ai_model
vector_dimension = params.vector_dimension
index_name = params.index_name

# openAI embedding model
embeddings = OpenAIEmbeddings(model=ai_model, dimensions=vector_dimension)

# Initialize MongoDBAtlasVectorSearch with correct keys
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,  # Your embedding model
    text_key="text",  # Field in MongoDB for the text you want to retrieve
    embedding_key="embedding",  # Field in MongoDB for the stored embeddings
    index_name=index_name,  # Name of Vector Index in MongoDB Atlas
    relevance_score_fn="cosine"  # Use cosine similarity
)




################################
# get relevant docs from MongoDB
################################


def get_similarity_search():
  # Perform the similarity search
  similar_docs = vectorStore.similarity_search(query=query, include_scores=True)

  debug_print("\nQuery Response:")
  debug_print("---------------")

  # Access the closest matching document
  if similar_docs:
      # Iterate through each document and print its content
      for i, doc in enumerate(similar_docs):
          debug_print(f"Doc {i+1}: {doc.page_content}\n\n")
          debug_print(doc.metadata["score"], end="\n\n\n")

      # closest_match = similar_docs[0]
      # debug_print("Closest Match:", closest_match)
  else:
      debug_print("No matching document found.")



###################
# Set up RAG chain
###################

llm = ChatOpenAI(model="gpt-4o-mini")

search_kwargs = {
    "include_scores": True,
}

retriever = vectorStore.as_retriever(search_kwargs=search_kwargs)



def get_first_prompt():
    # Define the template as a string
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question} 
    Context: {context} 
    Answer:
    """

    # Create a PromptTemplate object with input variables
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Wrap it in a HumanMessagePromptTemplate
    human_message_prompt = HumanMessagePromptTemplate(prompt=prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    return chat_prompt

prompt = get_first_prompt()



def get_second_prompt():
    # Define the template as a string
    template = """
    If this is a fashion suggestion, please extract relevant items from this context. The items should include clothing, shoes, and accessories. 

    Additionally, based on the context, please guess the intended gender for the fashion suggestion. The gender should be one of the following:
    - "male" (if it seems intended for men)
    - "female" (if it seems intended for women)
    - "NA" (if there is no clear indication of gender)

    Return the response in the following JSON format:

    {{
      "is_fashion_suggestion": "yes" / "no",
      "items": [list of items],
      "gender": "male" / "female" / "NA"
    }}

    If the context is not related to fashion, return this:

    {{
      "is_fashion_suggestion": "no"
    }}

    For example, if the text is about fashion and contains suggestions like "black dress" or "casual sandals," return:

    {{
      "is_fashion_suggestion": "yes",
      "items": ["black dress", "casual sandals"],
      "gender": "female"
    }}

    If the text contains fashion suggestions like "sneakers" or "t-shirts," and there is no clear gender, return:

    {{
      "is_fashion_suggestion": "yes",
      "items": ["sneakers", "t-shirts"],
      "gender": "NA"
    }}

    Context: {context} 
    Answer:
    """

    # Create a PromptTemplate object with input variables
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template=template
    )

    # Wrap it in a HumanMessagePromptTemplate
    human_message_prompt = HumanMessagePromptTemplate(prompt=prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    return chat_prompt





def format_docs(docs):
    # debug_print("\nRetriver:")
    # debug_print("---------------")

    # for i, doc in enumerate(docs):
    #     debug_print(f"Doc {i+1}: {doc.page_content}\n\n")
    #     debug_print(doc.metadata, end="\n\n\n")

    return "\n\n".join(
      [f"{doc.page_content}" for doc in docs]
    )


first_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


first_response = first_rag_chain.invoke(query)

debug_print("\n First RAG Chain Response:")
debug_print("-------------------")
debug_print(first_response)


second_prompt = get_second_prompt()

second_rag_chain = (
    {"context": RunnablePassthrough()}
    | second_prompt
    | llm
    | StrOutputParser()
)

second_response = second_rag_chain.invoke(first_response)

debug_print("\n Second RAG Chain Response:")
debug_print("-------------------")
debug_print(second_response)


pg_connection = psycopg.connect(
      dbname="sales_data",
      user="postgres",
      password="example",
      host="localhost",
      port="5432"
  )


def get_similar_fashion_descriptions(response_json, pg_connection):
    # Parse the JSON response
    response_data = json.loads(response_json)

    # Check if the suggestion is related to fashion
    if response_data.get("is_fashion_suggestion") == "no":
        # If it's not a fashion suggestion, don't process further
        return response_data

    # Extract items from the response
    items = response_data.get("items", [])
    search_results = []

    # Loop through each item and perform a pgvector search in the PostgreSQL table
    for item in items:
        item_embedding = embeddings.embed_query(item)
        similar_items = search_similar_items(pg_connection, item_embedding)

        debug_print(item, end="\n\n")
        debug_print(similar_items)

        search_results.append({
            "item": item,
            "similar_items": similar_items
        })

    # Return the search results for each item
    return search_results


def search_similar_items(pg_connection, item_embedding):
    # Convert the embedding to a string in PostgreSQL array format
    embedding_str = '[' + ', '.join(map(str, item_embedding)) + ']'

    try:
        # Connect to the PostgreSQL, use DictRow factory to get results as dictionaries
        with pg_connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            # Perform an embedding search using pgvector

            # For cosine similarity, the range is [-1, 1], where:
            # 1 means the vectors are identical.
            # 0 means the vectors are orthogonal (completely dissimilar).
            # -1 means the vectors are opposite.

            # For cosine distance, use: 1 - cosine similarity
            # [1-(-1),  1-1] => [2, 0]
            # where:
            # 2 means opposite
            # 0 means identical

            query = """
                SELECT id, description, embedding <=> %s as distance
                FROM product_descriptions
                ORDER BY embedding <=> %s
                LIMIT 5;
            """

            # Execute the query, passing the embedding as a parameter
            cursor.execute(query, (embedding_str, embedding_str,))
            results = cursor.fetchall()

            debug_print(results)

            # with cosine distance > 0.6, the items appear irrelevant, so we filter them out
            similar_items = [row['id'] for row in results if row['distance'] <= 0.6]

            return similar_items

    except Exception as e:
        print(f"Error during pgVector search: {str(e)}")
        return []



# Generate SQL query using the AI model
def generate_sql_via_ai(description_id, gender):
    debug_print("description_id: ", description_id)
    debug_print("gender: ", gender)

    gender_filter = ""

    if gender == "male":
        gender_filter = " and, filter 'gender' to include 'Men', 'Boys', and 'Unisex'."
    elif gender == "female":
        gender_filter = " and, filter 'gender' to include 'Women', 'Girls', and 'Unisex'."
    elif gender == "NA":
        gender_filter = ""

    # Construct a natural language instruction for the AI model
    instruction = f"""
    You are tasked with generating an SQL query for a PostgreSQL table named 'products'. The table has the following fields:

    - id: integer (Primary Key)
    - product_id: integer
    - product_display_name: string
    - description_id: integer
    - year: integer
    - gender: string (values: 'Women', 'Men', 'Boys', 'Girls', 'Unisex')

    The query should retrieve one record where 'description_id' matches {description_id}
    {gender_filter}

    The query should only return these fields: 'id', 'product_id', 'product_display_name'.
    The result should be ordered by 'year' in descending order and limited to 1 record.

    Provide only the SQL query in plain text, without any explanation, formatting or new lines.
"""

    debug_print(instruction)

    completion = OpenAI().chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
      ]
    )

    sql_query = completion.choices[0].message.content
    debug_print(sql_query)

    return sql_query


def retrieve_product(conn, description_id, gender):
    sql_query = generate_sql_via_ai(description_id, gender)

    # Execute the AI-generated SQL query, using dict_row to return results as a dictionary
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql_query)
        result = cur.fetchone()

    # Return result as a dictionary, or None if no record is found
    return result if result else None



def get_similar_fashion_products(conn, search_results, gender):
    product_suggestions = []
    
    for result in search_results:
        item = result["item"]
        similar_items = result["similar_items"]

        # get the first similar item, if found
        if similar_items:
            for description_id in similar_items:
                # Try to retrieve the product based on description_id and gender
                product = retrieve_product(conn, description_id, gender)

                # If a valid product is found, break the loop and return it
                if product:
                    product_suggestions.append({
                        "item": item,
                        "suggestion": product
                    })
                    break

    return product_suggestions  # Return the list of product suggestions



response_data = json.loads(second_response)

# if is not related to fashion suggestion, then output the response as is
if response_data.get("is_fashion_suggestion") == "no":
    print(first_response)
    exit()

gender = response_data.get("gender")

# Step 1: Vector embedding search in product_descriptions table
search_results = get_similar_fashion_descriptions(second_response, pg_connection)

# Step 2: Product lookup in products table based on description_id from step 1
product_suggestions = get_similar_fashion_products(pg_connection, search_results, gender)


# Gather the suggested product names
suggestions = [suggestion.get("suggestion") for suggestion in product_suggestions]

# Format the output for the end user
print(f"{first_response}\n")
print("Our top picks for you:\n")

# Print each suggested product in a numbered list
for idx, suggestion in enumerate(suggestions, 1):
    product_id = suggestion.get("product_id")
    product_display_name = suggestion.get("product_display_name")

    print(f"{idx}. {product_display_name}")



# Close DB connections
mongo_client.close()
pg_connection.close()
