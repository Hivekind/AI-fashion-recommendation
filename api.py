from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from openai import OpenAI
import warnings
import os
import psycopg
import json
import params
from flask import url_for
from postgres_retriever import FashionQARetriever


# Global debug mode flag
debug_mode = True  # Set to False to disable debug printing


# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

mongodb_conn_string = os.getenv("MONGODB_CONN_STRING")

ai_model = params.ai_model
vector_dimension = params.vector_dimension

def debug_print(*args, **kwargs):
    """Custom debug print function that behaves like print() but only prints when debug_mode is True."""
    if debug_mode:
        print(*args, **kwargs)


def process_user_query(query):
    debug_print(f"User question: {query}\n")

    llm = ChatOpenAI(model="gpt-4o-mini")

    # openAI embedding model
    embedding_model = OpenAIEmbeddings(model=ai_model, dimensions=vector_dimension)

    # Use `with` to manage the PostgreSQL connection
    with psycopg.connect(
        dbname="fashion_data",
        user="postgres",
        password="example",
        host="localhost",
        port="5432"
    ) as pg_connection:
        # The connection will be automatically closed when the block is exited
        first_response = first_RAG_chain(query, llm, embedding_model, pg_connection)

        second_response = second_chain(first_response, llm)

        final_response = process_fashion_suggestion(first_response, second_response, embedding_model, pg_connection)

    return final_response



############################
# 1st RAG chain: PgVector Similarity Search -> format_docs -> prompt -> llm -> StrOutputParser
############################

def first_RAG_chain(query, llm, embedding_model, pg_connection):
    # custome fashion qa retriever from postgres
    retriever = FashionQARetriever(
        conn=pg_connection,
        table="fashion_qa",
        embedding_field="qa_embedding",
        text_field="qa",
        embedding_model=embedding_model,
        top_k=4  # Fetch top 4 results
    )

    prompt = get_first_prompt()

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

    return first_response



############################
# 2nd RAG chain: RunnablePassthrough -> second_prompt -> llm -> StrOutputParser
############################

def second_chain(first_response, llm):
    prompt = get_second_prompt()

    chain = (
        {"context": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(first_response)

    debug_print("\n Second Chain Response:")
    debug_print("-------------------")
    debug_print(response)

    return response


############################
# process the fashion suggestion
############################

def process_fashion_suggestion(first_response, second_response, embeddingModel, pg_connection):
    # Parse the JSON response
    response_data = json.loads(second_response)
    gender = response_data.get("gender")

    # not fashion suggestion related, return as is
    if response_data.get("is_fashion_suggestion") == "no":
        print(first_response)
        return { "response" : first_response }

    # Step 1: Vector embedding search in product_descriptions table
    search_results = get_similar_fashion_descriptions(second_response, pg_connection, embeddingModel)

    # Step 2: Product lookup in products table based on description_id from step 1
    product_suggestions = get_similar_fashion_products(pg_connection, search_results, gender)

    # Gather the suggested product names
    suggestions = [suggestion.get("suggestion") for suggestion in product_suggestions]

    final_response = format_response(first_response, suggestions)
    debug_print(final_response)

    return final_response


# Format the response with the assistant's response and product suggestions
def format_response(first_response, suggestions):
    # Format the main response and suggestions
    final_response = {}
    
    # Add the assistant's response
    final_response["response"] = first_response

    # Add the product suggestions
    final_response["items"] = []
    
    for idx, suggestion in enumerate(suggestions, 1):
        product_id = suggestion.get("product_id")

        product_display_name = suggestion.get("product_display_name")
        final_response["items"].append({
            "index": idx,
            "name": product_display_name,
            "image_url": url_for('static', filename=f'images/{product_id}.jpg', _external=True)
        })

    return final_response


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
    debug_print("\nRetriever:")
    debug_print("---------------")

    for i, doc in enumerate(docs):
        debug_print(f"Doc {i+1}: {doc.page_content}\n\n")
        debug_print(doc.metadata, end="\n\n\n")

    return "\n\n".join(
      [f"{doc.page_content}" for doc in docs]
    )



def get_similar_fashion_descriptions(response_json, pg_connection, embeddingModel):
    # Parse the JSON response
    response_data = json.loads(response_json)

    # Check if the suggestion is related to fashion
    if response_data.get("is_fashion_suggestion") == "no":
        # If it's not a fashion suggestion, don't process further
        return response_data

    # Extract items from the response
    items = response_data.get("items", [])
    gender = response_data.get("gender")

    # Check if gender is either 'male' or 'female'
    if gender in ['male', 'female']:
        # Update each item by concatenating it with the gender
        items = [f"{item} for {gender}" for item in items]

    print("\n\nItems: ", items, end="\n\n")

    search_results = []

    item_embeddings = embeddingModel.embed_documents(items)

    # Loop through each item and perform a pgvector search in the PostgreSQL table
    for item, item_embedding in zip(items, item_embeddings):
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

            # with cosine distance > 0.7, the items appear irrelevant, so we filter them out
            similar_items = [row['id'] for row in results if row['distance'] <= 0.7]

            return similar_items

    except Exception as e:
        print(f"Error during pgVector search: {str(e)}")
        return []



# Generate SQL query using the AI model
def generate_sql_via_ai(gender):
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

    The query should retrieve one record where 'description_id' is a placeholder value, represented as %s,
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
    print(f"\n{sql_query}\n")

    return sql_query


def retrieve_product(conn, sql_query, description_id):
    # Execute the AI-generated SQL query, using dict_row to return results as a dictionary
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql_query, (description_id,))
        result = cur.fetchone()

    # Return result as a dictionary, or None if no record is found
    return result if result else None



def get_similar_fashion_products(conn, search_results, gender):
    product_suggestions = []
    
    # Generate the SQL query once
    sql_query = generate_sql_via_ai(gender)

    for result in search_results:
        item = result["item"]
        similar_items = result["similar_items"]

        # get the first similar item, if found
        if similar_items:
            for description_id in similar_items:
                # Try to retrieve the product based on description_id and gender
                product = retrieve_product(conn, sql_query, description_id)

                # If a valid product is found, break the loop and return it
                if product:
                    product_suggestions.append({
                        "item": item,
                        "suggestion": product
                    })

                    break

    return product_suggestions  # Return the list of product suggestions
