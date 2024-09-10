from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
import warnings
import argparse
import os
import psycopg
import json
import params


mongodb_conn_string = os.getenv("MONGODB_CONN_STRING")

# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Process arguments
parser = argparse.ArgumentParser(description='Fashion Shop Assistant')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

query = args.question

print("\nYour question:")
print("-------------")
print(query)

# Connect to MongoDB Atlas
client = MongoClient(mongodb_conn_string)
db = client[params.db_name]
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

print(f"User question: {query}\n")


################################
# get relevant docs from MongoDB
################################


def get_similarity_search():
  # Perform the similarity search
  similar_docs = vectorStore.similarity_search(query=query, include_scores=True)

  print("\nQuery Response:")
  print("---------------")

  # Access the closest matching document
  if similar_docs:
      # Iterate through each document and print its content
      for i, doc in enumerate(similar_docs):
          print(f"Doc {i+1}: {doc.page_content}\n\n")
          print(doc.metadata["score"], end="\n\n\n")

      closest_match = similar_docs[0]
      # print("Closest Match:", closest_match)
  else:
      print("No matching document found.")



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
    # print("\nRetriver:")
    # print("---------------")

    # for i, doc in enumerate(docs):
    #     print(f"Doc {i+1}: {doc.page_content}\n\n")
    #     print(doc.metadata, end="\n\n\n")

    return "\n\n".join(
      [f"{doc.page_content}" for doc in docs]
    )


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


first_response = rag_chain.invoke(query)

print("\nRAG Chain Response:")
print("-------------------")
print(first_response)


second_prompt = get_second_prompt()

items_chain = (
    {"context": RunnablePassthrough()}
    | second_prompt
    | llm
    | StrOutputParser()
)

second_response = items_chain.invoke(first_response)

print(second_response)



pg_connection = psycopg.connect(
      dbname="sales_data",
      user="postgres",
      password="example",
      host="localhost",
      port="5432"
  )

# Function to process the response
def process_fashion_suggestion(response_json, pg_connection):
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

        print(item, end="\n\n")
        print(similar_items)

        search_results.append({
            "item": item,
            "similar_items": similar_items
        })

    # Return the search results for each item
    return search_results



# Function to perform pgvector search in PostgreSQL using psycopg3
def search_similar_items(pg_connection, item_embedding):
    # Convert the embedding to a string in PostgreSQL array format
    embedding_str = '[' + ', '.join(map(str, item_embedding)) + ']'

    try:
        # Connect to the PostgreSQL database
        with pg_connection.cursor() as cursor:
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

            print(results)

            # Extract similar product description id from the results
            similar_items = [row[0] for row in results]
            return similar_items

    except Exception as e:
        print(f"Error during pgVector search: {str(e)}")
        return []




def get_product_display_name(conn, description_id, gender):
    # Map gender input to the corresponding gender in the database
    gender_filter = []

    if gender == "male":
        gender_filter = ['Men', 'Boys', 'Unisex']
    elif gender == "female":
        gender_filter = ['Women', 'Girls', 'Unisex']
    elif gender == "NA":
        gender_filter = ['Unisex', 'Men', 'Women', 'Boys', 'Girls']
  
    # Query to get the product display name using the description_id
    query = """
        SELECT product_display_name
        FROM products
        WHERE
          description_id = %s
          AND gender = ANY(%s)
        ORDER BY year DESC
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(query, (description_id, gender_filter))
        result = cur.fetchone()  # Fetch the first result
        if result:
            return result[0]  # Return the product_display_name
        else:
            return None  # Return None if no result is found


def process_suggestions(conn, search_results, gender):
    product_suggestions = []  # This will store the product_display_name for each suggestion
    
    # Loop through search_results
    for result in search_results:
        item = result["item"]
        similar_items = result["similar_items"]
        
        # Get the first matching ID from similar_items
        if similar_items:
            first_id = similar_items[0]  # Assuming similar_items is an array of IDs

            # Use the first_id to get the product_display_name from the products table
            product_display_name = get_product_display_name(conn, first_id, gender)
            
            if product_display_name:
                # Add the product display name to the suggestions list
                product_suggestions.append({
                    "item": item,
                    "suggestion": product_display_name
                })
    
    return product_suggestions  # Return the list of product suggestions



response_data = json.loads(second_response)

# if is not related to fashion suggestion, then output the response as is
if response_data.get("is_fashion_suggestion") == "no":
    print(first_response)
    exit()

gender = response_data.get("gender")

search_results = process_fashion_suggestion(second_response, pg_connection)

# Process the search results and get product suggestions
product_suggestions = process_suggestions(pg_connection, search_results, gender)


# Gather the suggested product names
suggestions = [suggestion['suggestion'] for suggestion in product_suggestions]

# Format the output for the end user
print(f"{first_response}\n")
print("We recommend the following items:\n")

# Print each suggested product in a numbered list
for idx, suggestion in enumerate(suggestions, 1):
    print(f"{idx}. {suggestion}")












# Close connections
client.close()

pg_connection.close()
