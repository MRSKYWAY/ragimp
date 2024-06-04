import openai
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Document
import logging
import sys

# Load environment variables from .env file
load_dotenv()

# # Set logging level to DEBUG for verbose output
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize API keys from environment variables for security
openai.api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

PERSIST_DIR = "./storage"

# Check if storage already exists
if not os.path.exists(PERSIST_DIR):
    # Load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

def google_search(query, api_key, cse_id, num_results=5):
    """Perform a Google search and return the top results."""
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return results.get('items', [])
    except Exception as e:
        logging.error(f"Error during Google search: {e}")
        return []

def extract_content(search_results):
    """Extract content from Google search results and create documents."""
    documents = []
    for result in search_results:
        title = result.get('title')
        snippet = result.get('snippet')
        link = result.get('link')
        content = f"{title}\n{snippet}\n{link}"
        documents.append(Document(text=content))  # Create Document objects
    return documents
def react_reasoning(prompt):
    """Generate reasoning using OpenAI's API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Think step by step and explain your reasoning. Decide whether to act (perform a new search) or reason with the given information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during reasoning: {e}")
        return ""

def react_retrieval_and_reasoning(initial_query, google_api_key, google_cse_id, num_results=5, iterations=3):
    """Perform iterative retrieval and reasoning."""
    all_documents = []
    query = initial_query
    
    for _ in range(iterations):
        search_results = google_search(query, google_api_key, google_cse_id, num_results)
        documents = extract_content(search_results)
        all_documents.extend(documents)
        
        # Use VectorStoreIndex to index documents
        store = VectorStoreIndex.from_documents(documents)
        query_engine = store.as_query_engine()
        
        # Get relevant information and reason about the next steps
        relevant_info = query_engine.query(query)
        reasoning_prompt = (
            f"Given the following information:\n\n{relevant_info}\n\n"
            "What should the next query be to find more detailed and relevant information? "
            "Or should we generate the final context now?"
        )
        reasoning_output = react_reasoning(reasoning_prompt)
        
        logging.debug(f"Reasoning Output: {reasoning_output}")
        
        if "generate the final context" in reasoning_output.lower():
            break
        
        # Extract new query from the reasoning output
        query = reasoning_output.split("Next query:")[-1].strip()
        
    return all_documents

def generate_context(prompt, max_tokens=300):
    """Generate context using OpenAI's API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
            
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during context generation: {e}")
        return ""

def research_agent(topic, user_insights, desired_words, google_api_key, google_cse_id, num_results=5, iterations = 3):
    """Main function to generate context based on the given topic and insights."""
    initial_query = f"{topic} {user_insights}"
    documents = react_retrieval_and_reasoning(initial_query, google_api_key, google_cse_id, num_results, iterations)
    
    
    # Use VectorStoreIndex to index documents
    store = VectorStoreIndex.from_documents(documents)
    query_engine = store.as_query_engine()
    
    # Query the indexed documents for relevant information
    query = f"{topic} {user_insights}"
    relevant_info = query_engine.query(query)
    
    # Generate the context with OpenAI's API
    context_prompt = f"Using the following information, generate a context of around {desired_words} words:\n\n{relevant_info}"
    context = generate_context(context_prompt, max_tokens=int(desired_words * 1.5))  # Adjust max tokens as needed
    
    return context

# Example usage
if __name__ == "__main__":
    topic = "Climate Change Impact on Agriculture"
    user_insights = "Focus on developing countries"
    desired_words = 500
    
    context = research_agent(topic, user_insights, desired_words, google_api_key, google_cse_id)
    print(context)
