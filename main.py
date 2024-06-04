import openai
import os
from googleapiclient.discovery import build
from llama_index.core import Document, SimpleDocumentStore, QueryEngine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize API keys from environment variables for security
openai.api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

def google_search(query, api_key, cse_id, num_results=5):
    """Perform a Google search and return the top results."""
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return results.get('items', [])
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

def extract_content(search_results):
    """Extract content from Google search results and create documents."""
    documents = []
    for result in search_results:
        title = result.get('title')
        snippet = result.get('snippet')
        link = result.get('link')
        content = f"{title}\n{snippet}\n{link}"
        documents.append(Document(content))
    return documents

def generate_context(prompt, max_tokens=300):
    """Generate context using OpenAI's API."""
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error during context generation: {e}")
        return ""

def research_agent(topic, user_insights, desired_words, google_api_key, google_cse_id, num_results=5):
    """Main function to generate context based on the given topic and insights."""
    search_results = google_search(topic, google_api_key, google_cse_id, num_results)
    documents = extract_content(search_results)
    
    # Use LlamaIndex to index documents
    store = SimpleDocumentStore(documents)
    engine = QueryEngine(store)
    
    # Query the indexed documents for relevant information
    query = f"{topic} {user_insights}"
    relevant_info = engine.query(query)
    
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
