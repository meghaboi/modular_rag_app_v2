import os
from dotenv import load_dotenv
from langsmith import Client

def test_langsmith_connection():
    """Test connection to LangSmith"""
    # Load environment variables
    load_dotenv()
    
    # Check if the API key is set
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("Error: LANGCHAIN_API_KEY environment variable not set.")
        print("Please set your LangSmith API key in the .env file.")
        return False
    
    try:
        # Create a LangSmith client
        client = Client()
        
        # Test the connection by listing datasets
        datasets = client.list_datasets()
        print(f"Connection successful! Found {len(list(datasets))} datasets.")
        return True
    except Exception as e:
        print(f"Error connecting to LangSmith: {str(e)}")
        return False

if __name__ == "__main__":
    test_langsmith_connection()