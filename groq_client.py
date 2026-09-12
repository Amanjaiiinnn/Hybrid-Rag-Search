import os
import requests

DEFAULT_SYSTEM_PROMPT = """You are an expert e-commerce shopping assistant for an electronics catalog.
Your goal is to answer the user's queries accurately using ONLY the retrieved product details and reviews provided in the context below.

Context:
{context}

Guidelines:
1. Base your answer strictly on the provided Context.
2. If multiple products are retrieved, compare them objectively (by price, specs, rating, or customer feedback).
3. Pay close attention to pricing constraints. For example, if the query is "Best laptop under ₹80k", filter the retrieved list for laptops that cost less than 80,000 INR and rank them by rating/reviews.
4. If the context does not contain sufficient details to answer, state clearly that "The provided catalog does not contain enough information to answer this specific query," but summarize the closest matches available.
5. Format your response in clean Markdown. Use tables for specifications and bold text for recommendations. Keep prices in Indian Rupees (₹).
"""

class GroqClient:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        """
        Initializes the Groq LLM client.
        :param api_key: Optional Groq API Key. If not provided, will look for GROQ_API_KEY environment variable.
        :param model: Groq model identifier (e.g. 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant').
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def is_configured(self):
        """Checks if the API key is set."""
        return len(self.api_key.strip()) > 0

    def generate_answer(self, query, retrieved_docs, system_prompt=None):
        """
        Generates an answer using Groq LLM based on the query and retrieved context documents.
        :param query: The user's query string.
        :param retrieved_docs: List of dicts, each representing a document/product with at least a 'text' key.
        :param system_prompt: Optional custom system prompt overrides.
        """
        if not self.is_configured():
            return "Error: Groq API Key is not set. Please provide a valid API Key in the sidebar."

        # Compile retrieved documents into a single context string
        context_parts = []
        for idx, doc in enumerate(retrieved_docs):
            doc_text = doc.get("text", "")
            # We can also include search scores or ranks if available
            score_info = f" (Score: {doc['score']:.4f})" if 'score' in doc else ""
            context_parts.append(f"Document [{idx+1}]{score_info}:\n{doc_text}\n")
            
        context = "\n".join(context_parts)
        
        # Build prompt messages
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.format(context=context)
        # In case the custom prompt needs context injection:
        if system_prompt and "{context}" in system_prompt:
            sys_prompt = system_prompt.format(context=context)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                return f"Groq API Error (Status {response.status_code}): {error_msg}"
        except requests.exceptions.Timeout:
            return "Error: Connection to Groq API timed out. Please try again."
        except Exception as e:
            return f"Error connecting to Groq API: {str(e)}"

if __name__ == "__main__":
    # Quick test (will fail or warn without key)
    client = GroqClient(api_key="gsk_mock_test_key")
    retrieved = [{"text": "Product Name: TechNova Laptop Model 9 | Price: INR 39,780 | Specs: i5, 12GB RAM, 256GB SSD | Reviews: Great value"}]
    ans = client.generate_answer("Recommend a laptop under 40k", retrieved)
    print(ans)
