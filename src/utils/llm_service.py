import os
import google.generativeai as genai
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables. LLM features will be disabled.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("LLMService initialized with Gemini 2.0 Flash")

    def generate_text(self, prompt: str, max_tokens: int = 4096) -> str:
        if not self.model:
            return "LLM Service unavailable (Missing API Key)."
        
        try:
            logger.debug(f"Generating text with prompt: {prompt[:200]}...")
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            logger.debug(f"LLM Response: {response.text[:200]}...")
            return response.text
        except Exception as e:
            logger.error(f"LLM Generation failed: {str(e)}")
            return f"Error generating content: {str(e)}"

# Singleton instance
llm_service = LLMService()
