import os
import logging
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0):
    """
    Factory function to get the appropriate LangChain Chat Model 
    based on environment variables, supporting OpenAI and Amazon Nova (via Bedrock).
    """
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    
    logger.info(f"Initializing LLM for provider: {provider} with temperature: {temperature}")

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
             raise ImportError("Please install langchain-openai to use OpenAI models: pip install langchain-openai")
             
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY is missing. Please set it in the backend/.env file.")
        
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=temperature
        )

    elif provider in ["bedrock", "nova"]:
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError("Please install langchain-aws to use Bedrock/Nova models: pip install langchain-aws boto3")
            
        # 1. Force Region to us-east-1 as requested
        region = "us-east-1"
        logger.info(f"Forcing AWS Region to: {region}")
        
        # 2. Determine Model ID
        if provider == "nova":
            model_id = os.getenv("NOVA_MODEL_ID", "amazon.nova-pro-v1:0")
        else:
            model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            
        logger.info(f"Initializing ChatBedrock with model: {model_id}, region: {region}")

        # 3. Create Client
        return ChatBedrock(
            model_id=model_id,
            region_name=region,
            provider="amazon" if "amazon" in model_id.lower() or "nova" in model_id.lower() else "anthropic",
            temperature=temperature,
            model_kwargs={
                "inferenceConfig": {
                     "topP": 0.9
                }
            } if "nova" in model_id.lower() else None
        )
    
    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")

# Map Studio UI aliases → concrete embedding model ids
_EMBEDDING_ALIASES = {
    "openai-3-small": "text-embedding-3-small",
    "openai-3-large": "text-embedding-3-large",
    "openai-ada-002": "text-embedding-ada-002",
    "bedrock-titan": "amazon.titan-embed-text-v1",
    "bedrock-titan-v1": "amazon.titan-embed-text-v1",
    "bedrock-titan-v2": "amazon.titan-embed-text-v2:0",
}


def get_embeddings(model_name: Optional[str] = None):
    """
    Returns the appropriate LangChain Embeddings model based on the provider.
    Allows for custom model selection to ensure handshake compatibility.
    """
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    resolved = _EMBEDDING_ALIASES.get(model_name or "", model_name)
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=resolved or "text-embedding-3-small")
    elif provider in ["bedrock", "nova"]:
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(
            region_name="us-east-1", 
            model_id=resolved or "amazon.titan-embed-text-v1"
        )
    else:
        return None
