from huggingface_hub import login
from config.settings import HUGGINGFACE_TOKEN
import logging

logger = logging.getLogger("eduassist.hf_login")

def hf_auto_login():
    """
    Login to HuggingFace once at startup.
    Token is loaded from .env via settings — never hardcoded.
    """
    if HUGGINGFACE_TOKEN:
        try:
            login(token=HUGGINGFACE_TOKEN)
            logger.info("✅ HuggingFace login successful")
        except Exception as e:
            logger.warning("⚠️ HuggingFace login failed: %s", e)
    else:
        logger.warning("⚠️ HUGGINGFACE_TOKEN not set — gated models will fail.")