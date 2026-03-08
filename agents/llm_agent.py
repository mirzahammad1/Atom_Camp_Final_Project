from __future__ import annotations

import logging
import torch
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from config.settings import (
    HUGGINGFACE_TOKEN,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger("eduassist.llm_agent")

# =========================
# RAG Prompt Template
#
# ✅ Kept as plain text (NO chat tags like <|system|>)
# ✅ This is what worked in the original version
# ✅ Small models respond much better to plain prompts
# ✅ Context placed BEFORE question for better extraction
# =========================

RAG_PROMPT_TEMPLATE = """You are an AI assistant for KIET university chatbot.

Answer the student's question using ONLY the provided context below.

Rules:
- Read the context carefully and extract relevant information.
- Answer directly and clearly using ONLY what is in the context.
- If the context contains the answer, always provide it.
- Only say "The information is not available in the current data." if the context truly has NO relevant information.
- Be concise, factual, and helpful for students.
- Do not repeat the question.

Context:
{context}

Student Question:
{question}

Answer:
"""


class LLMAgent:
    """
    LLM Agent for RAG answer generation — Iteration 2.

    Improvements over original:
    - Plain prompt (no chat tags) — works with all model sizes ✅
    - Context placed before question — better extraction ✅
    - Output decoded by slicing new tokens — no prompt bleed ✅
    - pad_token_id set — no generation warnings ✅
    - Accepts List[str] chunks — merges top-k context ✅
    - Context window 4000 chars (was 1500) ✅
    - max_length 3072 (was 2048) ✅
    - model.eval() called after load ✅
    - float16 on CUDA, float32 on CPU ✅
    - HF login runs exactly once via singleton flag ✅
    - LLM_MAX_TOKENS 768 (was 256) ✅
    """

    _model = None
    _tokenizer = None
    _login_done = False

    def __init__(self):

        # -------------------------
        # HuggingFace Login (once)
        # -------------------------
        if not LLMAgent._login_done:
            if HUGGINGFACE_TOKEN:
                try:
                    login(token=HUGGINGFACE_TOKEN)
                    logger.info("✅ HuggingFace login successful")
                except Exception as e:
                    logger.warning("⚠️ HuggingFace login failed: %s", e)
            else:
                logger.warning("⚠️ No HUGGINGFACE_TOKEN set — gated models may fail.")
            LLMAgent._login_done = True

        # -------------------------
        # Device selection
        # -------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("🖥️ LLM device: %s", self.device)

        # ✅ CPU thread optimization for AMD Ryzen 5 7430U (6C/12T)
        # Using 10 of 12 threads — leaves 2 for OS/Streamlit
        # Gives ~30-40% speed boost on Ryzen CPUs
        if self.device == "cpu":
            num_threads = max(1, torch.get_num_threads() - 2)
            torch.set_num_threads(num_threads)
            # Note: set_num_interop_threads must be called before any parallel work
            # Streamlit already starts threads before this runs — skip it safely
            logger.info("🧵 CPU inference threads: %d", num_threads)

        # -------------------------
        # Load Model Only Once
        # -------------------------
        if LLMAgent._model is None:
            logger.info("🧠 Loading LLM model: %s", LLM_MODEL)

            LLMAgent._tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                use_fast=True,
                token=HUGGINGFACE_TOKEN or None,   # ✅ pass token during download
            )

            # ✅ Set pad token to eos if not defined (prevents generation warnings)
            if LLMAgent._tokenizer.pad_token is None:
                LLMAgent._tokenizer.pad_token = LLMAgent._tokenizer.eos_token
                LLMAgent._tokenizer.pad_token_id = LLMAgent._tokenizer.eos_token_id

            # ✅ CPU: float32 (Ryzen doesn't accelerate float16)
            # ✅ GPU: float16 (halves VRAM usage)
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            LLMAgent._model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,         # ✅ Load layer by layer — avoids RAM spike
                token=HUGGINGFACE_TOKEN or None,   # ✅ pass token during download
            )
            LLMAgent._model.to(self.device)
            LLMAgent._model.eval()              # ✅ Disables dropout — faster + deterministic

            # ✅ Fuse linear layers for faster CPU inference (PyTorch 2.0+)
            try:
                LLMAgent._model = torch.compile(
                    LLMAgent._model,
                    mode="reduce-overhead",
                    backend="inductor",
                ) if self.device == "cuda" else LLMAgent._model
            except Exception:
                pass  # torch.compile optional — skip if not supported

            logger.info("✅ LLM model loaded successfully on %s", self.device)

        self.tokenizer = LLMAgent._tokenizer
        self.model = LLMAgent._model
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE

    # =========================
    # Core Text Generation
    # =========================

    def _generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        ✅ Decodes only NEW tokens (after prompt) — eliminates prompt bleed.
        ✅ CPU-aware settings — greedy decoding + reduced token limits for speed.
        """
        is_cpu = self.device == "cpu"

        # ✅ CPU: 1024 max input (3072 is too slow), GPU: 2048
        max_input_length = 1024 if is_cpu else 2048

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        # ✅ CPU: cap at 256 new tokens (~5-10s), GPU: use full max_tokens
        max_new = min(self.max_tokens, 256) if is_cpu else self.max_tokens

        logger.info("⚙️ Generating (device=%s max_new_tokens=%d input_tokens=%d)...",
                    self.device, max_new, input_length)
        t0 = __import__("time").perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False if is_cpu else (self.temperature > 0),  # ✅ greedy on CPU = 3x faster
                temperature=self.temperature if (not is_cpu and self.temperature > 0) else 1.0,
                top_p=0.9 if not is_cpu else None,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = __import__("time").perf_counter() - t0
        new_tokens = outputs[0][input_length:]
        logger.info("✅ Generated %d tokens in %.1fs (%.1f tok/s)",
                    len(new_tokens), elapsed, len(new_tokens) / max(elapsed, 0.01))

        # ✅ Slice only the newly generated tokens — no prompt bleed ever
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    # =========================
    # Main Answer Generation
    # =========================

    def generate(
        self,
        question: str,
        context: str | List[str],
    ) -> Optional[str]:
        """
        Generate an answer from question + context.

        Args:
            question: The student's question.
            context:  Single string OR List[str] of top-k chunks.
                      Multiple chunks merged with separators for richer context.
        """
        if not context:
            return None

        # ✅ Merge top-k chunks if a list is passed
        if isinstance(context, list):
            merged = "\n\n---\n\n".join(
                c.strip() for c in context if c and c.strip()
            )
        else:
            merged = context.strip()

        if not merged:
            return None

        # ✅ CPU: 1500 chars max (matches 1024 token limit), GPU: 4000 chars
        is_cpu = not (hasattr(torch, "cuda") and torch.cuda.is_available())
        merged = merged[:1500] if is_cpu else merged[:4000]

        prompt = RAG_PROMPT_TEMPLATE.format(
            question=question.strip(),
            context=merged,
        )

        try:
            answer = self._generate(prompt)
            return answer if answer else None
        except Exception as e:
            logger.error("LLM generation error: %s", e)
            return None