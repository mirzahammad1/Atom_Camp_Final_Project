from __future__ import annotations

import re
import time
import logging
from typing import Dict, List, Optional

from agents.faq_agent import FAQAgent
from agents.vectordb_agent import VectorDBAgent
from agents.webscraper_agent import WebScraperAgent
from agents.llm_agent import LLMAgent
from agents.consistency_check_agent import ConsistencyCheckAgent
from agents.summarizing_agent import SummarizingAgent

# ── Tool Layer ────────────────────────────────────────────────────────────────
from tools.email_draft_tool import EmailDraftTool
from tools.document_generator_tool import DocumentGeneratorTool
from tools.academic_info_tool import AcademicInfoTool
from agents.response_log_agent import ResponseLogAgent

logger = logging.getLogger("eduassist.orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# Intent constants
# ══════════════════════════════════════════════════════════════════════════════
INTENT_RAG    = "RAG"
INTENT_ACTION = "ACTION"
INTENT_HYBRID = "HYBRID"


# ══════════════════════════════════════════════════════════════════════════════
# Email patterns  (checked FIRST — highest priority)
# ══════════════════════════════════════════════════════════════════════════════
_EMAIL_PATTERNS = [
    r"\bdraft\b.*\bemail\b",
    r"\bwrite\b.*\bemail\b",
    r"\bcompose\b.*\bemail\b",
    r"\bmake\b.*\bemail\b",
    r"\bgenerate\b.*\bemail\b",
    r"\bhelp\b.*\bemail\b",
    r"\bneed\b.*\bemail\b",
    r"\bsend\b.*\bemail\b",
    r"\bemail\b.*\bdraft\b",
    r"\bemail\b.*\bto\b.*\b(professor|department|admin|registrar|dean|faculty|sir|madam|office)\b",
    r"\bi\b.*\bwant\b.*\bemail\b",
    r"\bwrite\b.*\b(letter|request|complaint|application)\b",
    r"\bdraft\b.*\b(letter|request|complaint|application)\b",
    r"\bmake\b.*\b(letter|request|complaint|application)\b",
    r"\bgenerate\b.*\b(letter|request|complaint|application)\b",
    r"\bapplication\b.*\b(leave|absence|extension|fee|scholarship|internship)\b",
    r"\bleave\b.*\b(application|request|letter)\b",
    r"\bwrite\b.*\bto\b.*\b(professor|sir|madam|teacher|department|admin|dean)\b",
    r"\bdraft\b.*\bto\b.*\b(professor|sir|madam|teacher|department|admin|dean)\b",
]

# ══════════════════════════════════════════════════════════════════════════════
# Document patterns
# ══════════════════════════════════════════════════════════════════════════════
_DOCUMENT_PATTERNS = [
    r"\bgenerate\b.*\b(document|doc|certificate|report|noc|slip|affidavit)\b",
    r"\bcreate\b.*\b(document|doc|certificate|report|noc|slip)\b",
    r"\bmake\b.*\b(document|doc|certificate|report|noc|slip)\b",
    r"\bprepare\b.*\b(document|doc|certificate|report|noc|slip)\b",
    r"\bdraft\b.*\b(noc|bonafide|character certificate|experience letter|internship letter|recommendation)\b",
    r"\bwrite\b.*\b(noc|bonafide|character certificate|experience letter|internship letter|recommendation)\b",
    r"\b(noc|no.?objection certificate)\b",
    r"\bbonafide\b",
    r"\bcharacter certificate\b",
    r"\bexperience letter\b",
    r"\binternship letter\b",
    r"\brecommendation letter\b",
    r"\b(generate|create|make|prepare|draft|write)\b.*\b(application form|request form)\b",
    r"\bmake\b.*\bleave application\b",
    r"\bneed\b.*\b(noc|certificate|bonafide|internship letter|recommendation)\b",
    r"\bi\b.*\bwant\b.*\b(noc|certificate|bonafide|document)\b",
]

# ══════════════════════════════════════════════════════════════════════════════
# Academic info patterns
# ══════════════════════════════════════════════════════════════════════════════
_ACADEMIC_PATTERNS = [
    r"\bfee structure\b",
    r"\b(fee|fees)\b.*\b(structure|detail|breakdown|semester|total)\b",
    r"\bgrading scale\b",
    r"\bgpa scale\b",
    r"\b(grading|grade)\b.*\b(policy|system|scale|criteria)\b",
    r"\bacademic calendar\b",
    r"\bsemester schedule\b",
    r"\badmission requirements\b",
    r"\beligibility criteria\b",
    r"\b(admission|admissions)\b.*\b(requirement|process|document|eligibility|criteria)\b",
    r"\b(requirement|requirements)\b.*\b(admission|apply|applying)\b",
    r"\bdocuments?\b.*\b(required|needed|admission|apply)\b",
    r"\bwhat programs\b",
    r"\ball programs\b",
    r"\blist of programs\b",
    r"\bprograms?\b.*\b(offer|available|kiet|list)\b",
    r"\b(what|list|show)\b.*\b(program|programs|degree|degrees)\b.*\b(offer|available|kiet)\b",
    r"\bhow many\b.*\b(program|degree|course|semester|credit)\b",
    r"\bkiet\b.*\b(facilities|hostel|transport|sports|library)\b",
    r"\b(hostel|transport|library|lab|sports)\b.*\b(detail|info|available|fee|cost)\b",
    r"\b(credit hours?|total credits?)\b",
    r"\b(check|show|get|find|view)\b.*\b(cgpa|gpa|grade|marks|result)\b",
    r"\b(gpa|cgpa)\b.*\b(calculat|compute)\b",
]


def _match(patterns: List[str], text: str) -> bool:
    """Return True if text matches any pattern (case-insensitive)."""
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


class OrchestratorAgent:
    """
    UniAssist+ — Tool-Augmented Multi-Agent RAG System

    Intent Classification (checked before RAG pipeline):
      ACTION  → Email Draft Tool / Document Generator / Academic Info Tool
      HYBRID  → RAG context fetch first, then tool with enriched request
      RAG     → Full 8-step RAG pipeline (FAQ → VECTOR → WEB → LLM → ...)

    RAG Pipeline (8 steps):
      [1] FAQ Agent       — MongoDB fuzzy match
      [2] Smart Routing   — Decides VECTOR-first or WEB-first
      [3] VectorDB Agent  — FAISS semantic search
      [4] WebScraper      — Live KIET website crawl
      [5] LLM Agent       — Generates answer from merged chunks
      [6] Consistency     — Hallucination detection + retry
      [7] Summarizer      — Student-friendly bullet-point output
      [8] Link Appender + FAQ Storage
    """

    LINK_KEYWORDS = {
        "admission", "apply", "application", "form",
        "last date", "closing date", "deadline", "eligibility",
        "entry test", "merit", "apply now",
        "program", "programs", "degree", "bachelor", "bs", "ms",
        "courses", "department", "faculty",
        "code jung", "event", "hackathon", "seminar", "workshop", "fest",
        "fee", "fees", "scholarship", "financial",
        "contact", "address", "location", "campus",
        "website", "portal", "link", "page", "online",
    }

    FALLBACK_LINKS = {
        "admission":    ("🎓 Admissions",            "https://admissions.kiet.edu.pk/"),
        "apply":        ("🎓 Apply Online",           "https://kiet.edu.pk/apply"),
        "eligibility":  ("📋 Admission Process",      "https://admissions.kiet.edu.pk/admission-process/"),
        "requirement":  ("📋 Admission Process",      "https://admissions.kiet.edu.pk/admission-process/"),
        "entry test":   ("📝 Aptitude Test Samples",  "https://admissions.kiet.edu.pk/sample-test-paper/"),
        "aptitude":     ("📝 Aptitude Test Samples",  "https://admissions.kiet.edu.pk/sample-test-paper/"),
        "last date":    ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "closing date": ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "deadline":     ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "schedule":     ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "merit":        ("📅 Admission Schedule",     "https://admissions.kiet.edu.pk/admission-schedule/"),
        "fee":          ("💰 Fee Structure",          "https://kiet.edu.pk/fee-structure/"),
        "scholarship":  ("🏆 Scholarship & Discount", "https://admissions.kiet.edu.pk/scholarship-fee-discount/"),
        "discount":     ("🏆 Scholarship & Discount", "https://admissions.kiet.edu.pk/scholarship-fee-discount/"),
        "program":      ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "courses":      ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "bachelor":     ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "degree":       ("📚 All Programs",           "https://kiet.edu.pk/programs/"),
        "software":     ("💻 COCIS Faculty",          "https://cocis.kiet.edu.pk/"),
        "computer":     ("💻 COCIS Faculty",          "https://cocis.kiet.edu.pk/"),
        "ai":           ("💻 COCIS Faculty",          "https://cocis.kiet.edu.pk/"),
        "management":   ("💼 COMS Faculty",           "https://coms.kiet.edu.pk/"),
        "mba":          ("💼 COMS Faculty",           "https://coms.kiet.edu.pk/"),
        "bba":          ("💼 COMS Faculty",           "https://coms.kiet.edu.pk/"),
        "event":        ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "code jung":    ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "hackathon":    ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "seminar":      ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "workshop":     ("📅 Events & News",          "https://kiet.edu.pk/events-news/"),
        "portal":       ("🖥 Student LMS Portal",     "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php"),
        "lms":          ("🖥 Student LMS Portal",     "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php"),
        "transport":    ("🚌 Transport Services",      "https://kiet.edu.pk/services/students-transport-services/"),
        "sports":       ("⚽ Sports Activities",       "https://kiet.edu.pk/sports-activities/"),
        "alumni":       ("🤝 Alumni",                  "https://kiet.edu.pk/alumni/"),
        "job":          ("💼 Jobs at KIET",            "https://kiet.edu.pk/jobs/"),
        "contact":      ("📞 Departments Contact",    "https://kiet.edu.pk/departments-contact/"),
        "address":      ("🗺 Route Map",               "https://kiet.edu.pk/route-map/"),
        "location":     ("🗺 Route Map",               "https://kiet.edu.pk/route-map/"),
        "calendar":     ("📆 Academic Calendar",       "https://kiet.edu.pk/academics/academic-calendar/"),
        "faq":          ("❓ Admissions FAQs",          "https://kiet.edu.pk/faq/"),
        "about":        ("🏫 About KIET",              "https://kiet.edu.pk/about/"),
    }

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        vector_path: str,
        urls=None,
        base_domain: Optional[str] = None,
        *,
        faq_min_score: int = 82,
        enable_smart_routing: bool = True,
        enable_timing_logs: bool = False,
        consistency_max_retries: int = 1,
        debug: bool = False,
    ):
        # ── RAG Agents ────────────────────────────────────────────────────────
        self.faq_agent = FAQAgent(mongo_uri, db_name, collection_name)
        self.vector_agent = VectorDBAgent(vector_path)
        self.webscraper_agent = WebScraperAgent(
            urls=urls,
            base_domain=base_domain or "kiet.edu.pk",
        )
        self.llm_agent = LLMAgent()
        self.consistency_agent = ConsistencyCheckAgent(
            self.llm_agent,
            max_retries=consistency_max_retries,
            enable_llm_verify=False,
            debug=debug,
        )
        self.summarizing_agent = SummarizingAgent(self.llm_agent, debug=debug)

        # ── Tool Layer ────────────────────────────────────────────────────────
        self.email_tool    = EmailDraftTool(self.llm_agent)
        self.doc_tool      = DocumentGeneratorTool(self.llm_agent)
        self.academic_tool = AcademicInfoTool()
        self.response_log  = ResponseLogAgent(mongo_uri, db_name)

        # ── Config ────────────────────────────────────────────────────────────
        self.faq_min_score        = int(faq_min_score)
        self.enable_smart_routing = bool(enable_smart_routing)
        self.enable_timing_logs   = bool(enable_timing_logs)
        self.debug                = debug

    # ══════════════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ══════════════════════════════════════════════════════════════════════════

    def answer(self, question: str) -> Dict[str, str]:
        question = (question or "").strip()
        if not question:
            return self._fallback()

        t0 = time.perf_counter()

        # ── STEP 1: Intent Classification ─────────────────────────────────────
        intent = self._classify_intent(question)
        logger.info("🧠 Intent: %s | Query: '%s'", intent, question[:60])

        # ── STEP 2: Route to correct pipeline ────────────────────────────────
        if intent == INTENT_ACTION:
            result = self._run_tool_pipeline(question)

        elif intent == INTENT_HYBRID:
            result = self._run_hybrid_pipeline(question)

        else:
            result = self._run_rag_pipeline(question, t0)

        if result is None:
            result = self._fallback()

        # ── STEP 3: Log to MongoDB ─────────────────────────────────────────────
        try:
            self.response_log.log(
                query=question,
                answer=result.get("answer", ""),
                source=result.get("source", "Unknown"),
                intent_type=intent,
                consistency_passed=str(result.get("consistency_passed", "true")).lower() == "true",
                consistency_attempts=int(result.get("consistency_attempts", 0)),
            )
        except Exception as e:
            logger.warning("Response log error: %s", e)

        if self.enable_timing_logs:
            logger.info("⏱ Total: %.3fs | Intent: %s", time.perf_counter() - t0, intent)

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Intent Classifier
    # ══════════════════════════════════════════════════════════════════════════

    def _classify_intent(self, question: str) -> str:
        """
        Priority order: EMAIL > DOCUMENT > ACADEMIC_INFO > HYBRID > RAG
        EMAIL and DOCUMENT are checked before HYBRID to avoid misclassification.
        """
        # Check email first — highest specificity
        if _match(_EMAIL_PATTERNS, question):
            return INTENT_ACTION

        # Check document
        if _match(_DOCUMENT_PATTERNS, question):
            return INTENT_ACTION

        # Check academic info (structured data)
        if _match(_ACADEMIC_PATTERNS, question):
            return INTENT_ACTION

        # HYBRID: has an action word + a context bridge word
        # e.g. "draft an email about scholarship eligibility" — caught above
        # e.g. "write something about the admission process" — hybrid
        action_words = {"draft", "write", "compose", "generate", "create", "make", "prepare"}
        bridge_words = {"about", "regarding", "for", "related", "concerning", "on"}
        tokens = set(question.lower().split())
        if tokens & action_words and tokens & bridge_words:
            return INTENT_HYBRID

        return INTENT_RAG

    # ══════════════════════════════════════════════════════════════════════════
    # Tool Pipeline  (ACTION intent)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_tool_pipeline(self, question: str) -> Dict[str, str]:
        # 1. Email Draft Tool
        if _match(_EMAIL_PATTERNS, question):
            logger.info("🔧 Tool: EmailDraftTool")
            try:
                result = self.email_tool.draft(question)
                return {
                    "source": result.get("source", "Email Draft Tool"),
                    "answer": result.get("answer", ""),
                    "consistency_passed": "true",
                    "consistency_attempts": "0",
                }
            except Exception as e:
                logger.warning("EmailDraftTool error: %s", e)

        # 2. Document Generator Tool
        if _match(_DOCUMENT_PATTERNS, question):
            logger.info("🔧 Tool: DocumentGeneratorTool")
            try:
                result = self.doc_tool.generate(question)
                return {
                    "source": result.get("source", "Document Generator Tool"),
                    "answer": result.get("answer", ""),
                    "consistency_passed": "true",
                    "consistency_attempts": "0",
                }
            except Exception as e:
                logger.warning("DocumentGeneratorTool error: %s", e)

        # 3. Academic Info Tool
        if _match(_ACADEMIC_PATTERNS, question):
            logger.info("🔧 Tool: AcademicInfoTool")
            try:
                result = self.academic_tool.query(question)
                return {
                    "source": result.get("source", "Academic Info Tool"),
                    "answer": result.get("answer", ""),
                    "consistency_passed": "true",
                    "consistency_attempts": "0",
                }
            except Exception as e:
                logger.warning("AcademicInfoTool error: %s", e)

        # Fallback to RAG if tool fails
        logger.warning("Tool pipeline had no match — falling back to RAG")
        return self._run_rag_pipeline(question, time.perf_counter())

    # ══════════════════════════════════════════════════════════════════════════
    # Hybrid Pipeline  (HYBRID intent)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_hybrid_pipeline(self, question: str) -> Dict[str, str]:
        """Fetch RAG context first, then invoke tool with enriched request."""
        logger.info("🔀 Hybrid pipeline started")
        try:
            # Fetch top-2 context chunks
            chunks = self._run_retriever("VECTOR", question) or \
                     self._run_retriever("WEB", question) or []
            context = "\n\n".join(chunks[:2]) if chunks else ""

            enriched = question
            if context:
                enriched = f"{question}\n\nContext:\n{context}"

            # Try email tool with enriched context
            if _match(_EMAIL_PATTERNS, question):
                result = self.email_tool.draft(enriched)
                return {
                    "source": "Hybrid: Email Draft Tool",
                    "answer": result.get("answer", ""),
                    "consistency_passed": "true",
                    "consistency_attempts": "0",
                }

            # Try document tool with enriched context
            if _match(_DOCUMENT_PATTERNS, question):
                result = self.doc_tool.generate(enriched)
                return {
                    "source": "Hybrid: Document Generator Tool",
                    "answer": result.get("answer", ""),
                    "consistency_passed": "true",
                    "consistency_attempts": "0",
                }

        except Exception as e:
            logger.warning("Hybrid pipeline error: %s", e)

        # Fallback to RAG
        return self._run_rag_pipeline(question, time.perf_counter())

    # ══════════════════════════════════════════════════════════════════════════
    # RAG Pipeline  (RAG intent — original 8-step flow preserved exactly)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_rag_pipeline(self, question: str, t0: float) -> Dict[str, str]:
        timings: Dict[str, float] = {}

        # Step 1: FAQ fast path
        faq_result = self._timed("FAQ", timings, lambda: self._run_faq(question))
        if faq_result is not None:
            faq_result["answer"] = self._append_links(
                question, faq_result["answer"], [], "FAQ"
            )
            if self.enable_timing_logs:
                self._log(["FAQ"], timings, time.perf_counter() - t0, "FAQ")
            return faq_result

        # Step 2: Smart routing
        order = self._decide_order(question)
        chunk: Optional[List[str]] = None
        chunk_source: Optional[str] = None

        for name in order:
            if name == "FAQ":
                continue
            result = self._timed(name, timings, lambda n=name: self._run_retriever(n, question))
            if result:
                chunk = result
                chunk_source = name
                logger.info("Retrieval success via %s (%d chunks)", name, len(chunk))
                break

        if not chunk:
            logger.info("No chunks found for: '%s'", question[:60])
            if self.enable_timing_logs:
                self._log(order, timings, time.perf_counter() - t0, "Fallback")
            return self._fallback()

        # Step 3: Extract URLs + strip Source: lines
        source_urls = self._extract_source_urls(chunk)
        clean_chunk = self._strip_source_lines(chunk)

        # Step 4: LLM generation
        t_llm = time.perf_counter()
        raw_answer = self.llm_agent.generate(question=question, context=clean_chunk)
        timings["LLM"] = time.perf_counter() - t_llm

        if not raw_answer:
            logger.warning("LLM returned empty answer.")
            return self._fallback()

        # Step 5: Consistency check
        chunk_text = "\n\n".join(clean_chunk) if isinstance(clean_chunk, list) else clean_chunk
        t_cc = time.perf_counter()
        final_answer, passed, attempts = self.consistency_agent.check_and_fix(
            question=question, answer=raw_answer, chunk=chunk_text,
        )
        timings["ConsistencyCheck"] = time.perf_counter() - t_cc
        logger.info("Consistency check: passed=%s attempts=%d", passed, attempts)

        # Step 6: Summarize
        t_sum = time.perf_counter()
        summarized = self.summarizing_agent.summarize(final_answer)
        timings["Summarize"] = time.perf_counter() - t_sum

        # Step 7: Append links
        summarized = self._append_links(question, summarized, source_urls, chunk_source)

        # Step 8: Store to FAQ
        self._store_to_faq(question, summarized)

        if self.enable_timing_logs:
            self._log(
                order + ["LLM", "ConsistencyCheck", "Summarize"],
                timings, time.perf_counter() - t0, chunk_source or "Unknown"
            )

        return {
            "source": f"{chunk_source} Agent -> LLM",
            "answer": summarized,
            "consistency_passed": str(passed),
            "consistency_attempts": str(attempts),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Retrieval Runners (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_faq(self, question: str) -> Optional[Dict[str, str]]:
        try:
            dbg = self.faq_agent.answer_with_debug(question)
            if dbg and int(dbg.get("score", 0) or 0) >= self.faq_min_score:
                ans = str(dbg.get("answer", "")).strip()
                if ans:
                    return {"source": "FAQ Agent", "answer": ans}
        except Exception as e:
            logger.warning("FAQ agent error: %s", e)
        return None

    def _run_retriever(self, name: str, question: str) -> Optional[List[str]]:
        try:
            if name == "VECTOR":
                result = self.vector_agent.search(question)
            elif name == "WEB":
                result = self.webscraper_agent.scrape(question)
            else:
                return None

            if isinstance(result, list):
                chunks = [c for c in result if isinstance(c, str) and c.strip()]
                return chunks if chunks else None
            if isinstance(result, str) and result.strip():
                return [result.strip()]
            return None

        except Exception as e:
            logger.warning("%s retriever error: %s", name, e)
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # URL Extraction & Link Appending (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_source_urls(self, chunks: List[str]) -> List[str]:
        urls: List[str] = []
        seen: set = set()
        for chunk in chunks:
            for line in chunk.splitlines():
                line = line.strip()
                if line.lower().startswith("source:"):
                    url = line.split(":", 1)[1].strip()
                    if url.startswith("//"):
                        url = "https:" + url
                    if url not in seen and url.startswith("http"):
                        seen.add(url)
                        urls.append(url)
        return urls

    def _strip_source_lines(self, chunks: List[str]) -> List[str]:
        cleaned: List[str] = []
        for chunk in chunks:
            kept = [
                line for line in chunk.splitlines()
                if not line.strip().lower().startswith("source:")
            ]
            cleaned.append("\n".join(kept).strip())
        return cleaned

    def _is_link_worthy(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in self.LINK_KEYWORDS)

    def _get_fallback_links(self, question: str) -> List[tuple]:
        q = question.lower()
        seen_urls: set = set()
        links: List[tuple] = []
        for kw, (label, url) in self.FALLBACK_LINKS.items():
            if kw in q and url not in seen_urls:
                seen_urls.add(url)
                links.append((label, url))
        return links[:2]

    def _append_links(
        self,
        question: str,
        answer: str,
        source_urls: List[str],
        chunk_source: Optional[str],
    ) -> str:
        if not self._is_link_worthy(question):
            return answer
        if "contact the University directly" in answer:
            return answer
        if "not available in the current data" in answer:
            return answer

        link_lines: List[str] = []

        if source_urls and chunk_source == "WEB":
            link_lines.append("\n\U0001f517 *Relevant Links:*")
            for url in source_urls[:2]:
                link_lines.append("   \u2022 " + url)
        else:
            fallbacks = self._get_fallback_links(question)
            if fallbacks:
                link_lines.append("\n\U0001f517 *Useful Links:*")
                for label, url in fallbacks:
                    link_lines.append("   \u2022 " + label + ": " + url)

        if link_lines:
            return answer + "\n" + "\n".join(link_lines)
        return answer

    # ══════════════════════════════════════════════════════════════════════════
    # FAQ Storage (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════

    def _store_to_faq(self, question: str, answer: str) -> None:
        if not question or not answer:
            return
        if answer.strip() == self._fallback()["answer"].strip():
            return
        if "not available in the current data" in answer.lower():
            return
        if len(answer.strip()) < 30:
            return

        try:
            existing = self.faq_agent.collection.find_one(
                {"question": question.strip()}, {"_id": 1}
            )
            if existing:
                return
            self.faq_agent.collection.insert_one(
                {"question": question.strip(), "answer": answer.strip()}
            )
            self.faq_agent._cache = []
            self.faq_agent._cache_loaded_at = 0.0
            self.faq_agent._cache_initialized = False
            logger.info("Stored new FAQ entry: '%s'", question[:60])
        except Exception as e:
            logger.warning("FAQ store error: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # Smart Routing (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════

    def _decide_order(self, question: str) -> List[str]:
        base = ["FAQ", "VECTOR", "WEB"]
        if not self.enable_smart_routing:
            return base

        q = question.lower()

        time_sensitive = any(k in q for k in [
            "latest", "recent", "updated", "notice", "announcement",
            "deadline", "timetable", "schedule", "merit list",
            "result", "today", "this week", "new",
            "last date", "closing date", "due date", "last day",
            "when is", "when are", "when does", "when will",
            "open now", "currently", "this year", "2025", "2026",
            "admission date", "apply by", "submission date",
        ])
        if time_sensitive:
            logger.info("Smart routing: time-sensitive -> WEB first")
            return ["FAQ", "WEB", "VECTOR"]

        web_first = any(k in q for k in [
            "event", "competition", "fest", "code jung", "hackathon",
            "seminar", "workshop", "expo", "tech fest", "gaming",
            "how many", "list of", "all programs", "all courses",
            "programs offering", "programs offered", "kiet offering",
            "bachelor", "bachelors", "bs programs", "ms programs",
            "degree", "courses offered", "what programs",
            "admission", "apply", "eligibility", "requirement",
            "merit", "entry test", "form", "apply online",
        ])
        if web_first:
            logger.info("Smart routing: web-first query -> WEB before VECTOR")
            return ["FAQ", "WEB", "VECTOR"]

        return base

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════

    def _timed(self, name: str, timings: dict, fn):
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    def _fallback(self) -> Dict[str, str]:
        return {
            "source": "None",
            "answer": (
                "I'm sorry, I couldn't find information on that.\n"
                "Please contact the University directly:\n"
                "\u2022 \U0001f4de Phone: 02136628381 / 02136679314\n"
                "\u2022 \U0001f4e7 Email: admissions@kiet.edu.pk"
            ),
        }

    def _log(self, order, timings, total, chosen):
        parts = " | ".join([f"{k}={timings.get(k, 0):.3f}s" for k in order])
        logger.info("\u23f1 chosen=%s total=%.3fs | %s", chosen, total, parts)

    def close(self) -> None:
        try:
            self.faq_agent.close()
        except Exception:
            pass
        try:
            self.response_log.close()
        except Exception:
            pass
