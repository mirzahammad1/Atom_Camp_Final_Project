from __future__ import annotations

import re
import logging
from typing import Dict

logger = logging.getLogger("eduassist.tools.document_generator")

# =========================
# Document Prompt Template
# =========================

DOCUMENT_PROMPT = """You are a professional document writer for university students.

Generate a formal document based on the student's request below.

Rules:
- Use formal, structured language appropriate for a university document.
- Include a title, relevant sections, and proper formatting using plain text.
- Keep it concise and professional.
- Do NOT add fictional data (leave placeholders like [Name], [Date], [ID] where needed).
- Start the document with its title on the first line in CAPS.

Student Request:
{request}

Document:"""


# =========================
# Intent Patterns
# =========================

DOCUMENT_PATTERNS = [
    r"\bgenerate\b.*\b(document|doc|letter|form|certificate|report|noc|slip|affidavit)\b",
    r"\bcreate\b.*\b(document|doc|letter|form|certificate|report|noc|slip)\b",
    r"\bwrite\b.*\b(noc|no.?objection|bonafide|character certificate|experience letter)\b",
    r"\b(noc|no.?objection certificate)\b",
    r"\bbonafide\b",
    r"\bcharacter certificate\b",
    r"\bexperience letter\b",
    r"\binternship letter\b",
    r"\brecommendation letter\b",
    r"\b(generate|create|make|prepare|draft)\b.*\b(application form|request form)\b",
    r"\bfee\b.*\b(receipt|slip|document)\b",
    r"\bsemester\b.*\b(report|document|record)\b",
]

# =========================
# Pre-built Templates
# =========================

TEMPLATES: Dict[str, Dict[str, str]] = {
    "noc": {
        "title": "NO OBJECTION CERTIFICATE (NOC)",
        "body": (
            "TO WHOM IT MAY CONCERN\n\n"
            "This is to certify that [Student Name], bearing Student ID [ID], "
            "is currently enrolled in [Program Name] at KIET University, Karachi.\n\n"
            "The university has no objection to the student [purpose, e.g., 'applying for an internship / "
            "appearing in an external examination / traveling abroad'] as mentioned in the request.\n\n"
            "This NOC is issued on the request of the student for the purpose stated above.\n\n"
            "Issued on: [Date]\n\n"
            "________________________\n"
            "Authorized Signatory\n"
            "Registrar / Department Head\n"
            "KIET University, Karachi"
        ),
    },
    "bonafide": {
        "title": "BONAFIDE CERTIFICATE",
        "body": (
            "TO WHOM IT MAY CONCERN\n\n"
            "This is to certify that [Student Name], son/daughter of [Father's Name], "
            "bearing CNIC No. [CNIC], is a bonafide student of KIET University, Karachi.\n\n"
            "Details:\n"
            "  • Program:   [Program Name]\n"
            "  • Semester:  [Current Semester]\n"
            "  • Session:   [Academic Year]\n"
            "  • Student ID: [ID]\n\n"
            "This certificate is issued on the student's request for [purpose].\n\n"
            "Issued on: [Date]\n\n"
            "________________________\n"
            "Registrar\n"
            "KIET University, Karachi"
        ),
    },
    "internship_letter": {
        "title": "INTERNSHIP RECOMMENDATION LETTER",
        "body": (
            "Date: [Date]\n\n"
            "To,\n"
            "The HR Manager,\n"
            "[Company Name]\n\n"
            "Subject: Internship Recommendation for [Student Name]\n\n"
            "Dear Sir/Madam,\n\n"
            "It is our pleasure to recommend [Student Name], Student ID [ID], "
            "currently enrolled in [Program] at KIET University, Karachi, for an internship opportunity at your organization.\n\n"
            "The student has demonstrated strong academic performance and a keen interest in [field]. "
            "We are confident that they will be a valuable addition to your team.\n\n"
            "We request your kind consideration for this opportunity.\n\n"
            "Yours sincerely,\n\n"
            "________________________\n"
            "[Faculty Name]\n"
            "Department of [Department]\n"
            "KIET University, Karachi"
        ),
    },
    "leave_application": {
        "title": "LEAVE APPLICATION",
        "body": (
            "Date: [Date]\n\n"
            "To,\n"
            "The Class Instructor / Department Head,\n"
            "KIET University, Karachi\n\n"
            "Subject: Application for Leave\n\n"
            "Respected Sir/Madam,\n\n"
            "I, [Your Name], Student ID [ID], enrolled in [Program], Semester [Semester], "
            "hereby request leave for [number] days from [start date] to [end date].\n\n"
            "Reason: [State your reason here]\n\n"
            "I assure you that I will complete all missed assignments and coursework upon my return.\n\n"
            "Thanking you in anticipation.\n\n"
            "Yours sincerely,\n"
            "[Your Name]\n"
            "Student ID: [ID]\n"
            "Contact: [Phone Number]"
        ),
    },
}


class DocumentGeneratorTool:
    """
    Generates formal university documents for students.

    Handles:
    - NOC (No Objection Certificate)
    - Bonafide Certificate
    - Internship Letters
    - Leave Applications
    - Custom documents via LLM

    ✅ Pre-built templates for common documents (fast, no LLM needed)
    ✅ Falls back to LLM for custom document types
    """

    def __init__(self, llm_agent):
        self.llm = llm_agent

    # =========================
    # Intent Detection
    # =========================

    @staticmethod
    def is_document_request(question: str) -> bool:
        q = question.lower().strip()
        for pattern in DOCUMENT_PATTERNS:
            if re.search(pattern, q):
                return True
        return False

    # =========================
    # Main Method
    # =========================

    def generate(self, request: str) -> Dict[str, str]:
        """
        Generate a formal document based on student's request.

        Returns:
            dict with 'title', 'body', 'answer', 'source'
        """
        request = (request or "").strip()
        if not request:
            return self._error_response("Empty request provided.")

        # Check for pre-built template match first (faster)
        template_key = self._match_template(request)
        if template_key:
            return self._from_template(template_key)

        # Fall back to LLM generation for custom docs
        return self._llm_generate(request)

    # =========================
    # Template Matching
    # =========================

    def _match_template(self, request: str) -> str | None:
        q = request.lower()
        if "noc" in q or "no objection" in q:
            return "noc"
        if "bonafide" in q:
            return "bonafide"
        if "internship" in q and ("letter" in q or "recommend" in q):
            return "internship_letter"
        if "leave" in q and ("application" in q or "request" in q):
            return "leave_application"
        return None

    def _from_template(self, key: str) -> Dict[str, str]:
        t = TEMPLATES[key]
        title = t["title"]
        body = t["body"]
        return {
            "source": "Document Generator Tool (Template)",
            "title": title,
            "body": body,
            "answer": self._format_output(title, body),
        }

    # =========================
    # LLM Generation
    # =========================

    def _llm_generate(self, request: str) -> Dict[str, str]:
        prompt = DOCUMENT_PROMPT.format(request=request)
        try:
            raw = self.llm._generate(prompt)
            if not raw or len(raw.strip()) < 30:
                return self._generic_template(request)

            lines = raw.strip().splitlines()
            title = lines[0].strip() if lines else "FORMAL DOCUMENT"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw.strip()

            logger.info("Document generated via LLM for: '%s'", request[:50])
            return {
                "source": "Document Generator Tool (LLM)",
                "title": title,
                "body": body,
                "answer": self._format_output(title, body),
            }
        except Exception as e:
            logger.error("Document generation error: %s", e)
            return self._generic_template(request)

    # =========================
    # Output Formatter
    # =========================

    def _format_output(self, title: str, body: str) -> str:
        return (
            f"📄 **{title}**\n\n"
            f"---\n\n"
            f"{body}\n\n"
            f"---\n"
            f"*✏️ Replace all [ ] placeholders with your actual information before use.*"
        )

    def _generic_template(self, request: str) -> Dict[str, str]:
        title = "FORMAL REQUEST DOCUMENT"
        body = (
            "Date: [Date]\n\n"
            "To,\n"
            "The Concerned Authority,\n"
            "KIET University, Karachi\n\n"
            f"Subject: {request[:80]}\n\n"
            "Respected Sir/Madam,\n\n"
            f"I, [Your Name], Student ID [ID], enrolled in [Program], respectfully request your assistance "
            f"regarding: {request}\n\n"
            "I would be grateful if this matter could be addressed at the earliest.\n\n"
            "Thanking you.\n\n"
            "Yours sincerely,\n"
            "[Your Name]\n"
            "[Student ID]\n"
            "[Program & Semester]\n"
            "[Date]"
        )
        return {
            "source": "Document Generator Tool (Template)",
            "title": title,
            "body": body,
            "answer": self._format_output(title, body),
        }

    def _error_response(self, msg: str) -> Dict[str, str]:
        return {
            "source": "Document Generator Tool",
            "title": "",
            "body": "",
            "answer": f"❌ {msg}",
        }