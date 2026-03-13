from __future__ import annotations

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger("eduassist.tools.email_draft")

# =========================
# Email Prompt Template
# =========================

EMAIL_PROMPT = """You are a professional email writer for university students.

Write a formal email based on the student's request below.

Rules:
- Use formal, polite, professional language.
- Include: Subject line, greeting, body, closing, and signature placeholder.
- Keep it concise (max 150 words body).
- Do NOT add any information not implied by the request.
- Format: Start with "Subject:" on the first line.

Student Request:
{request}

Formal Email:
Subject:"""


# =========================
# Intent Patterns
# =========================

EMAIL_PATTERNS = [
    # draft/write/compose email
    r"\bdraft\b.*\bemail\b",
    r"\bwrite\b.*\bemail\b",
    r"\bcompose\b.*\bemail\b",
    r"\bmake\b.*\bemail\b",
    r"\bgenerate\b.*\bemail\b",
    r"\bhelp\b.*\bemail\b",
    r"\bemail\b.*\bdraft\b",
    r"\bemail\b.*\bto\b.*\b(professor|department|admin|registrar|dean|faculty|sir|ma'am|office)\b",
    r"\bneed\b.*\bemail\b",
    r"\bsend\b.*\bemail\b",
    r"\bi\b.*\bwant\b.*\bemail\b",
    # letters and requests
    r"\bwrite\b.*\b(letter|request|complaint|application)\b",
    r"\bdraft\b.*\b(letter|request|complaint|application)\b",
    r"\bmake\b.*\b(letter|request|complaint|application)\b",
    r"\bgenerate\b.*\b(letter|request|complaint|application)\b",
    # leave and applications
    r"\bapplication\b.*\b(leave|absence|extension|fee|scholarship|internship)\b",
    r"\bleave\b.*\b(application|request|letter)\b",
    r"\bwrite\b.*\bto\b.*\b(professor|sir|madam|teacher|department|admin|dean)\b",
    r"\bdraft\b.*\bto\b.*\b(professor|sir|madam|teacher|department|admin|dean)\b",
]


class EmailDraftTool:
    """
    Drafts professional emails for students based on their request.

    Handles:
    - Email to professors (leave, extension, queries)
    - Email to admin (fee, scholarship, documents)
    - Complaint letters
    - Formal applications

    ✅ Uses LLM for generation
    ✅ Falls back to template if LLM fails
    """

    def __init__(self, llm_agent):
        self.llm = llm_agent

    # =========================
    # Intent Detection
    # =========================

    @staticmethod
    def is_email_request(question: str) -> bool:
        q = question.lower().strip()
        for pattern in EMAIL_PATTERNS:
            if re.search(pattern, q):
                return True
        return False

    # =========================
    # Main Method
    # =========================

    def draft(self, request: str) -> Dict[str, str]:
        """
        Generate an email draft from student's request.

        Returns:
            dict with 'subject', 'body', 'full_email', 'source'
        """
        request = (request or "").strip()
        if not request:
            return self._error_response("Empty request provided.")

        prompt = EMAIL_PROMPT.format(request=request)

        try:
            raw = self.llm._generate(prompt)
            if not raw:
                return self._fallback_template(request)

            # Prepend "Subject:" since it was part of prompt ending
            full_email = "Subject:" + raw if not raw.strip().lower().startswith("subject:") else raw
            subject, body = self._parse_email(full_email)

            logger.info("Email drafted successfully for request: '%s'", request[:50])

            return {
                "source": "Email Draft Tool",
                "subject": subject,
                "body": body,
                "full_email": full_email.strip(),
                "answer": self._format_output(subject, body),
            }

        except Exception as e:
            logger.error("Email draft error: %s", e)
            return self._fallback_template(request)

    # =========================
    # Parsing
    # =========================

    def _parse_email(self, raw: str) -> tuple[str, str]:
        lines = raw.strip().splitlines()
        subject = ""
        body_lines = []
        in_body = False

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("subject:"):
                subject = line_stripped.split(":", 1)[1].strip()
                in_body = True
            elif in_body:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if not subject:
            subject = "Student Request"
        if not body:
            body = raw.strip()

        return subject, body

    # =========================
    # Output Formatter
    # =========================

    def _format_output(self, subject: str, body: str) -> str:
        return (
            f"📧 **Email Draft**\n\n"
            f"**Subject:** {subject}\n\n"
            f"---\n\n"
            f"{body}\n\n"
            f"---\n"
            f"*✏️ Feel free to edit before sending.*"
        )

    # =========================
    # Fallback Template
    # =========================

    def _fallback_template(self, request: str) -> Dict[str, str]:
        subject = "Student Request – Action Required"
        body = (
            "Dear Sir/Madam,\n\n"
            "I hope this email finds you well. I am writing to request your assistance "
            f"regarding the following matter:\n\n{request}\n\n"
            "I would be grateful if you could look into this matter at your earliest convenience.\n\n"
            "Thank you for your time and consideration.\n\n"
            "Yours sincerely,\n"
            "[Your Name]\n"
            "[Student ID]\n"
            "[Program & Semester]\n"
            "[Contact Number]"
        )
        full_email = f"Subject: {subject}\n\n{body}"
        return {
            "source": "Email Draft Tool (Template)",
            "subject": subject,
            "body": body,
            "full_email": full_email,
            "answer": self._format_output(subject, body),
        }

    def _error_response(self, msg: str) -> Dict[str, str]:
        return {
            "source": "Email Draft Tool",
            "subject": "",
            "body": "",
            "full_email": "",
            "answer": f"❌ {msg}",
        }
