from __future__ import annotations

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger("eduassist.tools.academic_info")

# =========================
# Simulated Academic Data
# (Acts as a mock university API)
# =========================

ACADEMIC_DATA = {
    "programs": {
        "BS Software Engineering": {
            "duration": "4 Years (8 Semesters)",
            "degree": "Bachelor of Science",
            "department": "COCIS",
            "eligibility": "FSc Pre-Engineering / ICS / A-Levels with Mathematics",
            "total_credit_hours": 136,
            "accreditation": "PEC Accredited",
        },
        "BS Computer Science": {
            "duration": "4 Years (8 Semesters)",
            "degree": "Bachelor of Science",
            "department": "COCIS",
            "eligibility": "FSc Pre-Engineering / ICS / A-Levels with Mathematics",
            "total_credit_hours": 134,
            "accreditation": "PEC Accredited",
        },
        "BS Artificial Intelligence": {
            "duration": "4 Years (8 Semesters)",
            "degree": "Bachelor of Science",
            "department": "COCIS",
            "eligibility": "FSc Pre-Engineering / ICS / A-Levels with Mathematics",
            "total_credit_hours": 133,
            "accreditation": "HEC Recognized",
        },
        "BBA": {
            "duration": "4 Years (8 Semesters)",
            "degree": "Bachelor of Business Administration",
            "department": "COMS",
            "eligibility": "Intermediate / A-Levels (any group)",
            "total_credit_hours": 130,
            "accreditation": "HEC Recognized",
        },
        "MBA": {
            "duration": "2 Years (4 Semesters)",
            "degree": "Master of Business Administration",
            "department": "COMS",
            "eligibility": "16 Years of Education (Bachelor's Degree)",
            "total_credit_hours": 60,
            "accreditation": "HEC Recognized",
        },
        "BS Electronics Engineering": {
            "duration": "4 Years (8 Semesters)",
            "degree": "Bachelor of Science",
            "department": "COEL",
            "eligibility": "FSc Pre-Engineering / A-Levels with Physics & Mathematics",
            "total_credit_hours": 138,
            "accreditation": "PEC Accredited",
        },
    },

    "fee_structure": {
        "BS Software Engineering": {"per_semester": "PKR 85,000 – 95,000", "registration": "PKR 15,000"},
        "BS Computer Science":     {"per_semester": "PKR 85,000 – 95,000", "registration": "PKR 15,000"},
        "BS Artificial Intelligence": {"per_semester": "PKR 85,000 – 95,000", "registration": "PKR 15,000"},
        "BBA":                     {"per_semester": "PKR 75,000 – 85,000", "registration": "PKR 12,000"},
        "MBA":                     {"per_semester": "PKR 90,000 – 100,000", "registration": "PKR 15,000"},
        "BS Electronics Engineering": {"per_semester": "PKR 90,000 – 100,000", "registration": "PKR 15,000"},
        "general": {"per_semester": "PKR 75,000 – 100,000 (varies by program)", "registration": "PKR 12,000 – 15,000"},
    },

    "admission": {
        "undergraduate": {
            "open_merit_seats": "Available",
            "self_finance_seats": "Available",
            "entry_test": "KIET Aptitude Test (Mathematics, English, IQ)",
            "required_documents": [
                "Matric Certificate & Transcripts",
                "Intermediate Certificate & Transcripts",
                "CNIC / B-Form Copy",
                "Domicile Certificate",
                "4 Passport Size Photographs",
                "Character Certificate from Previous Institution",
            ],
            "application_process": "Online via admissions.kiet.edu.pk or visit campus",
            "contact": "admissions@kiet.edu.pk | 021-36628381",
        },
        "postgraduate": {
            "entry_test": "HAT / GAT (HEC) or KIET internal test",
            "required_documents": [
                "Bachelor's Degree & Transcripts",
                "CNIC Copy",
                "2 Passport Size Photographs",
                "Experience Letter (if applicable)",
            ],
            "contact": "admissions@kiet.edu.pk | 021-36628381",
        },
    },

    "academic_calendar": {
        "fall_semester": {
            "start": "September",
            "end": "January",
            "midterms": "November",
            "finals": "January",
        },
        "spring_semester": {
            "start": "February",
            "end": "June",
            "midterms": "April",
            "finals": "June",
        },
        "summer_semester": {
            "available": "Yes (for backlog/repeat students)",
            "duration": "6–8 weeks (July – August)",
        },
    },

    "grading_policy": {
        "A+": {"range": "90–100", "gpa_points": 4.0},
        "A":  {"range": "85–89",  "gpa_points": 4.0},
        "A-": {"range": "80–84",  "gpa_points": 3.7},
        "B+": {"range": "75–79",  "gpa_points": 3.3},
        "B":  {"range": "70–74",  "gpa_points": 3.0},
        "B-": {"range": "65–69",  "gpa_points": 2.7},
        "C+": {"range": "60–64",  "gpa_points": 2.3},
        "C":  {"range": "55–59",  "gpa_points": 2.0},
        "D":  {"range": "50–54",  "gpa_points": 1.0},
        "F":  {"range": "Below 50", "gpa_points": 0.0},
        "passing_gpa": "2.0 CGPA required to pass a semester",
        "probation": "Below 2.0 CGPA results in academic probation",
    },

    "facilities": {
        "library":   "Digital & physical library with 50,000+ books, IEEE/ACM digital access",
        "labs":      "25+ computer labs, electronics labs, multimedia labs",
        "hostel":    "Separate hostels for male and female students",
        "transport": "University bus service available across Karachi",
        "sports":    "Cricket, football, basketball, table tennis, chess facilities",
        "cafeteria": "On-campus cafeteria and food courts",
        "mosque":    "On-campus mosque available",
        "clinic":    "Medical clinic with qualified staff",
    },
}

# =========================
# Intent Patterns
# =========================

ACADEMIC_INFO_PATTERNS = [
    r"\b(check|show|get|fetch|lookup|find|view)\b.*\b(cgpa|gpa|grade|marks|result|semester result)\b",
    r"\b(gpa|cgpa)\b.*\b(calculator|calculate|compute)\b",
    r"\b(fee|fees)\b.*\b(structure|detail|breakdown|semester|total)\b",
    r"\bfee structure\b",
    r"\b(grading|grade)\b.*\b(policy|system|scale|criteria)\b",
    r"\bgrading scale\b",
    r"\bgpa scale\b",
    r"\bacademic calendar\b",
    r"\bsemester schedule\b",
    r"\b(admission|admissions)\b.*\b(requirement|process|document|eligibility|criteria)\b",
    r"\b(requirement|requirements)\b.*\b(admission|apply|applying)\b",
    r"\bdocuments?\b.*\b(required|needed|admission|apply)\b",
    r"\b(what|list|show)\b.*\b(program|programs|degree|degrees|course|courses)\b.*\b(offer|available|kiet)\b",
    r"\bhow many\b.*\b(program|degree|course|semester|credit)\b",
    r"\bprograms?\b.*\b(offer|available|kiet|list)\b",
    r"\ball programs\b",
    r"\b(hostel|transport|library|lab|sports|facilities)\b.*\b(detail|info|available|fee|cost)\b",
    r"\bkiet\b.*\b(facilities|hostel|transport|sports|library)\b",
    r"\b(credit hours?|total credits?)\b",
    r"\bwhat programs\b",
    r"\blist of programs\b",
    r"\badmission requirements\b",
    r"\beligibility criteria\b",
]


class AcademicInfoTool:
    """
    Simulated Academic Information API for KIET University.

    Acts as a mock backend service providing structured data about:
    - Programs & eligibility
    - Fee structures
    - Admission requirements
    - Academic calendar
    - Grading policy
    - Campus facilities

    ✅ No LLM needed — returns structured data directly
    ✅ Simulates a real university API response
    ✅ Falls back to formatted summary for unmatched queries
    """

    # =========================
    # Intent Detection
    # =========================

    @staticmethod
    def is_academic_info_request(question: str) -> bool:
        q = question.lower().strip()
        for pattern in ACADEMIC_INFO_PATTERNS:
            if re.search(pattern, q):
                return True
        return False

    # =========================
    # Main Method
    # =========================

    def query(self, question: str) -> Dict[str, str]:
        """
        Query the simulated academic API.

        Returns:
            dict with 'answer', 'source', 'data_type'
        """
        question = (question or "").strip()
        if not question:
            return self._error_response("Empty query.")

        q = question.lower()

        # Route to appropriate data handler
        if any(k in q for k in ["fee", "fees", "cost", "tuition"]):
            return self._get_fees(q)

        if any(k in q for k in ["grading", "grade", "gpa", "cgpa", "marks", "result"]):
            return self._get_grading(q)

        if any(k in q for k in ["calendar", "semester schedule", "semester start", "semester end", "midterm", "final exam"]):
            return self._get_calendar()

        if any(k in q for k in ["admission", "requirement", "eligibility", "document", "apply", "entry test"]):
            return self._get_admission(q)

        if any(k in q for k in ["program", "programs", "degree", "courses", "offering", "offered", "bachelor", "master"]):
            return self._get_programs(q)

        if any(k in q for k in ["hostel", "transport", "library", "lab", "sports", "facility", "facilities", "cafeteria"]):
            return self._get_facilities(q)

        return self._get_summary()

    # =========================
    # Data Handlers
    # =========================

    def _get_fees(self, q: str) -> Dict[str, str]:
        # Try to match specific program
        for prog, data in ACADEMIC_DATA["fee_structure"].items():
            if prog == "general":
                continue
            if any(word in q for word in prog.lower().split()):
                answer = (
                    f"💰 **Fee Structure — {prog}**\n\n"
                    f"• Per Semester Fee: {data['per_semester']}\n"
                    f"• Registration Fee: {data['registration']}\n\n"
                    f"*Note: Fees may vary slightly. Check [Fee Structure](https://kiet.edu.pk/fee-structure/) for latest details.*"
                )
                return {"source": "Academic Info Tool", "data_type": "fee_structure", "answer": answer}

        # General fee info
        g = ACADEMIC_DATA["fee_structure"]["general"]
        answer = (
            f"💰 **KIET Fee Structure (General)**\n\n"
            f"• Per Semester Fee: {g['per_semester']}\n"
            f"• Registration Fee: {g['registration']}\n\n"
            f"*Fees vary by program. For exact details visit: [kiet.edu.pk/fee-structure](https://kiet.edu.pk/fee-structure/)*"
        )
        return {"source": "Academic Info Tool", "data_type": "fee_structure", "answer": answer}

    def _get_grading(self, q: str) -> Dict[str, str]:
        gp = ACADEMIC_DATA["grading_policy"]
        grades = "\n".join(
            f"• {g}: {v['range']}% → {v['gpa_points']} GPA points"
            for g, v in gp.items()
            if isinstance(v, dict)
        )
        answer = (
            f"📊 **KIET Grading Policy**\n\n"
            f"{grades}\n\n"
            f"• Passing CGPA: {gp['passing_gpa']}\n"
            f"• Probation: {gp['probation']}"
        )
        return {"source": "Academic Info Tool", "data_type": "grading_policy", "answer": answer}

    def _get_calendar(self) -> Dict[str, str]:
        cal = ACADEMIC_DATA["academic_calendar"]
        answer = (
            f"📆 **KIET Academic Calendar**\n\n"
            f"**Fall Semester:**\n"
            f"• Start: {cal['fall_semester']['start']} | End: {cal['fall_semester']['end']}\n"
            f"• Midterms: {cal['fall_semester']['midterms']} | Finals: {cal['fall_semester']['finals']}\n\n"
            f"**Spring Semester:**\n"
            f"• Start: {cal['spring_semester']['start']} | End: {cal['spring_semester']['end']}\n"
            f"• Midterms: {cal['spring_semester']['midterms']} | Finals: {cal['spring_semester']['finals']}\n\n"
            f"**Summer Semester:**\n"
            f"• {cal['summer_semester']['available']} — {cal['summer_semester']['duration']}\n\n"
            f"*For exact dates check: [Academic Calendar](https://kiet.edu.pk/academics/academic-calendar/)*"
        )
        return {"source": "Academic Info Tool", "data_type": "academic_calendar", "answer": answer}

    def _get_admission(self, q: str) -> Dict[str, str]:
        level = "postgraduate" if any(k in q for k in ["mba", "ms", "master", "postgrad", "graduate"]) else "undergraduate"
        adm = ACADEMIC_DATA["admission"][level]
        docs = "\n".join(f"  • {d}" for d in adm.get("required_documents", []))
        answer = (
            f"🎓 **Admission Requirements ({level.title()})**\n\n"
            f"• Entry Test: {adm.get('entry_test', 'N/A')}\n"
            f"• Application: {adm.get('application_process', 'Visit campus or apply online')}\n\n"
            f"**Required Documents:**\n{docs}\n\n"
            f"• Contact: {adm.get('contact', 'admissions@kiet.edu.pk')}"
        )
        return {"source": "Academic Info Tool", "data_type": "admission_info", "answer": answer}

    def _get_programs(self, q: str) -> Dict[str, str]:
        progs = ACADEMIC_DATA["programs"]

        # Specific program lookup
        for prog, data in progs.items():
            if any(word in q for word in prog.lower().split() if len(word) > 2):
                answer = (
                    f"📚 **{prog}**\n\n"
                    f"• Duration: {data['duration']}\n"
                    f"• Degree: {data['degree']}\n"
                    f"• Department: {data['department']}\n"
                    f"• Eligibility: {data['eligibility']}\n"
                    f"• Credit Hours: {data['total_credit_hours']}\n"
                    f"• Accreditation: {data['accreditation']}"
                )
                return {"source": "Academic Info Tool", "data_type": "program_info", "answer": answer}

        # All programs list
        prog_list = "\n".join(f"  • {p} ({d['department']})" for p, d in progs.items())
        answer = (
            f"📚 **Programs Offered at KIET University**\n\n"
            f"{prog_list}\n\n"
            f"*For full details visit: [kiet.edu.pk/programs](https://kiet.edu.pk/programs/)*"
        )
        return {"source": "Academic Info Tool", "data_type": "programs_list", "answer": answer}

    def _get_facilities(self, q: str) -> Dict[str, str]:
        fac = ACADEMIC_DATA["facilities"]
        specific_keys = ["library", "labs", "hostel", "transport", "sports", "cafeteria", "mosque", "clinic"]
        for key in specific_keys:
            if key in q or key.rstrip("s") in q:
                answer = f"🏫 **{key.title()} — KIET University**\n\n• {fac.get(key, 'Information not available.')}"
                return {"source": "Academic Info Tool", "data_type": "facilities", "answer": answer}

        # All facilities
        fac_list = "\n".join(f"  • **{k.title()}:** {v}" for k, v in fac.items())
        answer = f"🏫 **KIET University Facilities**\n\n{fac_list}"
        return {"source": "Academic Info Tool", "data_type": "facilities", "answer": answer}

    def _get_summary(self) -> Dict[str, str]:
        answer = (
            "ℹ️ **KIET Academic Info Tool**\n\n"
            "I can provide information on:\n"
            "• 📚 Programs & Eligibility\n"
            "• 💰 Fee Structure\n"
            "• 🎓 Admission Requirements & Documents\n"
            "• 📆 Academic Calendar\n"
            "• 📊 Grading Policy & GPA Scale\n"
            "• 🏫 Campus Facilities\n\n"
            "Please ask a specific question about any of the above!"
        )
        return {"source": "Academic Info Tool", "data_type": "summary", "answer": answer}

    def _error_response(self, msg: str) -> Dict[str, str]:
        return {"source": "Academic Info Tool", "data_type": "error", "answer": f"❌ {msg}"}
