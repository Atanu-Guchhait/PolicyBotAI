from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List

from utils.logger import setup_logger

logger = setup_logger(__name__)
# **************************** DEFINED LITERALS ****************************

DEPT_LITERALS = Literal[
    'Operations', 'Engineering', 'Sales', 'Customer Support', 
    'Marketing', 'HR', 'Finance', 'Legal', 'Admin', 'IT'
]

CATEGORY_LITERALS = Literal[
    'Leave Policy', 'Work From Home', 'Payroll & Compensation', 
    'Attendance & Timing', 'Code of Conduct', 'Performance Management', 
    'Recruitment & Onboarding', 'Training & Development', 
    'Travel & Expense', 'Separation & Exit'
]

POLICY_NAME_LITERALS = Literal[
    'Bereavement Leave — Leave Policy', 'Casual Leave — Leave Policy', 
    'Earned Leave — Leave Policy', 'Compensatory Off — Leave Policy', 
    'Maternity Leave — Leave Policy', 'Sick Leave — Leave Policy', 
    'Paternity Leave — Leave Policy', 'WFH Eligibility — Work From Home', 
    'Hybrid Work Schedule — Work From Home', 'WFH Equipment Policy — Work From Home', 
    'WFH Request Process — Work From Home', 'Remote Work Guidelines — Work From Home', 
    'Reimbursements — Payroll & Compensation', 'Variable Pay — Payroll & Compensation', 
    'Increment Policy — Payroll & Compensation', 'Salary Structure — Payroll & Compensation', 
    'Bonus Policy — Payroll & Compensation', 'Overtime Pay — Payroll & Compensation', 
    'Flexi-Time Policy — Attendance & Timing', 'Shift Policy — Attendance & Timing', 
    'Late Arrival Policy — Attendance & Timing', 'Office Hours — Attendance & Timing', 
    'Attendance Tracking — Attendance & Timing', 'Social Media Policy — Code of Conduct', 
    'Workplace Behaviour — Code of Conduct', 'Anti-Harassment — Code of Conduct', 
    'Conflict of Interest — Code of Conduct', 'Dress Code — Code of Conduct', 
    'Confidentiality — Code of Conduct', '360 Degree Feedback — Performance Management', 
    'Probation Review — Performance Management', 'Appraisal Process — Performance Management', 
    'Performance Improvement Plan — Performance Management', 'KPI Setting — Performance Management', 
    'Probation Period — Recruitment & Onboarding', 'Background Verification — Recruitment & Onboarding', 
    'Offer Letter Policy — Recruitment & Onboarding', 'Onboarding Checklist — Recruitment & Onboarding', 
    'Hiring Process — Recruitment & Onboarding', 'Referral Policy — Recruitment & Onboarding', 
    'Skill Development — Training & Development', 'E-Learning Policy — Training & Development', 
    'External Certification — Training & Development', 'Mandatory Training — Training & Development', 
    'Leadership Programme — Training & Development', 'Expense Claim Process — Travel & Expense', 
    'Hotel Booking — Travel & Expense', 'International Travel — Travel & Expense', 
    'Flight Booking — Travel & Expense', 'Business Travel Policy — Travel & Expense', 
    'Per Diem Allowance — Travel & Expense', 'Resignation Process — Separation & Exit', 
    'Notice Period Policy — Separation & Exit', 'Exit Interview — Separation & Exit', 
    'Rehire Policy — Separation & Exit', 'Full & Final Settlement — Separation & Exit', 
    'Non-Disclosure Agreement — Separation & Exit'
]

SUBCATEGORY_LITERALS = Literal[
    'Bereavement Leave', 'Casual Leave', 'Earned Leave', 'Compensatory Off', 
    'Maternity Leave', 'Sick Leave', 'Paternity Leave', 'WFH Eligibility', 
    'Hybrid Work Schedule', 'WFH Equipment Policy', 'WFH Request Process', 
    'Remote Work Guidelines', 'Reimbursements', 'Variable Pay', 
    'Increment Policy', 'Salary Structure', 'Bonus Policy', 
    'Overtime Pay', 'Flexi-Time Policy', 'Shift Policy', 
    'Late Arrival Policy', 'Office Hours', 'Attendance Tracking', 
    'Social Media Policy', 'Workplace Behaviour', 'Anti-Harassment', 
    'Conflict of Interest', 'Dress Code', 'Confidentiality', 
    '360 Degree Feedback', 'Probation Review', 'Appraisal Process', 
    'Performance Improvement Plan', 'KPI Setting', 'Probation Period', 
    'Background Verification', 'Offer Letter Policy', 'Onboarding Checklist', 
    'Hiring Process', 'Referral Policy', 'Skill Development', 
    'E-Learning Policy', 'External Certification', 'Mandatory Training', 
    'Leadership Programme', 'Expense Claim Process', 'Hotel Booking', 
    'International Travel', 'Flight Booking', 'Business Travel Policy', 
    'Per Diem Allowance', 'Resignation Process', 'Notice Period Policy', 
    'Exit Interview', 'Rehire Policy', 'Full & Final Settlement', 
    'Non-Disclosure Agreement'
]


# **************************** NORMALIZE FILTERS ****************************
def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans dictionary values for consistency in retrieval."""
    return {
        str(k).lower().strip(): (v if isinstance(v, bool) else str(v).lower().strip())
        for k, v in filters.items()
        if v not in [None, "", "null", "none", "unknown"]
    }




# **************************** STRUCTURED SCHEMA ****************************

from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

# [DEPT_LITERALS, CATEGORY_LITERALS, POLICY_NAME_LITERALS, SUBCATEGORY_LITERALS remain the same]

class HRMetadata(BaseModel):
    """Structured metadata for HR policy retrieval."""
    department: Optional[DEPT_LITERALS] = Field(None, description="The department name")
    category: Optional[CATEGORY_LITERALS] = Field(None, description="The high-level policy category")
    policy_name: Optional[POLICY_NAME_LITERALS] = Field(None, description="The full literal policy name")
    subcategory: Optional[SUBCATEGORY_LITERALS] = Field(None, description="The specific policy sub-topic")

def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans dictionary values for consistency in retrieval."""
    return {
        str(k).lower().strip(): (v if isinstance(v, bool) else str(v).lower().strip())
        for k, v in filters.items()
        if v not in [None, "", "null", "none", "unknown"]
    }

def extract_metadata_from_query(llm, question: str) -> Dict[str, Any]:
    """
    Uses structured output with explicit tool-call enforcement to prevent 400 errors.
    """
    try:
        # We bind the tool and force the model to use it
        structured_llm = llm.with_structured_output(HRMetadata)
        
        # Enhanced prompt to handle typos and pronouns explicitly
        prompt = f"""
        You are a strict HR Metadata Extractor. Your ONLY job is to call the HRMetadata tool.
        
        STRICT MAPPING RULES:
        1. POLICY_NAME: Must be the FULL literal (e.g., 'Notice Period Policy — Separation & Exit').
        2. PRONOUNS: If the user says 'it', 'this', 'that', or 'the leave', leave category, policy_name, and subcategory NULL.
        3. TYPOS: If the user makes a typo (e.g., 'caarry'), map it to the closest valid literal (e.g., 'Carry Forward' -> 'Earned Leave' or relevant policy).
        4. MANDATORY: You MUST return a tool call. Do not provide conversational text.
        
        User Input: {question}
        """
        
        # Invoke the LLM
        result = structured_llm.invoke(prompt)
        
        # Convert Pydantic model to dict, excluding None values
        extracted = result.model_dump(exclude_none=True) if result else {}
        
        # --- DETERMINISTIC FALLBACKS ---
        q_lower = question.lower().strip()
        
        # Fallback for Department
        if "department" not in extracted:
            depts = ['operations', 'engineering', 'sales', 'customer support', 'marketing', 'hr', 'finance', 'legal', 'admin', 'it']
            for d in depts:
                if d in q_lower:
                    extracted["department"] = d
                    break
        
        # Fallback for Subcategory typos (e.g., 'sick' -> 'Sick Leave')
        if "subcategory" not in extracted:
            if "sick" in q_lower: extracted["subcategory"] = "Sick Leave"
            elif "casual" in q_lower: extracted["subcategory"] = "Casual Leave"
            elif "notice" in q_lower or "period" in q_lower: extracted["subcategory"] = "Notice Period Policy"

        return normalize_filters(extracted)
        
    except Exception as e:
        # Catching the 400 error here allows chat.py to use its Topic Guard fallback
        logger.error(f"Metadata Extraction Tool Call Failed: {e}")
        return {}