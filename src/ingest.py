import pandas as pd
import json

from typing import List, Dict, Any
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger(__name__)



# *************************** LOAD CSV *************************************
def load_csv(file_path: str) -> pd.DataFrame:
    """" Load CSV File"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading CSV: {e}")





# ************************* CLEAN AND NORMALIZE DATA ****************** 
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Normalize, clean, Remove white space to the features"""
    df = df.copy()

    df.fillna("", inplace=True)

    # Normalize important columns 
    normalize_cols = [
    "department",
    "applicable_to",
    "category",
    "subcategory",
    "is_mandatory",
    "review_cycle"
]

    for col in normalize_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Clean text columns 
    text_cols = ["policy_name", "description"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
      

    return df

# *********************************** PARSE DETAILS COLUMN *****************************************
def parse_details_column(df: pd.DataFrame) -> pd.DataFrame:
    """Details colums is parse ex: '{}' > {}"""
    def safe_parse(x: str) -> Dict[str, Any]:
        try:
            return json.loads(x) if x else {}
        except:
            return {}

    df = df.copy()
    df["details_parsed"] = df["details"].apply(safe_parse)

    return df




# ********************************* CONVERT JSON TO TEXT ******************************************
def convert_details_to_text(details):
    """ Convert details column to textual format """
    parts = []

    for k, v in details.items():
        key = k.replace("_", " ")
        parts.append(f"The {key} is {v}")

    return ". ".join(parts)





# *********************************** BUILD DOCUMENT TEXT *******************************************
def build_document_text(row: pd.Series) -> str:
    """ Convert each row into text for document creation """
    
    details_text = convert_details_to_text(row.get("details_parsed", {}))

    return f"""
Policy Name: {row.get('policy_name', '')}

This policy applies specifically to:
Department: {row.get('department', '')}
Applicable To: {row.get('applicable_to', '')}

Category: {row.get('category', '')}
Subcategory: {row.get('subcategory', '')}

Description:
{row.get('description', '')}

Policy Details:
{details_text}
""".strip()



# **************************** METADATA CREATION ********************************
def build_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract important metadata for metadata filtering"""
   
    return {
        "policy_id": row.get("policy_id"),
        "policy_name": row.get("policy_name"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "department": row.get("department"),
        "version": row.get("version"),
        "year": row.get("last_updated_year"),
        "is_mandatory": row.get("is_mandatory"),
        "review_cycle": row.get("review_cycle"),
    }


# ******************************* CONVERT TO DOCUMENTS ********************************
def create_documents(df: pd.DataFrame) -> List[Document]:
    """Maunllay convert into document objects"""

    documents = []

    for _, row in df.iterrows():
        try:
            doc_text = build_document_text(row)
            metadata = build_metadata(row)

            documents.append(
                Document(
                    page_content=doc_text,
                    metadata=metadata
                )
            )
        except:
            logger.error("Error")
            continue

    return documents

# *********************************** DOCUMENT INGESTION *****************************

def ingest_pipeline(file_path: str) -> List[Document]:
    """Perform all the task in document ingestion like loading, cleaning, parsing and document creation"""
    df = load_csv(file_path)
    df = clean_dataframe(df)
    df = parse_details_column(df)

    documents = create_documents(df)

    logger.info(documents)
    return documents