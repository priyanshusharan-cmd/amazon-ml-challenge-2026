"""
Data Normalization Module for Business Names and Addresses.
Preserves original fields, clean normalized fields, and extracted blocking keys.
"""

import re
import unicodedata

# Legal suffixes and ultra-common noise words
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", 
    "pvt", "private", "gmbh", "sarl", "sa", "co", "company", "enterprises", 
    "enterprise", "services", "service", "group", "holdings", "holding", 
    "tech", "technologies", "technology", "intl", "international", "solutions",
    "solution", "traders", "trading", "store", "stores", "mart", "center", "centre",
    "p", "v", "t"
}

def normalize_text(text: str, remove_suffixes: bool = False) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unicode NFKD decomposition & ASCII convert
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Handle '&' and '+' -> 'and'
    text = re.sub(r"[\&\+]", " and ", text)
    
    # 4. Remove punctuation, keep alphanumeric and whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    
    # 5. Tokenize & collapse whitespace
    tokens = text.split()
    
    if remove_suffixes and len(tokens) > 1:
        filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
        if filtered:
            tokens = filtered
            
    return " ".join(tokens)

def extract_blocking_keys(name_norm: str, name_clean: str, addr_norm: str) -> dict:
    """Extracts fast, high-recall blocking keys for candidate generation."""
    clean_toks = name_clean.split() if name_clean else []
    norm_toks = name_norm.split() if name_norm else []
    
    # 1. First 4 chars of clean name
    clean_no_space = "".join(clean_toks)
    prefix_4 = clean_no_space[:4] if len(clean_no_space) >= 4 else clean_no_space
    
    # 2. Sorted first two tokens of clean name
    first_2 = ""
    if len(clean_toks) >= 2:
        first_2 = "_".join(sorted(clean_toks[:2]))
    elif len(clean_toks) == 1:
        first_2 = clean_toks[0]
        
    # 3. Extract numbers from address (street number, PIN/ZIP code if present)
    nums = re.findall(r"\b\d{3,6}\b", addr_norm) if addr_norm else []
    addr_num = nums[0] if nums else ""
    
    return {
        "prefix_4": prefix_4,
        "first_2_tokens": first_2,
        "addr_num": addr_num,
        "clean_tokens": clean_toks
    }

def get_tokens(text_norm: str) -> list:
    if not text_norm:
        return []
    return text_norm.split()


