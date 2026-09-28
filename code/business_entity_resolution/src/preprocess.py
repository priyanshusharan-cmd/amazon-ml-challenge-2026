import pandas as pd
import re
import unicodedata
import unidecode

# Precompile regexes for massive speedup

# Domain and URL cleaning
domain_suffix_re = re.compile(r'\.(com|org|net|in|fr|co\.in|co|biz|info)\b', re.I)
url_prefix_re = re.compile(r'^(?:https?://)?(?:www\.)?', re.I)

# Indic script detection
indic_re = re.compile(r'[\u0900-\u0D7F]')

# Indic phonetic legal & domain terms
indic_phonetic = [
    (re.compile(r'\bpra+[iy]?[vw]?[a-z]*[td]+\b', re.I), 'private'),
    (re.compile(r'\blimitt?e?dd?\b', re.I), 'limited'),
    (re.compile(r'\b(?:elelpii|ellpii)\b', re.I), 'llp'),
    (re.compile(r'\bi[Nn]ttrne?shnl\b', re.I), 'international'),
    (re.compile(r'\binphraasttr[a-z]*\b', re.I), 'infrastructure'),
    (re.compile(r'\bmaarketti[Nn]g\b', re.I), 'marketing'),
    (re.compile(r'\bpro[Nn]?prttiij?\b', re.I), 'properties'),
    (re.compile(r'\bknsttraakshn\b', re.I), 'construction'),
    (re.compile(r'\b(?:lo?njisttiks|lojisttiks)\b', re.I), 'logistics'),
    (re.compile(r'\bso?lyuush[a-z]*\b', re.I), 'solutions'),
    (re.compile(r'\bteknol[a-z]*\b', re.I), 'technology'),
    (re.compile(r'\benttr?praa?[iy]z[a-z]*\b', re.I), 'enterprises'),
    (re.compile(r'\bkaarp[a-z]*\b', re.I), 'corporation'),
]

# Legal entities (English & Indian)
pvt_re = re.compile(r'\b(pvt|pv\.t|private)\b', re.I)
ltd_re = re.compile(r'\b(ltd|lt\.d|limited)\b', re.I)
corp_re = re.compile(r'\b(corp|corp\.|corporation)\b', re.I)
inc_re = re.compile(r'\b(inc|inc\.|incorporated)\b', re.I)
llc_re = re.compile(r'\b(llc|l\.l\.c)\b', re.I)
and_re = re.compile(r'\b(&)\b')
punct_re = re.compile(r'[^\w\s]')
spaces_re = re.compile(r'\s+')
alphanumeric_re = re.compile(r'[^a-z0-9]')

# French legal entities
sarl_re = re.compile(r'\b(sarl|s\.a\.r\.l)\b', re.I)
sas_re = re.compile(r'\b(sas|s\.a\.s|sasu|s\.a\.s\.u)\b', re.I)
sa_re = re.compile(r'\b(sa|s\.a)\b', re.I)
eurl_re = re.compile(r'\b(eurl|e\.u\.r\.l)\b', re.I)
sci_re = re.compile(r'\b(sci|s\.c\.i)\b', re.I)
snc_re = re.compile(r'\b(snc|s\.n\.c)\b', re.I)
ste_re = re.compile(r'\b(ste|soc|societe)\b', re.I)
ets_re = re.compile(r'\b(ets|etablissement|etablissements)\b', re.I)

# Address abbreviations (English & Indian)
st_re = re.compile(r'\b(st|st\.|str)\b', re.I)
rd_re = re.compile(r'\b(rd|rd\.)\b', re.I)
ave_re = re.compile(r'\b(ave|ave\.|av|av\.)\b', re.I)
blvd_re = re.compile(r'\b(blvd|blvd\.|bd|bd\.|bvd|bvd\.)\b', re.I)
dr_re = re.compile(r'\b(dr|dr\.)\b', re.I)
ln_re = re.compile(r'\b(ln|ln\.)\b', re.I)
apt_re = re.compile(r'\b(apt|apt\.)\b', re.I)
ste_addr_re = re.compile(r'\b(ste|ste\.)\b', re.I)

# Address abbreviations (French)
rue_re = re.compile(r'\b(r|rue)\b', re.I)
che_re = re.compile(r'\b(chem|chemin)\b', re.I)
imp_re = re.compile(r'\b(imp|impasse)\b', re.I)
all_re = re.compile(r'\b(all|allee)\b', re.I)
pl_re = re.compile(r'\b(pl|place)\b', re.I)
rte_re = re.compile(r'\b(rte|route)\b', re.I)
cedex_re = re.compile(r'\bcedex\b', re.I)

postal_re = re.compile(r'\b\d{5,6}\b')
street_num_re = re.compile(r'^(?:#\s*\d+|kh\s+no\.?\s*[-/]?[\w/]+|plot\s+no\.?\s*\w+|flat\s+no\.?\s*\w+|[a-z]-\d+(?:/\d+)?|\d+(?:[a-z]\b|\s+(?:bis|ter)\b|\b))', re.I)

def normalize_text(text):
    """
    Full business name normalizer:
    1. Strips URLs and domain extensions (.com, .in, etc.)
    2. Transliterates Indic scripts (Hindi, Bengali, Telugu, etc.) to Latin phonetics
    3. Normalizes phonetic Indic legal suffixes and common words
    4. Strips European accents (NFKD -> ascii)
    5. Standardizes corporate suffixes (private, limited, corp, llc, sarl, sas)
    """
    if pd.isna(text) or not text:
        return ""
    text = str(text).strip()
    
    # 1. Strip domain/URL artifacts
    text = url_prefix_re.sub('', text)
    text = domain_suffix_re.sub('', text).strip(' /')
    
    # 2. Transliterate Indic scripts if present
    if indic_re.search(text):
        text = unidecode.unidecode(text)
        for pat, repl in indic_phonetic:
            text = pat.sub(repl, text)
            
    # 3. Unicode normalization to strip accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()
    
    # 4. Standardize abbreviations and legal suffixes
    text = pvt_re.sub('private', text)
    text = ltd_re.sub('limited', text)
    text = corp_re.sub('corporation', text)
    text = inc_re.sub('incorporated', text)
    text = llc_re.sub('llc', text)
    text = and_re.sub('and', text)
    
    # French legal entities
    text = sarl_re.sub('sarl', text)
    text = sas_re.sub('sas', text)
    text = sa_re.sub('sa', text)
    text = eurl_re.sub('eurl', text)
    text = sci_re.sub('sci', text)
    text = snc_re.sub('snc', text)
    text = ste_re.sub('societe', text)
    text = ets_re.sub('etablissements', text)
    
    # Remove punctuation
    text = punct_re.sub(' ', text)
    text = spaces_re.sub(' ', text).strip()
    return text

def get_compact_signature(name):
    """
    Returns alphanumeric signature stripped of common legal suffixes for exact DBA/domain matching.
    e.g., 'Warner Silver Blueport Inc' -> 'warnersilverblueport'
    """
    if not name:
        return ""
    norm = normalize_text(name)
    # Strip common legal suffixes
    for suffix in ['private limited', 'private', 'limited', 'corporation', 'incorporated', 'llc', 'sarl', 'sas', 'sa', 'eurl', 'societe', 'etablissements']:
        if norm.endswith(suffix):
            norm = norm[:-len(suffix)].strip()
            break
    return alphanumeric_re.sub('', norm)

def get_acronym(name):
    """
    Computes acronym prefix + last word for acronym domain matching.
    e.g., 'Primary Care Partners LLC' -> 'pcpartners'
    """
    if not name:
        return ""
    norm = normalize_text(name)
    words = [w for w in norm.split() if w not in {'private', 'limited', 'corporation', 'incorporated', 'llc', 'sarl', 'sas', 'sa', 'and'}]
    if len(words) >= 2:
        return ''.join(w[0] for w in words[:-1]) + words[-1]
    return ""

def decompose_address(text):
    """
    Decomposes raw address into:
    (clean_addr, street_num, street_name, city_state, postal_code)
    """
    if pd.isna(text) or not text or str(text).strip() == "":
        return "", "", "", "", ""
        
    raw = str(text)
    # Extract postal code before punctuation strip
    pc_match = postal_re.search(raw)
    postal_code = pc_match.group(0) if pc_match else ""
    
    # Normalize accents & basic case
    norm_raw = unicodedata.normalize('NFKD', raw).encode('ascii', 'ignore').decode('utf-8').lower()
    
    # Standardize common address terms (English & French)
    norm_raw = st_re.sub('street', norm_raw)
    norm_raw = rd_re.sub('road', norm_raw)
    norm_raw = ave_re.sub('avenue', norm_raw)
    norm_raw = blvd_re.sub('boulevard', norm_raw)
    norm_raw = dr_re.sub('drive', norm_raw)
    norm_raw = ln_re.sub('lane', norm_raw)
    norm_raw = apt_re.sub('apartment', norm_raw)
    norm_raw = ste_addr_re.sub('suite', norm_raw)
    norm_raw = rue_re.sub('rue', norm_raw)
    norm_raw = che_re.sub('chemin', norm_raw)
    norm_raw = imp_re.sub('impasse', norm_raw)
    norm_raw = all_re.sub('allee', norm_raw)
    norm_raw = pl_re.sub('place', norm_raw)
    norm_raw = rte_re.sub('route', norm_raw)
    norm_raw = cedex_re.sub('', norm_raw)
    
    # Split by commas or semicolons
    parts = [p.strip() for p in re.split(r'[,;]+', norm_raw) if p.strip()]
    if not parts:
        clean_addr = spaces_re.sub(' ', punct_re.sub(' ', norm_raw)).strip()
        return clean_addr, "", "", "", postal_code
        
    first_part = parts[0]
    num_m = street_num_re.match(first_part)
    if num_m:
        street_num = num_m.group(0).strip()
        street_name = first_part[len(num_m.group(0)):].strip(' -#/,')
    else:
        tokens = first_part.split()
        if tokens and tokens[0].isdigit():
            street_num = tokens[0]
            street_name = ' '.join(tokens[1:])
        else:
            street_num = ""
            street_name = first_part
            
    city_state = ' '.join(parts[1:]) if len(parts) > 1 else ""
    if postal_code:
        city_state = city_state.replace(postal_code, '').strip()
        
    # Clean up strings
    clean_addr = spaces_re.sub(' ', punct_re.sub(' ', norm_raw)).strip()
    street_num = spaces_re.sub(' ', punct_re.sub(' ', street_num)).strip()
    street_name = spaces_re.sub(' ', punct_re.sub(' ', street_name)).strip()
    city_state = spaces_re.sub(' ', punct_re.sub(' ', city_state)).strip()
    
    return clean_addr, street_num, street_name, city_state, postal_code

def normalize_address(text):
    """
    Backward-compatible wrapper for code expecting (clean_addr, postal_code).
    """
    clean_addr, _, _, _, postal_code = decompose_address(text)
    return clean_addr, postal_code

def preprocess_dataframe(df):
    """
    Applies normalization to business_name and business_address.
    """
    print("Normalizing text columns...")
    df['norm_name'] = df['business_name'].apply(normalize_text)
    
    address_features = df['business_address'].apply(normalize_address)
    df['norm_address'] = address_features.apply(lambda x: x[0])
    df['postal_code'] = address_features.apply(lambda x: x[1])
    return df
