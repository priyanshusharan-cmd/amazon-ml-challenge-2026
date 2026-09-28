import re
import unidecode
from rapidfuzz import fuzz

indic_re = re.compile(r'[\u0900-\u0D7F]')
domain_suffix_re = re.compile(r'\.(com|org|net|in|fr|co\.in|co|biz|info)\b', re.I)
url_prefix_re = re.compile(r'^(?:https?://)?(?:www\.)?', re.I)

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
]

def transliterate_indic_text(text):
    if not text:
        return ""
    if indic_re.search(text):
        text = unidecode.unidecode(text)
        for pat, repl in indic_phonetic:
            text = pat.sub(repl, text)
    return text

def parse_address_components(addr_str):
    if not addr_str:
        return '', '', '', '', ''
    
    pc_match = re.search(r'\b\d{5,6}\b', addr_str)
    postal_code = pc_match.group(0) if pc_match else ''
    
    clean_addr = addr_str.lower().strip()
    parts = [p.strip() for p in re.split(r'[,;]+', clean_addr) if p.strip()]
    if not parts:
        return clean_addr, '', '', '', postal_code
        
    first_part = parts[0]
    # Fixed street number regex:
    # Matches: '105', '5 bis', '5b', 'kh no. -570/13', 'c-66', 'plot no 12', '#12'
    num_match = re.match(r'^(?:#\s*\d+|kh\s+no\.?\s*[-/]?[\w/]+|plot\s+no\.?\s*\w+|flat\s+no\.?\s*\w+|[a-z]-\d+(?:/\d+)?|\d+(?:[a-z]\b|\s+(?:bis|ter)\b|\b))', first_part, re.I)
    if num_match:
        street_num = num_match.group(0).strip()
        street_name = first_part[len(num_match.group(0)):].strip(' -#/,')
    else:
        tokens = first_part.split()
        if tokens and tokens[0].isdigit():
            street_num = tokens[0]
            street_name = ' '.join(tokens[1:])
        else:
            street_num = ''
            street_name = first_part
            
    city_state = ' '.join(parts[1:]) if len(parts) > 1 else ''
    if postal_code:
        city_state = city_state.replace(postal_code, '').strip()
        
    return clean_addr, street_num, street_name, city_state, postal_code

if __name__ == "__main__":
    cases = [
        '5 bis Rue Pierre Dignac, La Teste-de-Buch, 33260',
        '105 ELM ST, MORGANTON, NC 28655',
        'KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041',
        '3033 Robin Hill Lane, Garland, TX',
        'Robin Hill Lane, Garland, Texas',
        'G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh'
    ]
    for c in cases:
        ca, sn, sna, cs, pc = parse_address_components(c)
        print(f"Input: {c}\n  -> Num: '{sn}' | Street: '{sna}' | City/State: '{cs}' | PC: '{pc}'\n")
