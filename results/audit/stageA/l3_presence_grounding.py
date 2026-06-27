"""L3 — presence-grounding, FIXED (broadened patterns).

Old L3 (run_stage_a.py) only matched `^the text uses the phrase`. Spec says
"uses/CONTAINS the phrase", and there are other verbatim-quote templates
(uses the word/term, quotes) that are equally string-matchable but were skipped.

This script: for every quoted-literal presence claim, extract the quoted X and
string-match (case-insensitive) against the source `text`. Reports per-pattern
claim counts + hallucination rate (X not found in source), and OLD vs NEW
coverage. stdlib only.
"""
import csv, re, glob, collections
csv.field_size_limit(2**31 - 1)

# Verbatim-quote presence templates. "old" = the only one the original L3 matched.
PATTERNS = [
    ("uses the phrase",    True),   # original (the ONLY one matched before)
    ("contains the phrase", False), # spec said "uses/contains" but code missed it
    ("uses the word",      False),
    ("uses the term",      False),
    ("quotes",             False),
]
# pull the quoted literal X (straight or curly quotes), >=2 chars
QUOTE_RE = re.compile(r"[\"'‘’“”]([^\"'‘’“”]{2,}?)[\"'‘’“”]")

def extract_X(prop, pat):
    # take the first quoted span occurring AFTER the pattern keyword
    i = prop.lower().find(pat)
    if i < 0: return None
    m = QUOTE_RE.search(prop, i)
    return m.group(1) if m else None

def clean(s):                      # lowercase + collapse whitespace
    return re.sub(r"\s+", " ", (s or "").lower()).strip()
def is_present(X, src):
    # strip EDGE punctuation from the quoted phrase, normalize whitespace, substring-match.
    # (matches the report's normalized check; raw exact-match overcounts ~15x on trailing commas)
    x = clean(X).strip(" .,;:!?\"'‘’“”")
    return bool(x) and x in clean(src)

stats = {p: {"claims":0, "X_extracted":0, "halluc":0} for p,_ in PATTERNS}
files = glob.glob("experiments/*/accuracy_outputs/*.csv")
for f in files:
    try: rows = list(csv.DictReader(open(f, newline="")))
    except: continue
    for r in rows:
        prop = (r.get("property") or "").strip()
        pl = prop.lower()
        src = r.get("text") or ""
        for pat,_ in PATTERNS:
            if pat in pl:
                st = stats[pat]; st["claims"] += 1
                X = extract_X(prop, pat)
                if X:
                    st["X_extracted"] += 1
                    if not is_present(X, src): st["halluc"] += 1
                break  # one bucket per claim (first match wins)

print(f"scanned {len(files)} accuracy_outputs files\n")
print(f"{'pattern':22s}{'claims':>9s}{'X-quoted':>10s}{'halluc':>9s}{'halluc%':>9s}{'in old L3?':>12s}")
old_c=old_h=new_c=new_h=0
for pat,is_old in PATTERNS:
    s=stats[pat]; hp = s["halluc"]/s["X_extracted"]*100 if s["X_extracted"] else 0
    print(f"{pat:22s}{s['claims']:>9d}{s['X_extracted']:>10d}{s['halluc']:>9d}{hp:>8.1f}%{('YES' if is_old else 'MISSED'):>12s}")
    new_c+=s["X_extracted"]; new_h+=s["halluc"]
    if is_old: old_c+=s["X_extracted"]; old_h+=s["halluc"]
print(f"\nOLD L3 (uses-the-phrase only): {old_c} quoted claims, halluc {old_h} ({old_h/old_c*100 if old_c else 0:.1f}%)")
print(f"NEW L3 (broadened):           {new_c} quoted claims, halluc {new_h} ({new_h/new_c*100 if new_c else 0:.1f}%)")
print(f"=> broadening adds {new_c-old_c} string-matchable presence claims ({(new_c-old_c)/old_c*100 if old_c else 0:.0f}% more coverage)")
