from torch.utils.data import Dataset, DataLoader
import re
import html
import urllib.parse
import unicodedata
import pandas as pd
from torch.utils.data import Dataset



UUID_RE = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-'
                     r'[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
                     r'[0-9a-fA-F]{12}\b')

BASE64_RE = re.compile(r'\b[A-Za-z0-9+\-_]{20,}={0,2}\b')
HEX_RE = re.compile(r'\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{16,}\b')
HASH_RE = re.compile(r'\b[a-fA-F0-9]{32,64}\b')
IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
TIME_RE = re.compile(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b')
FLOAT_RE = re.compile(r'\b\d+\.\d+\b')
INT_RE = re.compile(r'\b\d+\b')


def recursive_url_decode(text, max_iter=3):
    prev = text
    for _ in range(max_iter):
        curr = urllib.parse.unquote(prev)
        if curr == prev:
            break
        prev = curr
    return prev


def normalize_text(text):
    text = recursive_url_decode(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return text



def abstract_values(text):
    # 1. UUIDs are most specific
    text = UUID_RE.sub("<UUID>", text)
    # 2. IPs (Do this before floats/ints)
    text = IP_RE.sub("<IP>", text)
    # 3. Emails (Specific structure)
    text = EMAIL_RE.sub("<EMAIL>", text)
    # 4. Times
    text = TIME_RE.sub("<TIME>", text)
    # 5. Hashes (Do this BEFORE Base64, because a Hash looks like Base64)
    text = HASH_RE.sub("<HASH>", text)
    # 6. Base64 (Catches long random strings/JWTs)
    text = BASE64_RE.sub("<B64>", text)
    # 7. Hex (0x...)
    text = HEX_RE.sub("<HEX>", text)
    # 8. Floats
    text = FLOAT_RE.sub("<FLOAT>", text)
    # 9. Integers (Least specific, catches leftovers)
    text = INT_RE.sub("<INT>", text)
    return text


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


class HTTPLogsDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file).fillna('').astype(str)

        # Precompute normalized sequences once (important for performance)
        self.sequences = self.df.apply(self.build_sequence, axis=1).tolist()

    def canonicalize(self, text):
        text = normalize_text(text)
        text = abstract_values(text)
        text = normalize_whitespace(text)
        return text

    def build_sequence(self, row):
        # Support both new CSV columns and potential old ones
        method = (row.get("method") or row.get("request.method") or "").upper()
        
        raw_path = row.get("path") or row.get("request.path") or ""
        path = self.canonicalize(raw_path)
        
        protocol = row.get("protocol") or row.get("request.protocol") or ""
        
        # Truncate raw body BEFORE constructing sequence to save space for tags
        raw_body = row.get("request_body") or row.get("body") or ""
        if len(raw_body) > 1000: 
            raw_body = raw_body[:1000] 
        body = self.canonicalize(raw_body)

        # Use Content-Length for WAF (Request header), not Body Bytes Sent (Response)
        # CSV doesn't have content_length column, try to parse from headers or default to 0
        content_len_val = "0"
        if "content_length" in row:
            content_len_val = row["content_length"]
        # Optional: could parse headers here but avoiding overhead for now
            
        content_len = abstract_values(str(content_len_val))

        return (
            f"<request_method> {method} </request_method> "
            f"<request_path> {path} </request_path> "
            f"<request_protocol> {protocol} </request_protocol> "
            f"<content_length> {content_len} </content_length> "
            f"<request_body> {body} </request_body>"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
