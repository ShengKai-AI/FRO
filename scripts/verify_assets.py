import argparse,hashlib,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();receipt=json.loads(a.receipt.read_text())
    for row in receipt['completed']:
        path=a.receipt.parent/row['path']
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError(str(path))
    print('Verified',len(receipt['completed']),'files')
if __name__=='__main__':main()
