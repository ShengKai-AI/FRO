"""Fetch AudioCLIP resources; resume partial transfers and enforce fixed SHA256."""
import argparse,hashlib,json,shutil,subprocess
from exp.baseline.runtime.paths import ROOT,asset,repo

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    for record in json.loads((ROOT/'manifests/auxiliary.json').read_text()):
        folder=asset('weights') if record['destination']=='weights' else repo('OmniSelect')/'AudioCLIP/assets'
        target=folder/record['name'];print(record['url'],'->',target)
        if not a.execute:continue
        folder.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            if shutil.disk_usage(folder).free<record['bytes']+2**30:raise OSError('Insufficient space')
            partial=target.with_suffix(target.suffix+'.partial')
            subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-','--output',str(partial),record['url']],check=True)
            with partial.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            if sha!=record['sha256']:raise ValueError('Auxiliary checksum mismatch')
            partial.replace(target)
        with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=record['sha256']:raise ValueError('Auxiliary checksum mismatch')
    if a.execute:
        import nltk
        for name in ['punkt_tab','averaged_perceptron_tagger_eng']:
            if not nltk.download(name,download_dir=str(asset('weights')/'nltk_data'),quiet=True):raise RuntimeError('NLTK resource download failed: '+name)
if __name__=='__main__':main()
