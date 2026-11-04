import argparse,json,subprocess
from exp.baseline.runtime.paths import ROOT,repo

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    entries=json.loads((ROOT/'manifests/baselines.yaml').read_text())
    for name,entry in entries.items():
        target=repo(name);print(name,entry['url'],entry['revision'])
        if not a.execute:continue
        if not target.exists():subprocess.run(['git','clone',entry['url'],str(target)],check=True)
        current=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(target),'diff','--name-only'],text=True).strip()
        if current!=entry['revision']:
            if dirty:raise RuntimeError('Refuse to replace modified upstream checkout: '+str(target))
            subprocess.run(['git','-C',str(target),'fetch','origin',entry['revision']],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach',entry['revision']],check=True)
        if name=='OmniSelect':
            patch=ROOT/'exp/baseline/methods/omniselect.patch'
            applied=subprocess.run(['git','-C',str(target),'apply','--reverse','--check',str(patch)],capture_output=True).returncode==0
            if not applied:subprocess.run(['git','-C',str(target),'apply',str(patch)],check=True)
if __name__=='__main__':main()
