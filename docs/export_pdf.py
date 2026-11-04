from pathlib import Path
import json
from playwright.sync_api import sync_playwright
import fitz

R=Path(__file__).resolve().parent
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
 page=browser.new_page(viewport={'width':1000,'height':1300},device_scale_factor=1)
 errors=[]
 page.on('pageerror',lambda err:errors.append(str(err)))
 page.goto((R/'experiment_plan.html').as_uri(),wait_until='networkidle')
 page.evaluate('document.fonts.ready')
 page.emulate_media(media='print')
 layout=page.evaluate('''() => [...document.querySelectorAll('.page')].map(s=>({page:s.dataset.page,height:s.clientHeight,scroll:s.scrollHeight,lastContentBottom:s.children[s.children.length-2].getBoundingClientRect().bottom-s.getBoundingClientRect().top,footerTop:s.querySelector('footer').getBoundingClientRect().top-s.getBoundingClientRect().top}))''')
 font_status=page.evaluate('''() => [...document.fonts].map(f=>({family:f.family,status:f.status,weight:f.weight}))''')
 page.pdf(path=str(R/'experiment_plan.pdf'),format='A4',prefer_css_page_size=True,print_background=True,display_header_footer=False)
 browser.close()
d=fitz.open(R/'experiment_plan.pdf')
text='\n'.join(x.get_text() for x in d)
fonts=sorted(set(f[3] for pg in d for f in pg.get_fonts()))
validation={'html_to_pdf':'Chromium / Playwright','pages':len(d),'page_sizes_points':[list(pg.rect)[2:] for pg in d],'embedded_fonts':fonts,'font_load':font_status,'layout':layout,'page_errors':errors,'reference_last_page':d[-1].get_text().startswith('References'),'links':sum(len(pg.get_links()) for pg in d)}
(R/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
for i in [0,2,5,len(d)-1]:d[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(R/f'preview_{i+1}.png')
print(json.dumps(validation,ensure_ascii=False,indent=2))
assert len(d)==7
assert 'References' in d[-1].get_text() and 'Benchmarks' in d[-1].get_text() and 'Methods' in d[-1].get_text()
assert all(x not in text for x in ['ActivityNet','PerceptionTest','MLVU','Gemini'])
assert all(abs(pg.rect.width-595.28)<1 and abs(pg.rect.height-841.89)<1 for pg in d)
assert any('TimesNewRoman' in f for f in fonts) and any('SimSun' in f for f in fonts)
assert all(x['lastContentBottom']<x['footerTop'] for x in layout)
