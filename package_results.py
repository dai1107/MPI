"""Verify the final evidence and create the submission package (no graph data)."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT/'output/experiment'
if not EXPERIMENT.exists() and (ROOT/'experiment').exists():
    EXPERIMENT=ROOT/'experiment'
rows=json.loads((EXPERIMENT/'summary.json').read_text(encoding='utf-8'))
assert len(rows)==57 and all(r['correct'] and r['serial_median'] for r in rows)
checks=list((EXPERIMENT/'results').glob('check_*.log'))
assert len(checks)==10
assert all(p.read_text(encoding='utf-8').count('检查通过')==3 for p in checks)
assert (EXPERIMENT/'results/profiles.log').read_text().strip()=='PROFILES_ALL_DONE'
expected='76d9dd8b8a2e4f15f75af065fba01c75b5a8b8fe8b0d612c9a4ec9f55323da20'
assert hashlib.sha256((ROOT/'code/main').read_bytes()).hexdigest()==expected
pdf_dir=ROOT/'output/pdf'
if not pdf_dir.exists(): pdf_dir=ROOT/'report'
pdf=next(pdf_dir.glob('*.pdf'))
assert len(PdfReader(pdf).pages)==4
paths=[]
for p in (ROOT/'code').iterdir():
    if p.is_file(): paths.append((p,'code/'+p.name))
for p in EXPERIMENT.rglob('*'):
    if p.is_file(): paths.append((p,'experiment/'+p.relative_to(EXPERIMENT).as_posix()))
paths.append((pdf,'report/'+pdf.name))
for name in ['analyze_results.py','build_report.py','package_results.py']:
    paths.append((ROOT/name,name))
manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for p,name in paths}
(ROOT/'output').mkdir(exist_ok=True)
(ROOT/'output/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
package=ROOT/'output/MPI实验三_完整提交包.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
    for p,name in paths:
        assert p.suffix!='.bin'
        z.write(p,name)
    z.write(ROOT/'output/manifest.json','manifest.json')
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    assert not any(n.endswith('.bin') for n in z.namelist())
    print('Package files:',len(z.namelist()))
print('Correctness: 30/30; performance checks: 57/57')
print('PDF: 4 pages; package bytes:',package.stat().st_size)
print('Package:',package)
