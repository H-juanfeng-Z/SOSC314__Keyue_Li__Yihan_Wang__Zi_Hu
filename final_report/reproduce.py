"""Replay the published report from audited summaries without model inference."""
from pathlib import Path
import json,sys,tempfile,shutil,hashlib
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

def main():
    root=Path(__file__).resolve().parent;out=root/'reproduced';out.mkdir(exist_ok=True)
    source=root/'report_data.json';expected=json.loads((root/'provenance.json').read_text(encoding='utf8'))['report_data_sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'Report data hash mismatch'
    for name in ['report_data.json','qwen_example.json','qwen_prompts.json','code_element_support.json','code_element_summary.json','recompute_code_element.py','code_element_original.py']:shutil.copyfile(root/name,out/name)
    shutil.copytree(root/'assets',out/'assets',dirs_exist_ok=True)
    for name in ['README.md','provenance.json','partner_feature_audit.json','feature_id_support.json','partner_feature_review.py']:
        shutil.copyfile(root/name,out/name)
    nb=nbformat.read(root/'report.ipynb',as_version=4)
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    with tempfile.TemporaryDirectory(prefix='report-kernel-',dir=out) as tmp:
        k=Path(tmp)/'sosc314';k.mkdir()
        (k/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'SOSC314','language':'python'}),encoding='utf8')
        km=KernelManager(kernel_name='sosc314',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
        try:NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(out)}}).execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
    nbformat.validate(nb);nbformat.write(nb,out/'report.ipynb')
    ex=HTMLExporter();ex.exclude_input=True
    body,_=ex.from_notebook_node(nb);(out/'index.html').write_text(body,encoding='utf8')
    print('Report replayed:',out/'index.html')
if __name__=='__main__':main()
