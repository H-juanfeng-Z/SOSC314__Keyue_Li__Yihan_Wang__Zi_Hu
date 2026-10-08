"""Verify/unpack the frozen release and optionally recompute CPU summaries.
Never contacts a server, runs inference, executes post code, or overwrites a workspace.
"""
import argparse,gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
MANIFEST=REPO/'tables/annotation_diagnostics/followup/release_manifest.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def comparable(obj):
    # Runtime and local provenance change on replay; substantive outputs must not.
    ignored={'runtime_seconds','elapsed_seconds','source_sha256','script_sha256','script','path','inputs','source_sha256','source_manifest_sha256'}
    if isinstance(obj,dict):return {k:comparable(v) for k,v in obj.items() if k not in ignored}
    if isinstance(obj,list):return [comparable(v) for v in obj]
    return obj
def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    files=read(MANIFEST)['files'];payloads=[]
    for f in files:
        src=(REPO/f['path']).resolve();assert src.is_relative_to(REPO)
        raw=src.read_bytes();assert sha(raw)==f['sha256'],str(src)
        raw=gzip.decompress(raw) if src.suffix=='.gz' else raw
        assert sha(raw)==f['uncompressed_sha256'],f['path']
        payloads.append((f,raw))
    print(json.dumps({'verified_files':len(files)}),flush=True)
    if not a.work_dir:
        assert not a.run,'--run needs a NEW --work-dir';return
    work=a.work_dir.resolve();assert not work.exists(),'Use a new directory; existing files are never replaced'
    work.mkdir(parents=True);root=work/'experiments';root.mkdir()
    for f,raw in payloads:
        dest=(root/f['restore_path']).resolve();assert dest.is_relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    if not a.run:print(str(root));return
    results=[]
    def run(name,script,args=(),expected=(),clear=(),stdout_output=None):
        folder=root/name;before={s:read(folder/s) for s in expected}
        for s in clear:
            dest=(folder/s).resolve();assert dest.is_relative_to(root)
            if dest.exists():dest.unlink() # Only disposable outputs unpacked into this newly created workspace.
        env=dict(os.environ,PYTHONUTF8='1',PYTHONHASHSEED='0')
        cmd=[sys.executable,str(folder/script)]+[str(folder/v[1:]) if v.startswith('@') else v for v in args]
        proc=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,encoding='utf8')
        (folder/(script+'.stdout.txt')).write_text(proc.stdout,encoding='utf8')
        (folder/(script+'.stderr.txt')).write_text(proc.stderr,encoding='utf8')
        if proc.returncode:raise RuntimeError(name+'/'+script+'\n'+proc.stderr[-4000:])
        if stdout_output:(folder/stdout_output).write_text(proc.stdout,encoding='utf8')
        checks={s:comparable(before[s])==comparable(read(folder/s)) for s in expected}
        results.append(dict(experiment=name,script=script,returncode=0,archived_comparisons=checks))
        (work/'reproduction_checks.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        assert all(checks.values()),(name,checks)
        print(json.dumps(results[-1]),flush=True)
    run('measurement_expansion_20260930','analyze.py',['--root','@results','--selection','@selection_private.json'],['results/analysis.json'])
    run('semantic_diagnostics_20261005','analyze.py',['--root','@results/newserver_20261007'],['results/newserver_20261007/analysis.json'])
    run('rubric_check_20261007','analyze.py',['--root','@results'],['results/analysis.json'])
    run('evidence_dependence_20261007','analyze.py',['--root','@results'],['results/analysis.json'])
    run('association_sensitivity_20261007','experiment.py',expected=['results/analysis.json'],clear=['results/analysis.json','results/progress.json'])
    run('association_sensitivity_20261007','threshold_check.py',expected=['results/threshold_analysis.json'],clear=['results/threshold_analysis.json'])
    run('answer_trajectory_20261007','analyze.py',['--root','@results'],['results/analysis.json'])
    run('integrated_20261008','analyze.py',expected=['analysis.json'])
    run('explicit_evidence_20261008','analyze.py',stdout_output='recomputed_analysis.json')
    run('final_trajectory_20261008','batch.py',expected=['results/cohorts.json','results/authors.json','results/intent_groups.json','results/cross_model.json','results/question_level.json'],clear=['results/summary.json'])
    run('final_trajectory_20261008','verify.py',expected=['results/verification.json'])
    run('gpt_case_review_20261008','analyze_review.py',expected=['case_results.json','review_comparison.json'])
    run('post_review_20261008','preserve_context.py',expected=['results/context_candidate_checks.json'])
    print('Finished CPU reproduction; no new model calls. Full raw-HTML audit requires original processed Parquet inputs.')
if __name__=='__main__':main()
