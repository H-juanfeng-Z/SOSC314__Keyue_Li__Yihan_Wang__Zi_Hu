"""Read-only CSV contract/inventory check. Never downloads or edits input data."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
HEADERS={'Questions.csv':{'Id','OwnerUserId','CreationDate','Score','Title','Body'},'Answers.csv':{'Id','OwnerUserId','CreationDate','ParentId','Score','Body'},'Tags.csv':{'Id','Tag'}}
def inspect(root):
 limit=sys.maxsize
 while True:
  try:csv.field_size_limit(limit);break
  except OverflowError:limit//=10
 result={}
 for name,expected in HEADERS.items():
  p=root/name
  if not p.is_file():raise FileNotFoundError(f'Obtain {name} from the source archive and place it in {root}')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  with p.open(encoding='utf-8-sig',errors='replace',newline='') as f:
   rows=csv.reader(f);header=next(rows);assert set(header)==expected and len(header)==len(expected),(name,header)
   n=badchars=0
   for row in rows:
    if len(row)!=len(header):raise ValueError(f'{name}: row {rows.line_num} has {len(row)} fields, expected {len(header)}')
    n+=1;badchars+=sum(v.count('\ufffd') for v in row)
  result[name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size,'columns':header,'data_rows':n,'replacement_characters_after_decoding':badchars}
 return {'source':'https://www.kaggle.com/datasets/stackoverflow/pythonquestions','files':result,'scope':'Input contract and inventory, not certification of historical snapshot identity.'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-dir',type=Path,default=Path('data/raw_csv'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve() in {(a.raw_dir/n).resolve() for n in HEADERS}:raise ValueError('Inventory must not overwrite an input')
 out=inspect(a.raw_dir);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
