#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MANIFEST="models/manifest.json"
WEIGHTS_DIR="models/weights"

usage() {
  echo "usage: models/fetch.sh <model-id>"
  echo "       models/fetch.sh list"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

id="$1"

if [[ "$id" == "list" ]]; then
  python3 -c "
import json
m = json.load(open('$MANIFEST'))
for e in m['models']:
    print(f\"{e['id']:60s} {e['type']:12s} {e['size_gb']}GB\")
"
  exit 0
fi

entry=$(python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
for e in m['models']:
    if e['id'] == '$id':
        sys.stdout.write(json.dumps(e)); sys.exit(0)
sys.exit(1)
" || true)

if [[ -z "$entry" ]]; then
  echo "model '$id' not in manifest (see: models/fetch.sh list)"
  exit 1
fi

repo=$(python3 -c "import json; print(json.loads('''$entry''')['repo'])")
revision=$(python3 -c "import json; print(json.loads('''$entry''')['revision'])")
dest="$WEIGHTS_DIR/$id"

if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
  echo "already downloaded: $dest"
  exit 0
fi

mkdir -p "$dest"
echo "downloading $repo ($revision) from Hugging Face into $dest"

python3 -c "
import json, os, subprocess, sys
m = json.load(open('$MANIFEST'))
e = next(x for x in m['models'] if x['id'] == '$id')
repo = e['repo']
tree = json.loads(subprocess.check_output([
    'curl', '-sf', f'https://huggingface.co/api/models/{repo}/tree/main?recursive=true'
]))
paths = [f['path'] for f in tree]
with open(os.path.expanduser('~/.cache/hf-fetch-paths.txt'), 'w') as fh:
    fh.write('\n'.join(paths))
sys.stderr.write(f'{len(paths)} files\n')
"

dest="$WEIGHTS_DIR/$id"
while IFS= read -r path; do
  out="$dest/$path"
  mkdir -p "$(dirname "$out")"
  curl -sfL "https://huggingface.co/$repo/resolve/$revision/$path" -o "$out"
done < ~/.cache/hf-fetch-paths.txt

rm -f ~/.cache/hf-fetch-paths.txt

echo "done: $dest"
echo "note: weights are gitignored and never pushed to GitHub."