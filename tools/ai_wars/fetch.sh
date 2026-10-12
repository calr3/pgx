#!/bin/bash
# Download the A.I. Wars web edition's engine, maps and Cybug library (Tactical
# Neuronics; free, and free for educational use) and decrypt the Cybugs to plain CAICL.
# These are the reference for tools/ai_wars/conformance.py and are not part of the repo.
#
#   tools/ai_wars/fetch.sh [dest]      (default ~/calr3gh/aiwars_src; needs curl, python3, node)
#
# The site serves an incomplete certificate chain, so curl runs with -k (public files only).
set -euo pipefail
DEST=${1:-$HOME/calr3gh/aiwars_src}
B=https://www.tacticalneuronics.com/AIWars_Web
mkdir -p "$DEST/Maps" "$DEST/Cybugs" "$DEST/plain"
cd "$DEST"
for f in engine_core.js engine_caicl.js engine_decrypt.js engine_worker.js manifest.json; do
  curl -sSLkf -o "$f" "$B/$f"
done
python3 - "$B" <<'P'
import json, subprocess, sys, urllib.parse
B = sys.argv[1]
m = json.load(open("manifest.json"))
for p in ["Maps/" + x for x in m["maps"]] + ["Cybugs/" + c["file"] for c in m["cybugs"]]:
    subprocess.run(["curl", "-sSLkf", "-o", p, B + "/" + urllib.parse.quote(p)], check=False)
P
node -e '
const {decryptAI}=require("./engine_decrypt.js");const fs=require("fs");
for(const f of fs.readdirSync("Cybugs")){ if(!/\.ai$/i.test(f))continue;
 const raw=fs.readFileSync("Cybugs/"+f,"latin1"); const l=decryptAI(raw);
 let t=l?l.join("\n"):raw; t=t.replace(/\n+$/,"\n"); fs.writeFileSync("plain/"+f,t);}'
echo "maps: $(ls Maps/*.map | wc -l)  cybugs: $(ls plain | wc -l)  in $DEST"
