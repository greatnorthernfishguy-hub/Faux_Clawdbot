import re

path = "/home/josh/ecosystem-monitor/monitor.py"
content = open(path).read()

pattern = r'    "ces_monitor":\s*\{[^}]+\},\n'
if re.search(pattern, content):
    content = re.sub(pattern, "", content)
    changed_svc = True
else:
    changed_svc = False

sentinel = "# -------------------
"
entry = (
    "# [2026-04-18] Codemine (BLK-EM-144) -- Remove defunct ces_monitor endpoint
"
    "#   What: Removed ces_monitor (port 8847) from SERVICES dict
"
    "#   Why:  Port 8847 dead since Apr 5 ecosystem restructure; spurious down-count in logs
"
    "# -------------------
"
)
if entry not in content and sentinel in content:
    content = content.replace(sentinel, entry, 1)
    changed_cl = True
else:
    changed_cl = False

open(path, "w").write(content)
print("PATCHED svc=%s cl=%s" % (changed_svc, changed_cl))
