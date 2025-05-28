import os
import subprocess
from datetime import datetime

REPO_PATH = "/home/novatayler2025/NovaOpsRemote"

def git_push():
    now = datetime.utcnow().isoformat()
    os.chdir(REPO_PATH)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"AutoPush: {now}"], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=True)

if __name__ == "__main__":
    git_push()
from datetime import datetime

# ... your git logic ...

timestamp = datetime.utcnow().isoformat()
with open("/home/novatayler2025/NovaOpsRemote/genesiscore/autopush.log", "a") as log:
    log.write(f"[AutoPush] Commit: {timestamp}\n")
