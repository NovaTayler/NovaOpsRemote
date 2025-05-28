import os
import time
from datetime import datetime

COMMAND_FILE = os.path.expanduser("~/NovaOpsRemote/command/latest.txt")
LOG_FILE = os.path.expanduser("~/NovaOpsRemote/genesiscore/relay.out")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[NovaRelay] {datetime.utcnow().isoformat()} - {msg}\n")

def run_loop():
    last_cmd = ""
    while True:
        try:
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, "r") as f:
                    cmd = f.read().strip()
                if cmd and cmd != last_cmd:
                    log(f"Executing: {cmd}")
                    os.system(cmd)
                    last_cmd = cmd
        except Exception as e:
            log(f"ERROR: {e}")
        time.sleep(5)

if __name__ == "__main__":
    log("NovaShell Relay Online")
    run_loop()
