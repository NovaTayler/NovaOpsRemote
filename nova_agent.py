import time, os
from pathlib import Path
from subprocess import Popen

COMMAND_FILE = Path("/home/novatayler2025/NovaOpsRemote/commands/latest.txt")
LOG_FILE = Path("/opt/novaswarm/logs/command_exec.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

while True:
    if COMMAND_FILE.exists():
        with open(COMMAND_FILE) as f:
            cmd = f.read().strip()
            if cmd:
                with open(LOG_FILE, "a") as log:
                    log.write(f"\n\n[EXEC] {cmd}\n")
                    proc = Popen(cmd, shell=True, stdout=log, stderr=log)
                    proc.wait()
                COMMAND_FILE.write_text("")  # Clear the command
    time.sleep(5)
