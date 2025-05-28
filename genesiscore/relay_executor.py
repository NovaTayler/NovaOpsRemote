import os
import time
import subprocess
from datetime import datetime

COMMAND_FILE = os.path.expanduser("~/NovaOpsRemote/command/latest.txt")
LOG_FILE = os.path.expanduser("~/NovaOpsRemote/genesiscore/relay_exec.log")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[EXEC] {datetime.utcnow().isoformat()} - {msg}\n")

def read_and_execute():
    if not os.path.exists(COMMAND_FILE):
        log("No command file found.")
        return

    with open(COMMAND_FILE, "r") as f:
        command = f.read().strip()

    if command:
        log(f"Executing command: {command}")
        try:
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, timeout=60)
            log(f"Result:\n{result.decode()}")
        except subprocess.CalledProcessError as e:
            log(f"Error:\n{e.output.decode()}")
        except Exception as ex:
            log(f"Exception: {str(ex)}")

while True:
    read_and_execute()
    time.sleep(30)
