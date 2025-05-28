import time
from datetime import datetime

def ping_loop():
    while True:
        with open("/home/novatayler2025/NovaOpsRemote/genesiscore/telemetry.log", "a") as log:
            log.write(f"[PING] {datetime.utcnow()} - System alive\n")
        time.sleep(30)

if __name__ == "__main__":
    ping_loop()
