#!/bin/bash

# NovaPush Bridge Script
# Pulls latest changes from GitHub and logs the output

LOGFILE="/home/novatayler2025/NovaOpsRemote/novapush.log"
REPO_DIR="/home/novatayler2025/NovaOpsRemote"

echo "==== $(date) ====" >> $LOGFILE
cd $REPO_DIR
git pull origin main >> $LOGFILE 2>&1
