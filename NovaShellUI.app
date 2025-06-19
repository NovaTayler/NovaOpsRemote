#!/usr/bin/env python3
"""Simple Nova Shell Desktop UI."""
from pathlib import Path
from subprocess import run, PIPE
from tkinter import Tk, Text, Entry, Button, END, filedialog, messagebox

LOG_PATH = Path("novadash/command_log.txt")

class NovaShellApp:
    def __init__(self, master: Tk):
        self.master = master
        master.title("NovaShell")
        self.log = Text(master, height=20, width=80)
        self.log.pack()
        self.entry = Entry(master, width=80)
        self.entry.pack()
        Button(master, text="Run", command=self.execute).pack()

    def execute(self):
        cmd = self.entry.get().strip()
        if not cmd:
            return
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        output = result.stdout or result.stderr
        entry = f"CMD: {cmd}\nOUT: {output}\n\n"
        if LOG_PATH.exists():
            LOG_PATH.write_text(LOG_PATH.read_text() + entry)
        else:
            LOG_PATH.write_text(entry)
        self.log.insert(END, output + "\n")
        self.entry.delete(0, END)

def main():
    root = Tk()
    app = NovaShellApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
