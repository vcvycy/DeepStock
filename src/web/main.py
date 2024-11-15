import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

class RestartHandler(FileSystemEventHandler):
    def __init__(self, command):
        self.command = command
        self.process = None
        self.start_process()

    def start_process(self):
        self.process = subprocess.Popen(self.command, shell=True)

    def restart_process(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.start_process()

    def on_modified(self, event):
        print(event.src_path)
        if event.src_path.endswith("index.py"):
            print("="*100)
            print(f"{event.src_path} has been modified; restarting process.")
            print("="*100)
            self.restart_process()


if __name__ == "__main__":
    command = "python -m web.index"

    event_handler = RestartHandler(command)
    observer = Observer()
    observer.schedule(event_handler, "./", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()