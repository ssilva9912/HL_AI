import os

# Tests must never preload models or open the real vector store at app start-up.
os.environ.setdefault("HOMELAB_WARMUP_ON_STARTUP", "false")
