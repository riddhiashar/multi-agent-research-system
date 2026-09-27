import logging
import os

os.makedirs("data/logs", exist_ok=True)

logger = logging.getLogger("research_system")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("data/logs/system.log")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_agent_event(agent_name: str, event: str, **kwargs):
    """Structured log line: agent name, what happened, and any extra context."""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{agent_name}] {event} {extra}".strip())
