import sys
import logging
from datetime import datetime
from pathlib import Path

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

log_dir = Path.cwd() / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = log_dir / LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("sob")
