from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from .config import AppConfig
from .utils import setup_logging
from .viewer import EEGAcquisitionWindow


def main() -> None:
    setup_logging()
    app = QApplication(sys.argv)
    window = EEGAcquisitionWindow(config=AppConfig())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
