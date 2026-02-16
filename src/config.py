'''Centralized configuration values for the project.

These constants replace scattered 'magic numbers' and make tuning easier.
'''

from pathlib import Path

DATA_DIR: Path = Path('data/raw')
DEFAULT_RF_RATE: float = 0.0
PERIODS_PER_YEAR: int = 252
CRYPTO_PERIODS_PER_YEAR: int = 365
DEFAULT_COST_BPS: int = 10
