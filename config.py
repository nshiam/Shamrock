from dataclasses import dataclass

@dataclass
class ShamrockConfig():
    FILE_PATH = None
    NCORES = 5
    START = 0
    END = 100