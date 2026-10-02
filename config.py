from dataclasses import dataclass


@dataclass
class ShamrockConfig:
    FILE_PATH: str | None = None
    NCORES: int = 1
    HEADER_OVERRIDE: bool = True  # only True or False
    START: int = 0
    END: int | None = None
    ZERO_POINT: float = 31.4  # 31.4 for new SNANA sims or 27.5 for DES data
    SNR: float = 5.0

    ID_FILE: str | None = None
    PHOTFLAG: bool = False  # only True or False

    GRIZ_MIN: float = 0.0
    GRIZ_MAX: float = 0.9
    RIZ_MIN: float = 0.0
    RIZ_MAX: float = 1.2
    ZBIN_WIDTH: float = 0.1
    X1_LOWER: float = -3.5
    X1_UPPER: float = 3.5
    C_LOWER: float = -0.2
    C_UPPER: float = 0.5
    DELTA_MAG: float = 1.0
    ALPHA_COSMO: float = 0.16
    BETA_COSMO: float = 3.0
    PICKLE_FILES: bool = False  # only True or False

    
    #will have to change these when I pay attention
    RUN_DIR: str | None = None
    TYPE_Z: str = "PHOTOZ"
    SAVE_FILE: str | None = None

   
  