from __future__ import annotations

import pandas as pd
from rich.console import Console

from .config import PATHS

console = Console()


def load_flight_level() -> pd.DataFrame:
    path = PATHS.data_raw / "flight_level.csv"
    df = pd.read_csv(path)
    console.log(f"Loaded flight level data: {df.shape}")
    return df


def load_pnr_flight_level() -> pd.DataFrame:
    path = PATHS.data_raw / "pnr_flight_level.csv"
    df = pd.read_csv(path)
    console.log(f"Loaded PNR flight level data: {df.shape}")
    return df


def load_pnr_remarks() -> pd.DataFrame:
    path = PATHS.data_raw / "pnr_remarks.csv"
    df = pd.read_csv(path)
    console.log(f"Loaded PNR remarks data: {df.shape}")
    return df


def load_bag_level() -> pd.DataFrame:
    path = PATHS.data_raw / "bag_level.csv"
    df = pd.read_csv(path)
    console.log(f"Loaded bag level data: {df.shape}")
    return df


def load_airports() -> pd.DataFrame:
    path = PATHS.data_raw / "airports.csv"
    df = pd.read_csv(path)
    console.log(f"Loaded airports data: {df.shape}")
    return df
