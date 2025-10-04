from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized access to key project directories."""

    root: Path = Path(__file__).resolve().parents[1]

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts_figures(self) -> Path:
        return self.root / "artifacts" / "figures"

    @property
    def artifacts_tables(self) -> Path:
        return self.root / "artifacts" / "tables"

    @property
    def artifacts_models(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def output_score(self) -> Path:
        return self.root / "skyhack.csv"

    @property
    def report_file(self) -> Path:
        return self.reports_dir / "report.txt"

    @property
    def duckdb_path(self) -> Path:
        return self.root / "data" / "interim" / "skyhack.duckdb"


PATHS = ProjectPaths()
