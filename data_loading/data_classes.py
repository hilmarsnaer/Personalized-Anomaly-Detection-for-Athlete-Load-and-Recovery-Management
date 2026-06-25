# This file is used to define the data classes for the project

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Session:
    player_id: str
    session: str
    rows: Optional[int]
    columns: Optional[int]

    avg_speed: Optional[float]
    max_speed: Optional[float]
    avg_heart_rate: Optional[float]
    max_heart_rate: Optional[float]

    total_distance: Optional[float]
    acceleration_impulse: Optional[float]
    high_speed_distance: Optional[float]
    
    duration:  Optional[float] = None
    
    injury: Optional[bool] = None
    future_injury: Optional[bool] = None # The player sustains an injury after this session, and there are no sessions between this session and the injury date

    @classmethod
    def from_csv_row(cls, row):
        def to_float(value):
            if value in ("", None):
                return None
            value = float(value)
            return None if value == 0 else value

        def to_int(value):
            if value in ("", None):
                return None
            return int(value)

        return cls(
            player_id=row["player_id"],
            session=row["session"],
            rows=to_int(row["rows"]),
            columns=to_int(row["columns"]),

            avg_speed=to_float(row["avg_speed"]),
            max_speed=to_float(row["max_speed"]),
            avg_heart_rate=to_float(row["avg_heart_rate"]),
            max_heart_rate=to_float(row["max_heart_rate"]),

            total_distance=to_float(row["total_distance"]),
            acceleration_impulse=to_float(row["acceleration_impulse"]),
            high_speed_distance=to_float(row["high_speed_distance"]),
        )


@dataclass
class Player:
    player_id: str
    sessions: Dict[str, Session] = field(default_factory=dict)
    unmodified_values: Dict[str, List[float]] = field(default_factory=dict)
    injury_history: List[str] = field(default_factory=list)

    def add_injury_date(self, injury_date: str):
        if injury_date not in self.injury_history:
            self.injury_history.append(injury_date)

    def has_injury_on_date(self, date: str) -> bool:
        return date in self.injury_history

    def add_session(self, session: Session):
        """Add a session to the player."""
        self.sessions[session.session] = session
        

    def get_session(self, session: str):
        return self.sessions.get(session)

    def number_of_sessions(self) -> int:
        return len(self.sessions)

    def build_unmodified_values(self):
        """
        Build raw time series for this player from all sessions.
        """

        self.unmodified_values = {
            "avg_speed": [],
            "max_speed": [],
            "avg_heart_rate": [],
            "max_heart_rate": [],
            "total_distance": [],
            "acceleration_impulse": [],
            "high_speed_distance": [],
            "supervised_pca": [],
            "spca_ewma": [],
            "spca_bayesian": [],
            "spca_pop": [],
        }

        sorted_sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.session
        )

        for session in sorted_sessions:
            self.unmodified_values["avg_speed"].append(session.avg_speed)
            self.unmodified_values["max_speed"].append(session.max_speed)
            self.unmodified_values["avg_heart_rate"].append(session.avg_heart_rate)
            self.unmodified_values["max_heart_rate"].append(session.max_heart_rate)
            self.unmodified_values["total_distance"].append(session.total_distance)
            self.unmodified_values["acceleration_impulse"].append(session.acceleration_impulse)
            self.unmodified_values["high_speed_distance"].append(session.high_speed_distance)
            # supervised_pca may be attached to session objects later; include if present
            spca_val = getattr(session, "supervised_pca", None) 
            self.unmodified_values["supervised_pca"].append(spca_val)
            # model-specific PCA variants
            self.unmodified_values["spca_ewma"].append(getattr(session, "spca_ewma", None))
            self.unmodified_values["spca_bayesian"].append(getattr(session, "spca_bayesian", None))
            self.unmodified_values["spca_pop"].append(getattr(session, "spca_pop", None))



@dataclass
class PlayerBaselines:
    player_id: str

    ewma_baselines: Dict[str, List[float]] = field(default_factory=dict)
    bayesian_baselines: Dict[str, List[float]] = field(default_factory=dict)
    int_vs_ext_baselines: Dict[str, Any] = field(default_factory=dict)
    



@dataclass
class PlayerDeviationScores:
    player_id: str

    ewma_z_scores: Dict[str, List[float]] = field(default_factory=dict)
    bayesian_z_scores: Dict[str, List[float]] = field(default_factory=dict)
    population_based_moving_average_z_scores: Dict[str, List[float]] = field(default_factory=dict)
    external_vs_internal_z_scores: Dict[str, List[float]] = field(default_factory=dict)

    ewma_flags: Dict[str, List[str]] = field(default_factory=dict)
    bayesian_flags: Dict[str, List[str]] = field(default_factory=dict)
    population_based_moving_average_flags: Dict[str, List[str]] = field(default_factory=dict)
    external_vs_internal_flags: Dict[str, List[str]] = field(default_factory=dict)



@dataclass
class PlayerWorkloadMetrics:
    player_id: str

    ewma_values: Dict[str, List[float]] = field(default_factory=dict)
    acwr_values: Dict[str, List[float]] = field(default_factory=dict)
    mswr_values: Dict[str, List[float]] = field(default_factory=dict)
   
    def get_series(self, method: str) -> List[List[float]]:
        """
        Return all metric time series for a given method
        as a list of lists.

        method: "ewma", "acwr", or "mswr"
        """

        if method == "ewma":
            data = self.ewma_values
        elif method == "acwr":
            data = self.acwr_values
        elif method == "mswr":
            data = self.mswr_values
        else:
            raise ValueError("Method must be 'ewma', 'acwr', or 'mswr'")

        return list(data.values())

    def get_series_with_names(self, method: str):
        if method == "ewma":
            return self.ewma_values
        elif method == "acwr":
            return self.acwr_values
        elif method == "mswr":
            return self.mswr_values
        else:
            raise ValueError("Method must be 'ewma', 'acwr', or 'mswr'")

    def get_feature_matrix(self, method: str):
        """
        Returns a matrix where:
        rows = time steps
        columns = metrics
        """

        data = self.get_series_with_names(method)

        return list(zip(*data.values()))