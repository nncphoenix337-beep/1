# ================================================================
# AI MT5 TRADER - SMC ENGINE V4.3.11
# SMART MONEY CONCEPT ONLY
# H4 = MAPPING | H1 = MAPPING/POI | M15 = SETUP | M5 = CONFIRM
# CLOSED CANDLES ONLY | CAUSAL EVENTS | PERSISTENT POI
# ================================================================

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import MetaTrader5 as mt5


# ================================================================
# CONFIGURATION
# ================================================================

SYMBOL = "XAUUSD"

H4_BARS = 800
H1_BARS = 1600
M15_BARS = 3500
M5_BARS = 10000

# Causal swing confirmation
SWING_LEFT = 3
SWING_RIGHT = 3

# Liquidity
LIQUIDITY_TOLERANCE = 1.50

# M15 sequence
MAX_STRUCTURE_AFTER_SWEEP = 20
MAX_DISPLACEMENT_AFTER_STRUCTURE = 8
MAX_POI_AFTER_DISPLACEMENT = 8

# M5 sequence
M5_CONFIRM_LOOKBACK = 150
M5_MAX_CONFIRM_BARS = 50
M5_LOCAL_LOOKBACK = 12

# Displacement
DISPLACEMENT_BODY_RATIO = 0.65
DISPLACEMENT_RANGE_MULTIPLIER = 1.15
DISPLACEMENT_LOOKBACK = 10

# FVG
MIN_FVG_SIZE = 0.20

# Order Block
OB_LOOKBACK = 6

# POI validation
POI_TOUCH_TOLERANCE = 0.05
MAX_M5_DISTANCE_FROM_POI = 3.00
MAX_POI_DISPLACEMENT_GAP = 8.00
MIN_ZONE_OVERLAP = 0.01

# M5 touch validation
MAX_TOUCH_RANGE_MULTIPLIER = 3.0
MAX_TOUCH_RANGE_ABSOLUTE = 20.0
MAX_TOUCH_CLOSE_DISTANCE_MULTIPLIER = 3.0
MAX_TOUCH_CLOSE_DISTANCE_ABSOLUTE = 5.0


# ================================================================
# DATA CLASSES
# ================================================================

@dataclass
class SwingPoint:
    idx: int
    time: pd.Timestamp
    price: float
    kind: str
    label: str = ""


@dataclass
class StructureEvent:
    idx: int
    time: pd.Timestamp
    event: str
    direction: str
    price: float
    reference: str = ""


@dataclass
class LiquidityPool:
    pool_id: int
    direction: str
    price: float
    start_idx: int
    last_idx: int
    touches: int = 1
    consumed: bool = False


@dataclass
class Zone:
    zone_id: int
    zone_type: str
    direction: str
    high: float
    low: float
    created_idx: int
    created_time: pd.Timestamp
    source_idx: int
    state: str = "ACTIVE"
    mitigated_idx: Optional[int] = None
    invalidated_idx: Optional[int] = None


# ================================================================
# HELPERS
# ================================================================

def safe_float(value, default=0.0):
    try:
        result = float(value)

        if np.isnan(result):
            return default

        return result

    except Exception:
        return default


def fmt_price(value):
    if value is None:
        return "-"

    return f"{float(value):.2f}"


def fmt_time(value):
    if value is None:
        return "-"

    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")

    except Exception:
        return str(value)


# ================================================================
# SMC ENGINE
# ================================================================

class SMCEngine:

    def __init__(self, timeframe: str, df: pd.DataFrame):

        self.timeframe = timeframe

        self.df = self._prepare_data(df)

        self.swings_high: List[SwingPoint] = []
        self.swings_low: List[SwingPoint] = []

        self.structure: List[StructureEvent] = []
        self.events: List[StructureEvent] = []

        self.liquidity: List[LiquidityPool] = []

        self.fvg_zones: List[Zone] = []
        self.ob_zones: List[Zone] = []

        self.zones: List[Zone] = []

        self.bias = "NEUTRAL"

        self.protected_high = None
        self.protected_low = None


    # ============================================================
    # DATA PREPARATION
    # ============================================================

    def _prepare_data(self, df):

        if df is None or len(df) == 0:
            raise ValueError(
                f"{self.timeframe}: dataframe kosong"
            )

        d = df.copy()

        d.columns = [
            str(c).lower().strip()
            for c in d.columns
        ]

        required = [
            "time",
            "open",
            "high",
            "low",
            "close"
        ]

        missing = [
            c for c in required
            if c not in d.columns
        ]

        if missing:
            raise ValueError(
                f"{self.timeframe}: kolom hilang {missing}"
            )

        d["time"] = pd.to_datetime(
            d["time"],
            errors="coerce"
        )

        for column in [
            "open",
            "high",
            "low",
            "close"
        ]:

            d[column] = pd.to_numeric(
                d[column],
                errors="coerce"
            )

        d = (
            d
            .dropna(subset=required)
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        d["bull"] = d["close"] > d["open"]
        d["bear"] = d["close"] < d["open"]

        d["range"] = (
            d["high"] -
            d["low"]
        )

        d["body"] = (
            d["close"] -
            d["open"]
        ).abs()

        d["body_ratio"] = np.where(
            d["range"] > 0,
            d["body"] / d["range"],
            0.0
        )

        return d


    # ============================================================
    # SWING DETECTION
    # ============================================================

    def detect_swings(self):

        d = self.df

        highs = []
        lows = []

        if len(d) <= (
            SWING_LEFT +
            SWING_RIGHT
        ):
            return self

        for i in range(
            SWING_LEFT,
            len(d) - SWING_RIGHT
        ):

            high = float(
                d.iloc[i]["high"]
            )

            low = float(
                d.iloc[i]["low"]
            )

            left_high = float(
                d.iloc[
                    i - SWING_LEFT:i
                ]["high"].max()
            )

            right_high = float(
                d.iloc[
                    i + 1:i + SWING_RIGHT + 1
                ]["high"].max()
            )

            left_low = float(
                d.iloc[
                    i - SWING_LEFT:i
                ]["low"].min()
            )

            right_low = float(
                d.iloc[
                    i + 1:i + SWING_RIGHT + 1
                ]["low"].min()
            )

            # Confirmation happens only after RIGHT candles.
            confirmation_idx = (
                i + SWING_RIGHT
            )

            confirmation_time = (
                d.iloc[
                    confirmation_idx
                ]["time"]
            )

            if (
                high >= left_high
                and
                high >= right_high
            ):

                highs.append(
                    SwingPoint(
                        confirmation_idx,
                        confirmation_time,
                        high,
                        "HIGH"
                    )
                )

            if (
                low <= left_low
                and
                low <= right_low
            ):

                lows.append(
                    SwingPoint(
                        confirmation_idx,
                        confirmation_time,
                        low,
                        "LOW"
                    )
                )

        self.swings_high = highs
        self.swings_low = lows

        return self


    # ============================================================
    # SWING LABELING
    # ============================================================

    def label_swings(self):

        previous = None

        for swing in self.swings_high:

            if previous is None:
                swing.label = "H"

            elif swing.price > previous:
                swing.label = "HH"

            else:
                swing.label = "LH"

            previous = swing.price

        previous = None

        for swing in self.swings_low:

            if previous is None:
                swing.label = "L"

            elif swing.price > previous:
                swing.label = "HL"

            else:
                swing.label = "LL"

            previous = swing.price

        return self


    # ============================================================
    # STRUCTURE
    # ============================================================

    def process_structure(self):

        highs = {
            swing.idx: swing
            for swing in self.swings_high
        }

        lows = {
            swing.idx: swing
            for swing in self.swings_low
        }

        events = []

        bias = "NEUTRAL"

        external_high = None
        external_low = None

        protected_high = None
        protected_low = None

        broken_high = set()
        broken_low = set()

        for i in range(len(self.df)):

            row = self.df.iloc[i]

            close = float(row["close"])

            if i in highs:
                external_high = highs[i]

            if i in lows:
                external_low = lows[i]

            # ----------------------------------------------------
            # INITIAL STRUCTURE
            # ----------------------------------------------------

            if bias == "NEUTRAL":

                if (
                    external_high
                    and
                    close > external_high.price
                    and
                    external_high.idx not in broken_high
                ):

                    events.append(
                        StructureEvent(
                            i,
                            row["time"],
                            "INITIAL_BOS_UP",
                            "BUY",
                            close,
                            "external_high"
                        )
                    )

                    broken_high.add(
                        external_high.idx
                    )

                    bias = "BULL"

                    protected_low = (
                        external_low
                    )

                    continue

                if (
                    external_low
                    and
                    close < external_low.price
                    and
                    external_low.idx not in broken_low
                ):

                    events.append(
                        StructureEvent(
                            i,
                            row["time"],
                            "INITIAL_BOS_DOWN",
                            "SELL",
                            close,
                            "external_low"
                        )
                    )

                    broken_low.add(
                        external_low.idx
                    )

                    bias = "BEAR"

                    protected_high = (
                        external_high
                    )

                    continue


            # ----------------------------------------------------
            # BULL STRUCTURE
            # ----------------------------------------------------

            elif bias == "BULL":

                if (
                    external_high
                    and
                    close > external_high.price
                    and
                    external_high.idx not in broken_high
                ):

                    events.append(
                        StructureEvent(
                            i,
                            row["time"],
                            "BOS_UP",
                            "BUY",
                            close,
                            "external_high"
                        )
                    )

                    broken_high.add(
                        external_high.idx
                    )

                    if external_low:
                        protected_low = (
                            external_low
                        )

                    continue

                if (
                    protected_low
                    and
                    close < protected_low.price
                    and
                    protected_low.idx not in broken_low
                ):

                    events.append(
                        StructureEvent(
                            i,
                            row["time"],
                            "CHoCH_DOWN",
                            "SELL",
                            close,
                            "protected_low"
                        )
                    )

                    broken_low.add(
                        protected_low.idx
                    )

                    bias = "BEAR"

                    protected_high = (
                        external_high
                    )

                    continue


            # ----------------------------------------------------
            # BEAR STRUCTURE
            # ----------------------------------------------------

            else:

                if (
                    external_low
                    and
                    close < external_low.price
                    and
                    external_low.idx not in broken_low
                ):

                    events.append(
                        StructureEvent(
                            i,
                            row["time"],
                            "BOS_DOWN",
                            "SELL",
                            close,
                            "external_low"
                        )
                    )

                    broken_low.add(
                        external_low.idx
                    )

                    if external_high:
                        protected_high = (
                            external_high
                        )

                    continue

                if (
                    protected_high
                    and
                    close > protected_high.price
                    and
                    protected_high.idx not in broken_high
                ):

                    events.append(
                        StructureEvent(
                            i,
                            row["time"],
                            "CHoCH_UP",
                            "BUY",
                            close,
                            "protected_high"
                        )
                    )

                    broken_high.add(
                        protected_high.idx
                    )

                    bias = "BULL"

                    protected_low = (
                        external_low
                    )

                    continue

        self.structure = events

        self.events.extend(events)

        self.bias = bias

        self.protected_high = (
            protected_high
        )

        self.protected_low = (
            protected_low
        )

        return self


    # ============================================================
    # LIQUIDITY POOLS
    # ============================================================

    def build_liquidity_pools(self):

        pools = []

        # Highs = BUY SIDE LIQUIDITY
        # Lows  = SELL SIDE LIQUIDITY

        for direction, swings in (
            ("SELL", self.swings_high),
            ("BUY", self.swings_low)
        ):

            used = set()

            for i, swing in enumerate(swings):

                if i in used:
                    continue

                cluster = [swing]

                for j in range(
                    i + 1,
                    len(swings)
                ):

                    if j in used:
                        continue

                    if (
                        abs(
                            swings[j].price -
                            swing.price
                        )
                        <=
                        LIQUIDITY_TOLERANCE
                    ):

                        cluster.append(
                            swings[j]
                        )

                        used.add(j)

                average_price = float(
                    np.mean([
                        item.price
                        for item in cluster
                    ])
                )

                pools.append(
                    LiquidityPool(
                        0,
                        direction,
                        average_price,
                        min(
                            item.idx
                            for item in cluster
                        ),
                        max(
                            item.idx
                            for item in cluster
                        ),
                        len(cluster)
                    )
                )

        # Remove duplicate price regions.
        cleaned = []

        for pool in sorted(
            pools,
            key=lambda x: (
                x.direction,
                x.price
            )
        ):

            duplicate = False

            for existing in cleaned:

                if (
                    existing.direction ==
                    pool.direction
                    and
                    abs(
                        existing.price -
                        pool.price
                    )
                    <=
                    LIQUIDITY_TOLERANCE
                ):

                    duplicate = True
                    break

            if not duplicate:
                cleaned.append(pool)

        for number, pool in enumerate(
            cleaned,
            start=1
        ):

            pool.pool_id = number

        self.liquidity = cleaned

        return self


    # ============================================================
    # DISPLACEMENT
    # ============================================================

    def is_displacement(
        self,
        idx,
        direction
    ):

        if idx < DISPLACEMENT_LOOKBACK:
            return False

        if idx >= len(self.df):
            return False

        row = self.df.iloc[idx]

        average_range = safe_float(
            self.df.iloc[
                idx - DISPLACEMENT_LOOKBACK:idx
            ]["range"].mean()
        )

        if average_range <= 0:
            return False

        current_range = safe_float(
            row["range"]
        )

        if current_range <= 0:
            return False

        strong_body = (
            safe_float(
                row["body_ratio"]
            )
            >=
            DISPLACEMENT_BODY_RATIO
        )

        large_range = (
            current_range
            >=
            average_range *
            DISPLACEMENT_RANGE_MULTIPLIER
        )

        directional = (
            direction == "BUY"
            and
            bool(row["bull"])
        ) or (
            direction == "SELL"
            and
            bool(row["bear"])
        )

        return bool(
            strong_body
            and
            large_range
            and
            directional
        )


    # ============================================================
    # FAIR VALUE GAP
    # ============================================================

    def detect_fvg(self):

        d = self.df

        for i in range(
            2,
            len(d)
        ):

            first = d.iloc[i - 2]
            third = d.iloc[i]

            # Bullish FVG
            if (
                third["low"]
                >
                first["high"]
                and
                (
                    third["low"] -
                    first["high"]
                )
                >=
                MIN_FVG_SIZE
            ):

                self.fvg_zones.append(
                    Zone(
                        0,
                        "FVG",
                        "BUY",
                        float(third["low"]),
                        float(first["high"]),
                        i,
                        third["time"],
                        i - 1
                    )
                )

            # Bearish FVG
            elif (
                third["high"]
                <
                first["low"]
                and
                (
                    first["low"] -
                    third["high"]
                )
                >=
                MIN_FVG_SIZE
            ):

                self.fvg_zones.append(
                    Zone(
                        0,
                        "FVG",
                        "SELL",
                        float(first["low"]),
                        float(third["high"]),
                        i,
                        third["time"],
                        i - 1
                    )
                )

        return self


    # ============================================================
    # ORDER BLOCK
    # ============================================================

    def detect_order_blocks(self):

        d = self.df

        for i in range(
            1,
            len(d)
        ):

            if self.is_displacement(
                i,
                "BUY"
            ):

                direction = "BUY"

            elif self.is_displacement(
                i,
                "SELL"
            ):

                direction = "SELL"

            else:

                continue

            source = None

            for j in range(
                i - 1,
                max(
                    -1,
                    i - OB_LOOKBACK - 1
                ),
                -1
            ):

                if (
                    direction == "BUY"
                    and
                    bool(d.iloc[j]["bear"])
                ):

                    source = j
                    break

                if (
                    direction == "SELL"
                    and
                    bool(d.iloc[j]["bull"])
                ):

                    source = j
                    break

            if source is None:
                continue

            source_row = d.iloc[source]
            displacement_row = d.iloc[i]

            if direction == "SELL":

                gap = max(
                    0.0,
                    float(source_row["low"])
                    -
                    float(displacement_row["high"])
                )

            else:

                gap = max(
                    0.0,
                    float(displacement_row["low"])
                    -
                    float(source_row["high"])
                )

            if gap > MAX_POI_DISPLACEMENT_GAP:
                continue

            self.ob_zones.append(
                Zone(
                    0,
                    "OB",
                    direction,
                    float(source_row["high"]),
                    float(source_row["low"]),
                    i,
                    displacement_row["time"],
                    source
                )
            )

        return self


    # ============================================================
    # BUILD ZONES
    # ============================================================

    def build_zones(self):

        self.detect_fvg()

        self.detect_order_blocks()

        self.zones = (
            self.fvg_zones +
            self.ob_zones
        )

        for number, zone in enumerate(
            self.zones,
            start=1
        ):

            zone.zone_id = number

        return self


    # ============================================================
    # ZONE STATES
    # ============================================================

    def update_zone_states(self):

        d = self.df

        for zone in self.zones:

            for i in range(
                zone.created_idx + 1,
                len(d)
            ):

                row = d.iloc[i]

                high = float(
                    row["high"]
                )

                low = float(
                    row["low"]
                )

                close = float(
                    row["close"]
                )

                # BUY POI invalidation
                if (
                    zone.direction == "BUY"
                    and
                    close < zone.low
                ):

                    zone.state = (
                        "INVALIDATED"
                    )

                    zone.invalidated_idx = i

                    break

                # SELL POI invalidation
                if (
                    zone.direction == "SELL"
                    and
                    close > zone.high
                ):

                    zone.state = (
                        "INVALIDATED"
                    )

                    zone.invalidated_idx = i

                    break

                # First mitigation
                if (
                    high >= zone.low
                    and
                    low <= zone.high
                    and
                    zone.state == "ACTIVE"
                ):

                    zone.state = "MITIGATED"

                    zone.mitigated_idx = i

                    # Mitigation does not mean invalidation.

        return self


    # ============================================================
    # LIQUIDITY SWEEPS
    # ============================================================

    def detect_liquidity_sweeps(self):

        d = self.df

        for pool in self.liquidity:

            if pool.consumed:
                continue

            start = max(
                0,
                pool.last_idx + 1
            )

            for i in range(
                start,
                len(d)
            ):

                row = d.iloc[i]

                high = float(
                    row["high"]
                )

                low = float(
                    row["low"]
                )

                close = float(
                    row["close"]
                )

                # High liquidity sweep
                if pool.direction == "SELL":

                    swept = (
                        high > pool.price
                        and
                        close < pool.price
                    )

                    event_name = (
                        "LIQUIDITY_SWEEP_HIGH"
                    )

                # Low liquidity sweep
                else:

                    swept = (
                        low < pool.price
                        and
                        close > pool.price
                    )

                    event_name = (
                        "LIQUIDITY_SWEEP_LOW"
                    )

                if not swept:
                    continue

                self.events.append(
                    StructureEvent(
                        i,
                        row["time"],
                        event_name,
                        pool.direction,
                        pool.price,
                        f"pool_{pool.pool_id}"
                    )
                )

                pool.consumed = True

                break

        return self


    # ============================================================
    # ENGINE RUN
    # ============================================================

    def run(self):

        self.detect_swings()

        self.label_swings()

        self.process_structure()

        self.build_liquidity_pools()

        self.detect_liquidity_sweeps()

        self.build_zones()

        self.update_zone_states()

        self.events.sort(
            key=lambda event: event.idx
        )

        return self


    # ============================================================
    # STATISTICS
    # ============================================================

    def stats(self):

        return {

            "swing_high":
                len(self.swings_high),

            "swing_low":
                len(self.swings_low),

            "bos_up":
                sum(
                    e.event == "BOS_UP"
                    for e in self.structure
                ),

            "bos_down":
                sum(
                    e.event == "BOS_DOWN"
                    for e in self.structure
                ),

            "choch_up":
                sum(
                    e.event == "CHoCH_UP"
                    for e in self.structure
                ),

            "choch_down":
                sum(
                    e.event == "CHoCH_DOWN"
                    for e in self.structure
                ),

            "sweep_high":
                sum(
                    e.event ==
                    "LIQUIDITY_SWEEP_HIGH"
                    for e in self.events
                ),

            "sweep_low":
                sum(
                    e.event ==
                    "LIQUIDITY_SWEEP_LOW"
                    for e in self.events
                ),

            "fvg":
                len(self.fvg_zones),

            "ob":
                len(self.ob_zones),

            "active_zones":
                sum(
                    z.state == "ACTIVE"
                    for z in self.zones
                ),

            "mitigated_zones":
                sum(
                    z.state == "MITIGATED"
                    for z in self.zones
                ),

            "invalidated_zones":
                sum(
                    z.state == "INVALIDATED"
                    for z in self.zones
                )
        }


# ================================================================
# SMC SETUP ENGINE
# ================================================================

class SMCSetupEngine:

    def __init__(
        self,
        h4,
        h1,
        m15,
        m5
    ):

        self.h4 = h4
        self.h1 = h1
        self.m15 = m15
        self.m5 = m5


    # ============================================================
    # M15 SWEEPS
    # ============================================================

    def sweeps(self, direction):

        if direction == "BUY":

            event_name = (
                "LIQUIDITY_SWEEP_LOW"
            )

        else:

            event_name = (
                "LIQUIDITY_SWEEP_HIGH"
            )

        return sorted(
            [
                e
                for e in self.m15.events
                if e.event == event_name
            ],
            key=lambda x: x.idx,
            reverse=True
        )


    # ============================================================
    # M15 STRUCTURE AFTER SWEEP
    # ============================================================

    def structure_after(
        self,
        sweep,
        direction
    ):

        if direction == "BUY":

            valid_events = (
                "CHoCH_UP",
                "BOS_UP",
                "INITIAL_BOS_UP"
            )

            opposite_events = (
                "CHoCH_DOWN",
                "BOS_DOWN",
                "INITIAL_BOS_DOWN"
            )

        else:

            valid_events = (
                "CHoCH_DOWN",
                "BOS_DOWN",
                "INITIAL_BOS_DOWN"
            )

            opposite_events = (
                "CHoCH_UP",
                "BOS_UP",
                "INITIAL_BOS_UP"
            )

        end_idx = (
            sweep.idx +
            MAX_STRUCTURE_AFTER_SWEEP
        )

        candidates = [
            event
            for event in self.m15.structure
            if (
                event.event in valid_events
                and
                sweep.idx < event.idx <= end_idx
            )
        ]

        if not candidates:
            return None

        candidate = min(
            candidates,
            key=lambda x: x.idx
        )

        # Opposing structure before confirmation
        for event in self.m15.structure:

            if (
                event.event in opposite_events
                and
                sweep.idx < event.idx
                < candidate.idx
                and
                event.idx <= end_idx
            ):

                return None

        return candidate


    # ============================================================
    # M15 DISPLACEMENT
    # ============================================================

    def displacement_after(
        self,
        structure,
        direction
    ):

        start = (
            structure.idx + 1
        )

        end = min(
            len(self.m15.df),
            start +
            MAX_DISPLACEMENT_AFTER_STRUCTURE
        )

        for i in range(
            start,
            end
        ):

            if self.m15.is_displacement(
                i,
                direction
            ):

                return i

        return None


    # ============================================================
    # ZONE / CANDLE INTERSECTION
    # ============================================================

    def zone_intersects_candle(
        self,
        zone,
        row
    ):

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        overlap = (
            min(
                high,
                zone.high
            )
            -
            max(
                low,
                zone.low
            )
        )

        return (
            overlap > MIN_ZONE_OVERLAP,
            max(0.0, overlap)
        )


    # ============================================================
    # ZONE DISTANCE TO CANDLE
    # ============================================================

    def zone_distance_to_candle(
        self,
        zone,
        row
    ):

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        if high < zone.low:

            return (
                zone.low -
                high
            )

        if low > zone.high:

            return (
                low -
                zone.high
            )

        return 0.0


    # ============================================================
    # ZONE DISTANCE TO PRICE
    # ============================================================

    def zone_distance_to_price(
        self,
        zone,
        price
    ):

        price = float(price)

        if price < zone.low:

            return (
                zone.low -
                price
            )

        if price > zone.high:

            return (
                price -
                zone.high
            )

        return 0.0


    # ============================================================
    # POI GEOMETRY
    # ============================================================

    def poi_geometry_valid(
        self,
        zone,
        displacement_idx,
        direction
    ):

        if zone is None:
            return False

        if zone.state != "ACTIVE":
            return False

        # POI cannot come from an event before
        # the displacement.
        if zone.created_idx < displacement_idx:
            return False

        if (
            zone.created_idx
            >
            displacement_idx +
            MAX_POI_AFTER_DISPLACEMENT
        ):

            return False

        drow = self.m15.df.iloc[
            displacement_idx
        ]

        distance = (
            self.zone_distance_to_candle(
                zone,
                drow
            )
        )

        if (
            distance >
            MAX_POI_DISPLACEMENT_GAP
        ):

            return False

        # --------------------------------------------------------
        # Directional relationship
        # --------------------------------------------------------

        close = float(
            drow["close"]
        )

        if direction == "SELL":

            # A bearish displacement should
            # move away downward from its POI.
            if close > zone.high:
                return False

        else:

            # A bullish displacement should
            # move away upward from its POI.
            if close < zone.low:
                return False

        # --------------------------------------------------------
        # Source candle
        # --------------------------------------------------------

        source_idx = int(
            zone.source_idx
        )

        if (
            source_idx < 0
            or
            source_idx >= len(self.m15.df)
        ):

            return False

        if (
            abs(
                source_idx -
                displacement_idx
            )
            >
            OB_LOOKBACK + 2
        ):

            return False

        return True


    # ============================================================
    # FIND VALID POI
    # ============================================================

    def find_poi(
        self,
        displacement_idx,
        direction
    ):

        candidates = []

        for zone in self.m15.zones:

            if zone.direction != direction:
                continue

            if zone.state != "ACTIVE":
                continue

            if not self.poi_geometry_valid(
                zone,
                displacement_idx,
                direction
            ):

                continue

            distance = (
                self.zone_distance_to_candle(
                    zone,
                    self.m15.df.iloc[
                        displacement_idx
                    ]
                )
            )

            candidates.append(
                (
                    distance,
                    -zone.created_idx,
                    zone
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        return candidates[0][2]


    # ============================================================
    # LOCAL M5 LIQUIDITY SWEEP
    # ============================================================

    def local_sweep_after_touch(
        self,
        touch_idx,
        direction,
        end_idx,
        zone=None
    ):

        d = self.m5.df

        start = (
            touch_idx + 1
        )

        end = min(
            len(d),
            touch_idx +
            M5_MAX_CONFIRM_BARS +
            1,
            end_idx + 1
        )

        for i in range(
            start,
            end
        ):

            lookback_start = max(
                0,
                i - M5_LOCAL_LOOKBACK
            )

            lookback_end = i

            if (
                lookback_end
                <=
                lookback_start
            ):

                continue

            row = d.iloc[i]

            previous_high = float(
                d.iloc[
                    lookback_start:
                    lookback_end
                ]["high"].max()
            )

            previous_low = float(
                d.iloc[
                    lookback_start:
                    lookback_end
                ]["low"].min()
            )

            high = float(
                row["high"]
            )

            low = float(
                row["low"]
            )

            close = float(
                row["close"]
            )

            # Sweep must occur at POI.
            if zone is not None:

                touched, _ = (
                    self.zone_intersects_candle(
                        zone,
                        row
                    )
                )

                if not touched:
                    continue

            # SELL = sweep buy-side liquidity
            if direction == "SELL":

                swept = (
                    high > previous_high
                    and
                    close < previous_high
                )

                if swept:
                    return i

            # BUY = sweep sell-side liquidity
            else:

                swept = (
                    low < previous_low
                    and
                    close > previous_low
                )

                if swept:
                    return i

        return None


    # ============================================================
    # M5 STRUCTURE AFTER SWEEP
    # ============================================================

    def m5_structure_after(
        self,
        sweep_idx,
        direction
    ):

        if direction == "BUY":

            names = (
                "CHoCH_UP",
                "BOS_UP",
                "INITIAL_BOS_UP"
            )

            opposite = (
                "CHoCH_DOWN",
                "BOS_DOWN",
                "INITIAL_BOS_DOWN"
            )

        else:

            names = (
                "CHoCH_DOWN",
                "BOS_DOWN",
                "INITIAL_BOS_DOWN"
            )

            opposite = (
                "CHoCH_UP",
                "BOS_UP",
                "INITIAL_BOS_UP"
            )

        end = min(
            len(self.m5.df),
            sweep_idx +
            M5_MAX_CONFIRM_BARS +
            1
        )

        candidates = [
            event
            for event in self.m5.structure
            if (
                event.event in names
                and
                sweep_idx < event.idx < end
            )
        ]

        if not candidates:
            return None

        candidate = min(
            candidates,
            key=lambda x: x.idx
        )

        # Reject if opposite structure appears first.
        for event in self.m5.structure:

            if (
                event.event in opposite
                and
                sweep_idx < event.idx
                < candidate.idx
                and
                event.idx < end
            ):

                return None

        return candidate


    # ============================================================
    # M5 DISPLACEMENT AFTER STRUCTURE
    # ============================================================

    def m5_displacement_after(
        self,
        structure,
        direction
    ):

        start = (
            structure.idx + 1
        )

        end = min(
            len(self.m5.df),
            start + 10
        )

        for i in range(
            start,
            end
        ):

            if self.m5.is_displacement(
                i,
                direction
            ):

                return i

        return None


    # ============================================================
    # M5 RETEST
    # ============================================================

    def retest_after(
        self,
        displacement_idx,
        zone,
        direction
    ):

        d = self.m5.df

        end = min(
            len(d),
            displacement_idx + 16
        )

        for i in range(
            displacement_idx + 1,
            end
        ):

            row = d.iloc[i]

            touched, overlap = (
                self.zone_intersects_candle(
                    zone,
                    row
                )
            )

            if not touched:
                continue

            if (
                self.zone_distance_to_candle(
                    zone,
                    row
                )
                >
                POI_TOUCH_TOLERANCE
            ):

                continue

            if (
                direction == "BUY"
                and
                not bool(row["bull"])
            ):

                continue

            if (
                direction == "SELL"
                and
                not bool(row["bear"])
            ):

                continue

            return i

        return None


    # ============================================================
    # M5 CONFIRMATION
    # ============================================================

    def confirm_m5(
        self,
        zone,
        direction
    ):

        d = self.m5.df

        # POI must still be fresh.
        if (
            zone is None
            or
            zone.state != "ACTIVE"
        ):

            return {
                "state":
                    "WAITING",

                "reason":
                    "POI_NOT_ACTIVE"
            }

        # --------------------------------------------------------
        # PRICE RETURN TO POI
        # --------------------------------------------------------

        start_candidates = list(
            d.index[
                d["time"] >
                zone.created_time
            ]
        )

        if not start_candidates:

            return {
                "state":
                    "WAITING",

                "reason":
                    "NO_M5_AFTER_POI"
            }

        start = start_candidates[0]

        end = min(
            len(d),
            start +
            M5_CONFIRM_LOOKBACK
        )

        touch = None
        touch_overlap = 0.0

        for i in range(
            start,
            end
        ):

            row = d.iloc[i]

            touched, overlap = (
                self.zone_intersects_candle(
                    zone,
                    row
                )
            )

            if not touched:
                continue

            high = float(
                row["high"]
            )

            low = float(
                row["low"]
            )

            close = float(
                row["close"]
            )

            candle_range = (
                high - low
            )

            zone_height = max(
                zone.high -
                zone.low,
                0.01
            )

            max_range = max(
                MAX_TOUCH_RANGE_ABSOLUTE,
                zone_height *
                MAX_TOUCH_RANGE_MULTIPLIER
            )

            if (
                candle_range >
                max_range
            ):

                continue

            close_distance = (
                self.zone_distance_to_price(
                    zone,
                    close
                )
            )

            max_close_distance = max(
                MAX_TOUCH_CLOSE_DISTANCE_ABSOLUTE,
                zone_height *
                MAX_TOUCH_CLOSE_DISTANCE_MULTIPLIER
            )

            if (
                close_distance >
                max_close_distance
            ):

                continue

            touch = i
            touch_overlap = overlap

            break

        if touch is None:

            return {
                "state":
                    "WAITING",

                "reason":
                    "M5_POI_NOT_TOUCHED"
            }

        # --------------------------------------------------------
        # M5 SWEEP
        # --------------------------------------------------------

        sweep = (
            self.local_sweep_after_touch(
                touch,
                direction,
                end - 1,
                zone
            )
        )

        if sweep is None:

            return {
                "state":
                    "POI_TOUCHED",

                "reason":
                    "WAIT_M5_LIQUIDITY_SWEEP",

                "touch":
                    touch,

                "touch_overlap":
                    touch_overlap
            }

        # --------------------------------------------------------
        # M5 STRUCTURE
        # --------------------------------------------------------

        structure = (
            self.m5_structure_after(
                sweep,
                direction
            )
        )

        if structure is None:

            return {
                "state":
                    "LIQUIDITY_SWEPT",

                "reason":
                    "WAIT_M5_STRUCTURE",

                "touch":
                    touch,

                "touch_overlap":
                    touch_overlap,

                "sweep":
                    sweep
            }

        structure_row = (
            d.iloc[
                structure.idx
            ]
        )

        structure_distance = (
            self.zone_distance_to_candle(
                zone,
                structure_row
            )
        )

        if (
            structure_distance >
            MAX_M5_DISTANCE_FROM_POI
        ):

            return {
                "state":
                    "STRUCTURE_SHIFT",

                "reason":
                    "M5_STRUCTURE_TOO_FAR_FROM_POI",

                "touch":
                    touch,

                "touch_overlap":
                    touch_overlap,

                "sweep":
                    sweep,

                "structure":
                    structure.idx
            }

        # --------------------------------------------------------
        # M5 DISPLACEMENT
        # --------------------------------------------------------

        displacement = (
            self.m5_displacement_after(
                structure,
                direction
            )
        )

        if displacement is None:

            return {
                "state":
                    "STRUCTURE_SHIFT",

                "reason":
                    "WAIT_M5_DISPLACEMENT",

                "touch":
                    touch,

                "touch_overlap":
                    touch_overlap,

                "sweep":
                    sweep,

                "structure":
                    structure.idx
            }

        displacement_row = (
            d.iloc[
                displacement
            ]
        )

        displacement_distance = (
            self.zone_distance_to_candle(
                zone,
                displacement_row
            )
        )

        if (
            displacement_distance >
            MAX_M5_DISTANCE_FROM_POI
        ):

            return {
                "state":
                    "DISPLACEMENT",

                "reason":
                    "M5_DISPLACEMENT_TOO_FAR_FROM_POI",

                "touch":
                    touch,

                "touch_overlap":
                    touch_overlap,

                "sweep":
                    sweep,

                "structure":
                    structure.idx,

                "displacement":
                    displacement
            }

        # --------------------------------------------------------
        # M5 RETEST
        # --------------------------------------------------------

        retest = (
            self.retest_after(
                displacement,
                zone,
                direction
            )
        )

        if retest is None:

            return {
                "state":
                    "DISPLACEMENT",

                "reason":
                    "WAIT_M5_RETEST",

                "touch":
                    touch,

                "touch_overlap":
                    touch_overlap,

                "sweep":
                    sweep,

                "structure":
                    structure.idx,

                "displacement":
                    displacement
            }

        # --------------------------------------------------------
        # COMPLETE
        # --------------------------------------------------------

        return {

            "state":
                "ENTRY_READY",

            "reason":
                "COMPLETE_SMC_SEQUENCE",

            "touch":
                touch,

            "touch_overlap":
                touch_overlap,

            "sweep":
                sweep,

            "structure":
                structure.idx,

            "displacement":
                displacement,

            "retest":
                retest
        }


    # ============================================================
    # BUILD SETUP
    # ============================================================

    def build_setup(
        self,
        direction
    ):

        sweeps = self.sweeps(
            direction
        )

        if not sweeps:

            return {

                "direction":
                    direction,

                "state":
                    "WAIT_LIQUIDITY",

                "reason":
                    "NO_LIQUIDITY_SWEEP"
            }

        # Try newest valid causal sequence.
        for sweep in sweeps:

            structure = (
                self.structure_after(
                    sweep,
                    direction
                )
            )

            if structure is None:
                continue

            displacement = (
                self.displacement_after(
                    structure,
                    direction
                )
            )

            if displacement is None:

                return {

                    "direction":
                        direction,

                    "state":
                        "STRUCTURE_SHIFT",

                    "reason":
                        "WAIT_DISPLACEMENT",

                    "liquidity":
                        sweep,

                    "structure":
                        structure
                }

            poi = (
                self.find_poi(
                    displacement,
                    direction
                )
            )

            if poi is None:

                return {

                    "direction":
                        direction,

                    "state":
                        "DISPLACEMENT",

                    "reason":
                        "WAIT_FRESH_VALID_POI",

                    "liquidity":
                        sweep,

                    "structure":
                        structure,

                    "displacement":
                        displacement
                }

            m5 = (
                self.confirm_m5(
                    poi,
                    direction
                )
            )

            return {

                "direction":
                    direction,

                "state":
                    "POI_ACTIVE",

                "reason":
                    m5.get(
                        "reason",
                        "WAITING"
                    ),

                "liquidity":
                    sweep,

                "structure":
                    structure,

                "displacement":
                    displacement,

                "poi":
                    poi,

                "m5":
                    m5
            }

        return {

            "direction":
                direction,

            "state":
                "WAIT_LIQUIDITY",

            "reason":
                "NO_VALID_SMC_SEQUENCE"
        }


    # ============================================================
    # RUN SETUP ENGINE
    # ============================================================

    def run(self):

        return {

            "BUY":
                self.build_setup(
                    "BUY"
                ),

            "SELL":
                self.build_setup(
                    "SELL"
                )
        }


# ================================================================
# MT5 DATA
# ================================================================

def load_mt5_data():

    if not mt5.initialize():

        raise RuntimeError(
            "MT5 initialize gagal: "
            f"{mt5.last_error()}"
        )

    info = mt5.symbol_info(
        SYMBOL
    )

    if info is None:

        mt5.shutdown()

        raise RuntimeError(
            f"Symbol {SYMBOL} tidak ditemukan"
        )

    if not info.visible:

        mt5.symbol_select(
            SYMBOL,
            True
        )

    configs = {

        "H4": (
            mt5.TIMEFRAME_H4,
            H4_BARS
        ),

        "H1": (
            mt5.TIMEFRAME_H1,
            H1_BARS
        ),

        "M15": (
            mt5.TIMEFRAME_M15,
            M15_BARS
        ),

        "M5": (
            mt5.TIMEFRAME_M5,
            M5_BARS
        )
    }

    output = {}

    for name, (
        timeframe,
        bars
    ) in configs.items():

        rates = mt5.copy_rates_from_pos(
            SYMBOL,
            timeframe,
            0,
            bars
        )

        if (
            rates is None
            or
            len(rates) < 20
        ):

            mt5.shutdown()

            raise RuntimeError(
                f"Gagal mengambil {name}: "
                f"{mt5.last_error()}"
            )

        dataframe = pd.DataFrame(
            rates
        )

        dataframe["time"] = (
            pd.to_datetime(
                dataframe["time"],
                unit="s"
            )
        )

        # IMPORTANT:
        # Remove currently forming candle.
        if len(dataframe) > 1:

            dataframe = (
                dataframe
                .iloc[:-1]
                .copy()
            )

        output[name] = (
            dataframe
            .reset_index(drop=True)
        )

    return output


# ================================================================
# MARKET STATUS
# ================================================================

def market_status():

    tick = mt5.symbol_info_tick(
        SYMBOL
    )

    now = pd.Timestamp.now()

    # Saturday / Sunday
    if now.weekday() >= 5:

        return {

            "market":
                "CLOSED",

            "reason":
                "WEEKEND",

            "tick":
                tick,

            "time":
                now
        }

    if tick is None:

        return {

            "market":
                "CLOSED",

            "reason":
                "NO_TICK",

            "tick":
                None,

            "time":
                now
        }

    return {

        "market":
            "OPEN",

        "reason":
            "OK",

        "tick":
            tick,

        "time":
            now
    }


# ================================================================
# FINAL SIGNAL
# ================================================================

def final_signal(
    analysis,
    market
):

    if market["market"] != "OPEN":

        return {

            "signal":
                "WAIT",

            "reason":
                "MARKET_CLOSED"
        }

    # Only complete M5 sequence
    # can produce an entry.
    for direction in (
        "BUY",
        "SELL"
    ):

        setup = (
            analysis.get(
                direction,
                {}
            )
        )

        m5 = (
            setup.get(
                "m5",
                {}
            )
        )

        if (
            m5.get("state")
            ==
            "ENTRY_READY"
        ):

            return {

                "signal":
                    direction,

                "reason":
                    "COMPLETE_SMC_SEQUENCE"
            }

    return {

        "signal":
            "WAIT",

        "reason":
            "NO_COMPLETE_SMC_SEQUENCE"
    }


# ================================================================
# PRINT ENGINE
# ================================================================

def print_engine(
    name,
    engine
):

    stats = engine.stats()

    print(
        f"\n{name} "
        f"{len(engine.df)} candles "
        f"| Bias {engine.bias}"
    )

    print(
        f" Swing H/L     : "
        f"{stats['swing_high']}/"
        f"{stats['swing_low']}"
    )

    print(
        f" BOS UP/DOWN    : "
        f"{stats['bos_up']}/"
        f"{stats['bos_down']}"
    )

    print(
        f" CHoCH UP/DOWN  : "
        f"{stats['choch_up']}/"
        f"{stats['choch_down']}"
    )

    print(
        f" Sweep H/L      : "
        f"{stats['sweep_high']}/"
        f"{stats['sweep_low']}"
    )

    print(
        f" FVG / OB       : "
        f"{stats['fvg']} / "
        f"{stats['ob']}"
    )

    print(
        f" Zones A/M/I    : "
        f"{stats['active_zones']}/"
        f"{stats['mitigated_zones']}/"
        f"{stats['invalidated_zones']}"
    )


# ================================================================
# PRINT SETUP
# ================================================================

def print_setup(
    direction,
    setup,
    m15_engine,
    m5_engine
):

    print(
        f"\n{direction} M15 SETUP"
    )

    print(
        f" STATE          : "
        f"{setup.get('state')}"
    )

    print(
        f" REASON         : "
        f"{setup.get('reason')}"
    )

    # ------------------------------------------------------------
    # M15 LIQUIDITY
    # ------------------------------------------------------------

    liquidity = (
        setup.get("liquidity")
    )

    if liquidity:

        print(
            f" LIQUIDITY      : "
            f"{fmt_time(liquidity.time)} "
            f"@ {fmt_price(liquidity.price)}"
        )

    # ------------------------------------------------------------
    # M15 STRUCTURE
    # ------------------------------------------------------------

    structure = (
        setup.get("structure")
    )

    if structure:

        print(
            f" STRUCTURE      : "
            f"{structure.event} "
            f"@ {fmt_time(structure.time)} "
            f"Price {fmt_price(structure.price)}"
        )

    # ------------------------------------------------------------
    # M15 DISPLACEMENT
    # ------------------------------------------------------------

    displacement = (
        setup.get("displacement")
    )

    if displacement is not None:

        row = (
            m15_engine.df.iloc[
                int(displacement)
            ]
        )

        print(
            f" DISPLACEMENT   : "
            f"{fmt_time(row['time'])} "
            f"| Close "
            f"{fmt_price(row['close'])}"
        )

    # ------------------------------------------------------------
    # POI
    # ------------------------------------------------------------

    zone = (
        setup.get("poi")
    )

    if zone:

        print(
            f" POI            : "
            f"{zone.zone_type} "
            f"{fmt_price(zone.low)} - "
            f"{fmt_price(zone.high)} "
            f"[{zone.state}]"
        )

    # ------------------------------------------------------------
    # M5
    # ------------------------------------------------------------

    m5 = (
        setup.get("m5")
        or {}
    )

    if m5:

        print(
            f" M5 STATE       : "
            f"{m5.get('state')}"
        )

        print(
            f" M5 REASON      : "
            f"{m5.get('reason')}"
        )

        # --------------------------------------------------------
        # M5 POI TOUCH
        # --------------------------------------------------------

        touch = (
            m5.get("touch")
        )

        if touch is not None:

            row = (
                m5_engine.df.iloc[
                    int(touch)
                ]
            )

            print(
                f" M5 POI TOUCH   : "
                f"{fmt_time(row['time'])} "
                f"H {fmt_price(row['high'])} "
                f"L {fmt_price(row['low'])} "
                f"C {fmt_price(row['close'])} "
                f"overlap "
                f"{fmt_price(m5.get('touch_overlap'))}"
            )

        # --------------------------------------------------------
        # M5 SWEEP
        # --------------------------------------------------------

        sweep = (
            m5.get("sweep")
        )

        if sweep is not None:

            row = (
                m5_engine.df.iloc[
                    int(sweep)
                ]
            )

            print(
                f" M5 SWEEP       : "
                f"{fmt_time(row['time'])} "
                f"H {fmt_price(row['high'])} "
                f"L {fmt_price(row['low'])} "
                f"C {fmt_price(row['close'])}"
            )

        # --------------------------------------------------------
        # M5 STRUCTURE
        # --------------------------------------------------------

        structure_idx = (
            m5.get("structure")
        )

        if structure_idx is not None:

            event = next(
                (
                    event
                    for event
                    in m5_engine.structure
                    if event.idx ==
                    int(structure_idx)
                ),
                None
            )

            if event:

                print(
                    f" M5 STRUCTURE   : "
                    f"{event.event} "
                    f"@ {fmt_time(event.time)} "
                    f"Price {fmt_price(event.price)}"
                )

        # --------------------------------------------------------
        # M5 DISPLACEMENT
        # --------------------------------------------------------

        displacement_idx = (
            m5.get("displacement")
        )

        if displacement_idx is not None:

            row = (
                m5_engine.df.iloc[
                    int(displacement_idx)
                ]
            )

            print(
                f" M5 DISPLACE    : "
                f"{fmt_time(row['time'])} "
                f"Close "
                f"{fmt_price(row['close'])}"
            )

        # --------------------------------------------------------
        # M5 RETEST
        # --------------------------------------------------------

        retest = (
            m5.get("retest")
        )

        if retest is not None:

            row = (
                m5_engine.df.iloc[
                    int(retest)
                ]
            )

            print(
                f" M5 RETEST      : "
                f"{fmt_time(row['time'])} "
                f"H {fmt_price(row['high'])} "
                f"L {fmt_price(row['low'])} "
                f"C {fmt_price(row['close'])}"
            )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "=" * 72
    )

    print(
        "AI MT5 TRADER - "
        "SMC ENGINE V4.3.11"
    )

    print(
        "SMART MONEY CONCEPT ONLY "
        "| CLOSED CANDLES ONLY"
    )

    print(
        "=" * 72
    )

    try:

        # --------------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------------

        data = load_mt5_data()

        # --------------------------------------------------------
        # BUILD SMC ENGINES
        # --------------------------------------------------------

        engines = {

            timeframe:
                SMCEngine(
                    timeframe,
                    data[timeframe]
                ).run()

            for timeframe
            in (
                "H4",
                "H1",
                "M15",
                "M5"
            )
        }

        # --------------------------------------------------------
        # PRINT STATISTICS
        # --------------------------------------------------------

        for timeframe in (
            "H4",
            "H1",
            "M15",
            "M5"
        ):

            print_engine(
                timeframe,
                engines[timeframe]
            )

        # --------------------------------------------------------
        # SETUP ENGINE
        # --------------------------------------------------------

        setup_engine = (
            SMCSetupEngine(
                engines["H4"],
                engines["H1"],
                engines["M15"],
                engines["M5"]
            )
        )

        analysis = (
            setup_engine.run()
        )

        # --------------------------------------------------------
        # PRINT SETUPS
        # --------------------------------------------------------

        print_setup(
            "BUY",
            analysis["BUY"],
            engines["M15"],
            engines["M5"]
        )

        print_setup(
            "SELL",
            analysis["SELL"],
            engines["M15"],
            engines["M5"]
        )

        # --------------------------------------------------------
        # MARKET
        # --------------------------------------------------------

        market = (
            market_status()
        )

        print(
            "\nMARKET"
        )

        print(
            f" STATUS         : "
            f"{market['market']} "
            f"({market['reason']})"
        )

        tick = market.get(
            "tick"
        )

        if tick:

            print(
                f" BID/ASK        : "
                f"{tick.bid:.2f} / "
                f"{tick.ask:.2f}"
            )

        print(
            f" TIME           : "
            f"{fmt_time(market['time'])}"
        )

        # --------------------------------------------------------
        # FINAL SIGNAL
        # --------------------------------------------------------

        signal = (
            final_signal(
                analysis,
                market
            )
        )

        print(
            f"\nFINAL SIGNAL    : "
            f"{signal['signal']}"
        )

        print(
            f"REASON          : "
            f"{signal['reason']}"
        )

        if (
            market["market"]
            !=
            "OPEN"
        ):

            print(
                "EXECUTION       : "
                "WAIT - MARKET CLOSED"
            )

        else:

            print(
                f"EXECUTION       : "
                f"{signal['signal']}"
            )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise

    finally:

        mt5.shutdown()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()
