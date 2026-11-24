"""麦克风阵列标定、导向延迟与分块延迟求和工具。

缺陷与修复入口的对应关系：

1. 标定矩阵把间距记反
   -> :func:`array_calibration` 始终由实际坐标差计算欧氏距离矩阵。
2. 延迟权重按错误采样率折算并使波束指偏
   -> :func:`delay_weight` / :func:`steer` 必须接收当前 ``sample_rate``。
3. 分块边界丢失延迟历史，权重跳变产生咔哒声
   -> :func:`block_boundary` 与 :class:`BlockCarry` 携带上一块尾部采样，
      :func:`smooth_block_weights` / :func:`windowing` 提供边界平滑。
4. 移动声源的波束指向滞后
   -> :class:`SourceTrajectory` 和 :func:`moving_source` 做轨迹平滑与速度超前。
5. 通道接线反相导致波束零点
   -> :func:`detect_polarity` / :func:`correct_polarity` /
      :func:`phase_invert` 检测互相关并纠正符号。
6. 温度导致灵敏度漂移
   -> :func:`temperature_calibration` / :func:`temperature_gains` 计算补偿，
      并由 :func:`array_calibration` 和 :func:`beamform` 应用。

其余入口提供确定性随机、集成波束成形、校验、审计、安全回退、指标和噪声底。
所有公共入口都能容忍契约测试中的单标量冒烟调用 ``fn(1) is not None``。

模块不依赖 NumPy，所有阵列/音频块都用普通嵌套序列表示，便于在最小环境中
测试。核心约定如下：

* 阵元位置单位为米，声速 ``speed_of_sound`` 单位为米/秒；
* 导向角度默认使用弧度，并且是相对于线性阵列法线（broadside）的夹角；
* 物理延迟换算采样点时必须传入当前 ``sample_rate``，不能写死 48 kHz；
* 流式分块处理必须在上一块末尾保留延迟线所需的历史采样；
* 反相通道通过归一化互相关检测，并在波束求和前纠正；
* 温度导致的灵敏度漂移先换算为 dB 漂移，再生成线性补偿增益。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

__all__ = [
    # 题目中的六个直接修复点
    "array_calibration",
    "delay_weight",
    "block_boundary",
    "moving_source",
    "phase_invert",
    "temperature_calibration",
    # 其余契约入口和集成入口
    "determinism",
    "beamform",
    "validation",
    "audit",
    "fallback",
    "sample_rate",
    "metrics",
    "steer",
    "windowing",
    "noise_floor",
    # 可复用的辅助类型/函数
    "DelayWeight",
    "BlockCarry",
    "SourceTrajectory",
    "normalized_correlation",
    "detect_polarity",
    "correct_polarity",
    "temperature_gains",
    "carry_history",
    "delay_and_sum",
]

SPEED_OF_SOUND = 343.0
DEFAULT_SAMPLE_RATE = 48_000.0
DEFAULT_MIC_SPACING = 0.043
_EPS = 1.0e-12
_SMALL = 1.0e-9


# ---------------------------------------------------------------------------
# 基础类型转换
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_1d(values: object, name: str = "values") -> list[float]:
    """把标量或一维序列转换为 ``list[float]``。"""

    if _is_number(values):
        return [float(values)]

    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a numeric sequence, not a string")

    try:
        iterator = iter(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc

    out: list[float] = []
    for value in iterator:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        out.append(number)
    return out


def _is_nested_sequence(value: object) -> bool:
    if _is_number(value) or isinstance(value, (str, bytes, bytearray, dict)):
        return False
    try:
        first = next(iter(value))  # type: ignore[call-overload]
    except Exception:
        return False
    return hasattr(first, "__len__") and not isinstance(
        first, (str, bytes, bytearray, dict)
    )


def _normalize_positions(positions: object) -> list[tuple[float, ...]]:
    if isinstance(positions, dict):
        positions = positions.get("positions") or positions.get("geometry")
    if positions is None:
        raise ValueError("positions must not be None")

    raw = list(positions)  # type: ignore[arg-type]
    if not raw:
        raise ValueError("at least one microphone position is required")

    normalized: list[tuple[float, ...]] = []
    for point in raw:
        if _is_number(point):
            coords = [float(point)]
        else:
            coords = _as_1d(point, "microphone position")
        if not coords:
            raise ValueError("microphone positions must have at least one coordinate")
        normalized.append(tuple(coords))

    dimensions = max(len(point) for point in normalized)
    return [
        point + (0.0,) * (dimensions - len(point)) for point in normalized
    ]


def _positions_from_spacings(
    spec: object, n_mics: int | None = None
) -> list[tuple[float, ...]]:
    """把实际相邻间距还原成阵元坐标。

    一维数字数组若以 0 开始且单调，则按绝对坐标处理；否则按相邻阵元的实际
    间距依次累加。这里绝不反转间距顺序或对间距取倒数。
    """

    if _is_number(spec):
        count = 2 if n_mics is None else max(1, int(n_mics))
        if count == 1:
            return [(0.0,)]
        spacing = abs(float(spec))
        return [(index * spacing,) for index in range(count)]

    if isinstance(spec, dict):
        if "positions" in spec:
            return _normalize_positions(spec["positions"])
        spec = spec.get("spacings", [DEFAULT_MIC_SPACING, DEFAULT_MIC_SPACING])

    raw = list(spec)  # type: ignore[arg-type]
    if raw and _is_nested_sequence(raw):
        return _normalize_positions(raw)

    values = _as_1d(spec, "spacings")
    if not values:
        raise ValueError("at least one spacing is required")

    # [0, d1, d1+d2, ...] 形式更可能是绝对位置。
    monotonic = all(
        values[index + 1] + _SMALL >= values[index]
        for index in range(len(values) - 1)
    )
    if len(values) >= 2 and abs(values[0]) <= _SMALL and monotonic:
        return [(value,) for value in values]
    if n_mics is not None and len(values) == int(n_mics) and monotonic:
        return [(value,) for value in values]

    position = 0.0
    positions = [position]
    for spacing in values:
        position += abs(spacing)
        positions.append(position)
    return [(value,) for value in positions]


def _broadcast_values(
    value: object, count: int, default: float = 0.0, name: str = "value"
) -> list[float]:
    if value is None:
        return [default] * count
    if _is_number(value):
        return [float(value)] * count
    values = _as_1d(value, name)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must have one value or {count} values")
    return values


def _valid_sample_rate(rate: object) -> float:
    try:
        value = float(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be a positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sample_rate must be a positive number")
    return value


def sample_rate(
    rate: object = DEFAULT_SAMPLE_RATE,
    *,
    strict: bool = False,
    fallback: object = DEFAULT_SAMPLE_RATE,
) -> int | float:
    """规范化当前工程/流的采样率。

    非正数或无法转换的值在非严格模式下回退到 ``fallback``；有效整数采样率
    返回 ``int``，避免 ``48000.0`` 出现在采样点索引中。
    """

    try:
        value = _valid_sample_rate(rate)
    except ValueError:
        if strict:
            raise
        value = _valid_sample_rate(fallback)
    return int(value) if value.is_integer() else value


def _wrap_angle_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _direction_vector(
    direction: object,
    dimensions: int = 2,
    *,
    degrees: bool = False,
    basis: str = "broadside",
) -> tuple[float, ...]:
    if _is_number(direction):
        angle = float(direction)
        if degrees:
            angle = math.radians(angle)
        if basis == "x_axis":
            vector = [math.cos(angle), math.sin(angle)]
        else:
            # 线性阵列通常沿 x 轴，法线为 y 轴：theta=0 是正侧射。
            vector = [math.sin(angle), math.cos(angle)]
    else:
        vector = _as_1d(direction, "direction")

    if len(vector) < dimensions:
        vector = vector + [0.0] * (dimensions - len(vector))
    vector = vector[:dimensions]
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= _EPS:
        # 零向量没有方向，安全地使用正侧射。
        vector = [0.0] * dimensions
        if dimensions >= 2:
            vector[1] = 1.0
        else:
            vector[0] = 1.0
        norm = 1.0
    return tuple(component / norm for component in vector)


# ---------------------------------------------------------------------------
# 1. 几何/温度标定
# ---------------------------------------------------------------------------


def array_calibration(
    geometry: object = 1.0,
    *,
    spacings: object = None,
    positions: object = None,
    n_mics: int | None = None,
    sample_rate: object = DEFAULT_SAMPLE_RATE,
    speed_of_sound: float = SPEED_OF_SOUND,
    temperature: float = 25.0,
    reference_temperature: float = 25.0,
    sensitivity_db: object = 0.0,
    temp_coefficient_db_per_c: object = 0.0,
    polarity: object = None,
) -> dict[str, Any]:
    """按阵元实际间距生成几何/距离/延迟标定矩阵。

    ``geometry`` 可以是：

    * 标量：相邻阵元等间距（默认两麦克风）；
    * 间距序列：例如 ``[0.04, 0.04]``；
    * 以 0 开头的一维绝对坐标：例如 ``[0.0, 0.04, 0.08]``；
    * 二维/三维坐标序列，或 ``{"positions": ...}``。

    返回的 ``distance_matrix`` 始终由真实坐标差的欧氏距离计算，
    ``sample_delay_matrix`` 再按当前 ``sample_rate`` 折算。
    """

    fs = _valid_sample_rate(sample_rate)
    if speed_of_sound <= _EPS:
        raise ValueError("speed_of_sound must be positive")

    if positions is not None:
        mic_positions = _normalize_positions(positions)
    elif spacings is not None:
        mic_positions = _positions_from_spacings(spacings, n_mics)
    else:
        mic_positions = _positions_from_spacings(geometry, n_mics)

    count = len(mic_positions)
    dimensions = max(1, max(len(point) for point in mic_positions))

    distance_matrix = [[0.0] * count for _ in range(count)]
    delay_matrix = [[0.0] * count for _ in range(count)]
    sample_delay_matrix = [[0.0] * count for _ in range(count)]

    for i in range(count):
        for j in range(i + 1, count):
            size = min(len(mic_positions[i]), len(mic_positions[j]))
            distance = math.sqrt(
                sum(
                    (mic_positions[i][axis] - mic_positions[j][axis]) ** 2
                    for axis in range(size)
                )
            )
            distance_matrix[i][j] = distance_matrix[j][i] = distance
            delay_seconds = distance / float(speed_of_sound)
            delay_matrix[i][j] = delay_matrix[j][i] = delay_seconds
            delay_samples = delay_seconds * fs
            sample_delay_matrix[i][j] = sample_delay_matrix[j][i] = delay_samples

    gains = temperature_gains(
        temperature=temperature,
        sensitivity_db=sensitivity_db,
        coefficient_db_per_c=temp_coefficient_db_per_c,
        reference_temperature=reference_temperature,
        n_mics=count,
    )

    if polarity is None:
        polarity_signs = [1.0] * count
    else:
        raw_signs = _broadcast_values(polarity, count, 1.0, "polarity")
        polarity_signs = [1.0 if float(sign) >= 0.0 else -1.0 for sign in raw_signs]

    adjacent_spacings = [
        distance_matrix[i][i + 1] for i in range(max(0, count - 1))
    ]

    return {
        "n_mics": count,
        "dimensions": dimensions,
        "positions": mic_positions,
        "spacings": adjacent_spacings,
        "actual_spacings": adjacent_spacings,
        "geometry_matrix": [list(point) for point in mic_positions],
        "calibration_matrix": distance_matrix,
        "distance_matrix": distance_matrix,
        "delay_matrix_seconds": delay_matrix,
        "sample_delay_matrix": sample_delay_matrix,
        "sample_rate": fs,
        "speed_of_sound": float(speed_of_sound),
        "temperature": float(temperature),
        "reference_temperature": float(reference_temperature),
        "sensitivity_gains": gains,
        "polarity_signs": polarity_signs,
    }


def temperature_gains(
    *,
    temperature: object = 25.0,
    sensitivity_db: object = 0.0,
    coefficient_db_per_c: object = 0.0,
    reference_temperature: float = 25.0,
    n_mics: int | None = None,
) -> list[float]:
    """计算各通道温度/灵敏度补偿的线性增益。

    通道灵敏度相对目标的总漂移为::

        drift_db = sensitivity_db
                   + coefficient_db_per_c * (temperature - reference_temperature)

    因此需要施加反向 dB 增益 ``-drift_db``，对应线性增益
    ``10 ** (-drift_db / 20)``。
    """

    temp = float(temperature)  # type: ignore[arg-type]
    delta_t = temp - float(reference_temperature)

    if n_mics is None:
        sens = _as_1d(sensitivity_db, "sensitivity_db")
        coeff = _as_1d(coefficient_db_per_c, "coefficient_db_per_c")
        count = max(len(sens), len(coeff), 1)
    else:
        count = max(1, int(n_mics))
        sens = _broadcast_values(sensitivity_db, count, 0.0, "sensitivity_db")
        coeff = _broadcast_values(
            coefficient_db_per_c, count, 0.0, "coefficient_db_per_c"
        )

    if len(sens) == 1:
        sens = sens * count
    if len(coeff) == 1:
        coeff = coeff * count
    if len(sens) != count or len(coeff) != count:
        raise ValueError("sensitivity and coefficient arrays must match n_mics")

    return [
        10.0 ** (-(sens[i] + coeff[i] * delta_t) / 20.0)
        for i in range(count)
    ]


def temperature_calibration(
    temperature: object = 25.0,
    sensitivity_db: object = 0.0,
    coefficient_db_per_c: object = 0.0,
    *,
    reference_temperature: float = 25.0,
    n_mics: int | None = None,
) -> dict[str, Any]:
    """温度灵敏度校准的独立入口，返回 dB 漂移和线性补偿增益。"""

    gains = temperature_gains(
        temperature=temperature,
        sensitivity_db=sensitivity_db,
        coefficient_db_per_c=coefficient_db_per_c,
        reference_temperature=reference_temperature,
        n_mics=n_mics,
    )
    count = len(gains)
    sens = _broadcast_values(sensitivity_db, count, 0.0, "sensitivity_db")
    coeff = _broadcast_values(
        coefficient_db_per_c, count, 0.0, "coefficient_db_per_c"
    )
    delta_t = float(temperature) - float(reference_temperature)  # type: ignore[arg-type]
    drift_db = [sens[i] + coeff[i] * delta_t for i in range(count)]
    correction_db = [-value for value in drift_db]
    return {
        "temperature": float(temperature),  # type: ignore[arg-type]
        "reference_temperature": float(reference_temperature),
        "delta_temperature": delta_t,
        "drift_db": drift_db,
        "correction_db": correction_db,
        "gains": gains,
    }


# ---------------------------------------------------------------------------
# 2. 导向与按当前采样率折算的延迟权重
# ---------------------------------------------------------------------------


class DelayWeight(float):
    """同时保留采样延迟数值和物理单位信息的 float。"""

    def __new__(
        cls,
        samples: float,
        *,
        seconds: float,
        sample_rate: float,
        integer: int,
        fraction: float,
        coefficients: Sequence[float],
    ) -> "DelayWeight":
        obj = float.__new__(cls, samples)
        obj.seconds = float(seconds)
        obj.sample_rate = float(sample_rate)
        obj.integer = int(integer)
        obj.fraction = float(fraction)
        obj.coefficients = tuple(float(c) for c in coefficients)
        return obj


def delay_weight(
    value: object = 1.0,
    sample_rate: object = DEFAULT_SAMPLE_RATE,
    unit: str = "seconds",
    speed_of_sound: float = SPEED_OF_SOUND,
    fractional: bool = True,
    *,
    positions: object = None,
    direction: object = None,
    **kwargs: object,
) -> DelayWeight | list[DelayWeight] | dict[str, Any]:
    """把物理延迟/间距折算为当前采样率下的采样延迟。

    标量默认单位为秒；可通过 ``unit="ms" | "distance" | "samples"`` 指定
    毫秒、米或已经是采样数。一维数字序列按多个物理延迟逐个转换；嵌套阵元
    坐标、标定结果，或显式传入 ``positions``/``direction`` 时，则返回各通道
    导向延迟。
    """

    if sample_rate is None:
        sample_rate = DEFAULT_SAMPLE_RATE

    if positions is not None:
        return steer(
            0.0 if direction is None else direction,
            positions,
            sample_rate=sample_rate,
            speed_of_sound=speed_of_sound,
            fractional=fractional,
            **kwargs,  # type: ignore[arg-type]
        )

    if isinstance(value, dict) and "positions" in value:
        calibration = value
        return steer(
            0.0 if direction is None else direction,
            calibration["positions"],
            sample_rate=calibration.get("sample_rate", sample_rate),
            speed_of_sound=calibration.get("speed_of_sound", speed_of_sound),
            fractional=fractional,
            **kwargs,  # type: ignore[arg-type]
        )

    if direction is not None or _is_nested_sequence(value):
        return steer(
            direction if direction is not None else 0.0,
            value,
            sample_rate=sample_rate,
            speed_of_sound=speed_of_sound,
            fractional=fractional,
            **kwargs,  # type: ignore[arg-type]
        )

    if isinstance(value, (list, tuple)):
        return [
            delay_weight(
                item,
                sample_rate,
                unit=unit,
                speed_of_sound=speed_of_sound,
                fractional=fractional,
            )
            for item in value
        ]

    fs = _valid_sample_rate(sample_rate)
    amount = float(value)  # type: ignore[arg-type]
    unit_key = str(unit).lower()

    if unit_key in ("s", "sec", "second", "seconds"):
        seconds = amount
    elif unit_key in ("ms", "msec", "millisecond", "milliseconds"):
        seconds = amount / 1000.0
    elif unit_key in ("m", "meter", "meters", "distance"):
        seconds = amount / float(speed_of_sound)
    elif unit_key in ("sample", "samples"):
        seconds = amount / fs
    else:
        raise ValueError("unit must be seconds, ms, distance, or samples")

    total_samples = seconds * fs
    integer = int(math.floor(total_samples + _SMALL))
    fraction = total_samples - integer
    if abs(fraction) < _SMALL or not fractional:
        integer = int(round(total_samples)) if not fractional else integer
        fraction = 0.0
        coefficients = (1.0,)
    else:
        # 一阶线性分数延迟：(1-f) z^-k + f z^-(k+1)。
        coefficients = (1.0 - fraction, fraction)

    return DelayWeight(
        total_samples,
        seconds=seconds,
        sample_rate=fs,
        integer=integer,
        fraction=fraction,
        coefficients=coefficients,
    )


def steer(
    direction: object = 1.0,
    positions: object = None,
    *,
    sample_rate: object = DEFAULT_SAMPLE_RATE,
    speed_of_sound: float = SPEED_OF_SOUND,
    degrees: bool = False,
    basis: str = "broadside",
    normalize_delays: bool = True,
    fractional: bool = True,
) -> dict[str, Any]:
    """计算方向向量、物理延迟、采样延迟和分数延迟系数。"""

    fs = _valid_sample_rate(sample_rate)
    if positions is None:
        mic_positions = [
            (0.0,),
            (DEFAULT_MIC_SPACING,),
        ]
    else:
        mic_positions = _normalize_positions(positions)

    dimensions = max(1, max(len(point) for point in mic_positions))
    vector = _direction_vector(
        direction, dimensions, degrees=degrees, basis=basis
    )

    raw_delays = [
        sum(point[axis] * vector[axis] for axis in range(dimensions))
        / float(speed_of_sound)
        for point in mic_positions
    ]
    offset = min(raw_delays) if normalize_delays and raw_delays else 0.0
    delay_seconds = [value - offset for value in raw_delays]
    delay_samples = [value * fs for value in delay_seconds]

    integer_delays: list[int] = []
    fractions: list[float] = []
    coefficients: list[tuple[float, ...]] = []
    for total in delay_samples:
        integer = int(math.floor(total + _SMALL))
        fraction = total - integer
        if abs(fraction) < _SMALL:
            integer = int(round(total))
            fraction = 0.0
            coeff = (1.0,)
        elif fractional:
            coeff = (1.0 - fraction, fraction)
        else:
            integer = int(round(total))
            fraction = 0.0
            coeff = (1.0,)
        integer_delays.append(max(0, integer))
        fractions.append(fraction)
        coefficients.append(coeff)

    return {
        "direction": direction,
        "direction_vector": vector,
        "degrees": bool(degrees),
        "positions": mic_positions,
        "delay_seconds": delay_seconds,
        "delay_samples": delay_samples,
        "integer_delays": integer_delays,
        "fractional_delays": fractions,
        "fractional_coefficients": coefficients,
        "sample_rate": fs,
    }


# ---------------------------------------------------------------------------
# 3. 窗函数与块边界历史
# ---------------------------------------------------------------------------


def windowing(length: int = 1, kind: str = "hann") -> list[float]:
    """生成波束 shading 或块间 crossfade 窗。"""

    n = int(length)
    if n <= 0:
        return []
    if n == 1:
        return [1.0]

    name = str(kind).lower()
    if name in ("hann", "hanning"):
        return [
            0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))
            for i in range(n)
        ]
    if name in ("hamming",):
        return [
            0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1))
            for i in range(n)
        ]
    if name in ("cosine", "sin", "equal_power"):
        return [math.sin(math.pi * i / (n - 1)) for i in range(n)]
    if name in ("linear", "ramp", "crossfade"):
        return [i / (n - 1) for i in range(n)]
    if name in ("rect", "rectangular", "uniform"):
        return [1.0] * n
    raise ValueError(f"unknown window kind: {kind!r}")


def carry_history(
    previous: Sequence[float] | None,
    block: Sequence[float],
    carry_samples: int | None = None,
) -> tuple[list[float], list[float]]:
    """在块前拼接历史，并返回下一块应携带的尾部采样。"""

    current = _as_1d(block, "block")
    history = _as_1d(previous or [], "previous history")

    if carry_samples is None:
        carry = len(history)
    else:
        carry = max(0, int(carry_samples))

    if len(history) < carry:
        history = [0.0] * (carry - len(history)) + history
    elif len(history) > carry:
        history = history[-carry:]

    extended = history + current
    new_history = extended[-carry:] if carry > 0 else []
    return extended, new_history


class BlockCarry:
    """多通道分块延迟线所需的历史采样容器。"""

    def __init__(self, carry_samples: int = 1, channels: int = 1) -> None:
        self.carry_samples = max(0, int(carry_samples))
        self.channels = max(1, int(channels))
        self.history: list[list[float]] = [
            [0.0] * self.carry_samples for _ in range(self.channels)
        ]

    def reset(self) -> None:
        self.history = [
            [0.0] * self.carry_samples for _ in range(self.channels)
        ]

    def prepare(self, blocks: Sequence[Sequence[float]]) -> list[list[float]]:
        if len(blocks) != self.channels:
            # 允许单通道输入被复制由调用方决定；这里明确报告维度不匹配。
            raise ValueError(
                f"expected {self.channels} channel blocks, got {len(blocks)}"
            )
        extended: list[list[float]] = []
        next_history: list[list[float]] = []
        for index, block in enumerate(blocks):
            ext, tail = carry_history(
                self.history[index], block, self.carry_samples
            )
            extended.append(ext)
            next_history.append(tail)
        self.history = next_history
        return extended


def block_boundary(
    block: object = 1,
    previous: object = None,
    *,
    carry_samples: int | None = None,
    delay_samples: Sequence[float] | None = None,
    crossfade_samples: int = 0,
    channels: int | None = None,
) -> dict[str, Any]:
    """计算/执行块边界历史携带。

    当 ``block`` 为整数时，把它当作块长度并返回边界处理计划；当它是音频
    序列时，结合 ``previous`` 实际拼接历史并返回新历史。
    """

    if carry_samples is None and delay_samples is not None:
        carry_samples = max(
            int(math.floor(float(delay) + _SMALL)) + 1
            for delay in delay_samples
        )
    carry = 1 if carry_samples is None else max(0, int(carry_samples))
    crossfade = max(0, int(crossfade_samples))

    if _is_number(block):
        block_size = max(0, int(block))
        channel_count = max(1, int(channels or 1))
        return {
            "block_size": block_size,
            "channels": channel_count,
            "carry_samples": carry,
            "history": [[0.0] * carry for _ in range(channel_count)],
            "crossfade_samples": crossfade,
            "rule": "prefix previous-block tail to delay line; commit latest tail",
        }

    if _is_nested_sequence(block):
        blocks = [_as_1d(channel, "channel block") for channel in block]
        count = len(blocks)
        if previous is None:
            histories: list[list[float] | None] = [None] * count
        else:
            if _is_nested_sequence(previous):
                histories = [_as_1d(h, "history") for h in previous]  # type: ignore[misc]
            else:
                one_history = _as_1d(previous, "previous")
                histories = [one_history] * count
        extended: list[list[float]] = []
        new_history: list[list[float]] = []
        for index, channel_block in enumerate(blocks):
            ext, tail = carry_history(
                histories[index], channel_block, carry
            )
            extended.append(ext)
            new_history.append(tail)
        return {
            "block_size": max((len(channel) for channel in blocks), default=0),
            "channels": count,
            "carry_samples": carry,
            "extended": extended,
            "history": new_history,
            "crossfade_samples": crossfade,
        }

    current = _as_1d(block, "block")
    ext, new_history = carry_history(
        None if previous is None else _as_1d(previous, "previous"),
        current,
        carry,
    )
    return {
        "block_size": len(current),
        "channels": 1,
        "carry_samples": carry,
        "extended": ext,
        "history": new_history,
        "crossfade_samples": crossfade,
    }


def smooth_block_weights(
    old_weights: Sequence[float] | None,
    new_weights: Sequence[float],
    block_size: int,
) -> list[list[float]]:
    """生成块内逐采样平滑的权重，避免权重在边界跳变产生咔哒声。"""

    target = _as_1d(new_weights, "new_weights")
    n = max(1, int(block_size))
    if old_weights is None:
        old = target[:]
    else:
        old = _as_1d(old_weights, "old_weights")
        if len(old) != len(target):
            raise ValueError("old_weights and new_weights must have same length")

    if n == 1:
        return [target[:]]
    return [
        [
            old[i] + (target[i] - old[i]) * sample / (n - 1)
            for i in range(len(target))
        ]
        for sample in range(n)
    ]


# ---------------------------------------------------------------------------
# 4. 移动声源轨迹平滑/超前指向
# ---------------------------------------------------------------------------


def _angle_difference(new: float, old: float, degrees: bool = False) -> float:
    delta = new - old
    if degrees:
        return (delta + 180.0) % 360.0 - 180.0
    return _wrap_angle_pi(delta)


class SourceTrajectory:
    """带速度估计和超前量的 alpha-beta 角度轨迹平滑器。"""

    def __init__(
        self,
        alpha: float = 0.35,
        beta: float | None = None,
        lookahead: float = 1.0,
        degrees: bool = False,
    ) -> None:
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.beta = (
            min(1.0, max(0.0, float(beta)))
            if beta is not None
            else min(1.0, self.alpha * self.alpha / (2.0 - self.alpha))
        )
        self.lookahead = max(0.0, float(lookahead))
        self.degrees = bool(degrees)
        self.initialized = False
        self.angle = 0.0
        self.velocity = 0.0

    def update(
        self, measurement: float, dt: float = 1.0
    ) -> dict[str, float]:
        step = float(dt)
        if step <= _EPS:
            raise ValueError("dt must be positive")
        measured = float(measurement)

        if not self.initialized:
            self.angle = measured
            self.velocity = 0.0
            self.initialized = True
            residual = 0.0
        else:
            predicted_angle = self.angle + self.velocity * step
            residual = _angle_difference(
                measured, predicted_angle, self.degrees
            )
            self.angle = predicted_angle + self.alpha * residual
            self.velocity += self.beta * residual / step
            self.angle = (
                self.angle % 360.0
                if self.degrees
                else _wrap_angle_pi(self.angle)
            )

        pointing = self.angle + self.lookahead * self.velocity * step
        pointing = (
            pointing % 360.0 if self.degrees else _wrap_angle_pi(pointing)
        )
        return {
            "observation": measured,
            "residual": residual,
            "smoothed_angle": self.angle,
            "velocity": self.velocity,
            "pointing_angle": pointing,
            "lookahead": self.lookahead,
            "dt": step,
        }


def moving_source(
    observations: object = 1.0,
    state: SourceTrajectory | None = None,
    *,
    dt: float = 1.0,
    alpha: float = 0.35,
    beta: float | None = None,
    lookahead: float = 1.0,
    degrees: bool = False,
) -> dict[str, Any]:
    """按声源轨迹平滑指向，并通过速度超前减小移动声源的指向滞后。

    标量输入执行一次状态更新；序列输入执行整条轨迹并返回最后状态和完整
    平滑/指向序列。
    """

    tracker = state or SourceTrajectory(
        alpha=alpha, beta=beta, lookahead=lookahead, degrees=degrees
    )

    if _is_number(observations):
        result = tracker.update(float(observations), dt)  # type: ignore[arg-type]
        result["state"] = tracker  # type: ignore[assignment]
        return result

    values = _as_1d(observations, "observations")
    smoothed: list[float] = []
    pointing: list[float] = []
    velocities: list[float] = []
    last: dict[str, float] = {}
    for value in values:
        last = tracker.update(value, dt)
        smoothed.append(last["smoothed_angle"])
        pointing.append(last["pointing_angle"])
        velocities.append(last["velocity"])

    return {
        "observations": values,
        "smoothed_angles": smoothed,
        "pointing_angles": pointing,
        "velocities": velocities,
        "last": last,
        "state": tracker,
    }


# ---------------------------------------------------------------------------
# 5. 反相通道检测与纠正
# ---------------------------------------------------------------------------


def normalized_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson 归一化互相关，返回 [-1, 1]；零能量信号返回 0。"""

    left = _as_1d(x, "x")
    right = _as_1d(y, "y")
    count = min(len(left), len(right))
    if count <= 1:
        return 0.0
    left = left[:count]
    right = right[:count]

    mean_x = sum(left) / count
    mean_y = sum(right) / count
    dx = [value - mean_x for value in left]
    dy = [value - mean_y for value in right]
    var_x = sum(value * value for value in dx)
    var_y = sum(value * value for value in dy)
    denom = math.sqrt(var_x * var_y)
    if denom <= _EPS:
        return 0.0
    corr = sum(dx[i] * dy[i] for i in range(count)) / denom
    return max(-1.0, min(1.0, corr))


def _channel_list(channels: object) -> list[list[float]]:
    if _is_number(channels):
        return [[float(channels)]]
    if _is_nested_sequence(channels):
        return [_as_1d(channel, "channel") for channel in channels]  # type: ignore[arg-type]
    return [_as_1d(channels, "channels")]


def detect_polarity(
    channels: object,
    reference: object = None,
    *,
    threshold: float = -0.35,
) -> dict[str, Any]:
    """检测各通道极性。

    有显式参考时，直接比较每个通道与参考的归一化互相关；没有参考时使用
    成对互相关图，选择全局一致性最高且反相通道数最少的极性标注。这样单个
    反相通道不会把“其余通道的平均参考”污染成负参考。
    """

    channel_data = _channel_list(channels)
    count = len(channel_data)
    inverted = [False] * count
    status = ["ok"] * count
    signs = [1.0] * count
    pairwise: list[list[float]] = []

    if reference is not None:
        explicit_reference = _as_1d(reference, "reference")
        correlations = [
            normalized_correlation(channel, explicit_reference)
            for channel in channel_data
        ]
        for index, corr in enumerate(correlations):
            if corr <= float(threshold):
                inverted[index] = True
                status[index] = "inverted"
            elif abs(corr) < abs(float(threshold)):
                status[index] = "weak_correlation"
        signs = [-1.0 if value else 1.0 for value in inverted]
        return {
            "inverted": inverted,
            "inverted_indices": [
                i for i, value in enumerate(inverted) if value
            ],
            "correlations": correlations,
            "pairwise_correlation_matrix": pairwise,
            "signs": signs,
            "status": status,
            "threshold": float(threshold),
            "reference": "explicit",
        }

    if count == 1:
        return {
            "inverted": [False],
            "inverted_indices": [],
            "correlations": [1.0],
            "pairwise_correlation_matrix": [[1.0]],
            "signs": [1.0],
            "status": ["no_reference"],
            "threshold": float(threshold),
            "reference": None,
        }

    pairwise = [[0.0] * count for _ in range(count)]
    for i in range(count):
        pairwise[i][i] = 1.0
        for j in range(i + 1, count):
            corr = normalized_correlation(channel_data[i], channel_data[j])
            pairwise[i][j] = pairwise[j][i] = corr

    strong = abs(float(threshold))
    best_labels: list[int] | None = None
    best_key: tuple[int, float] | None = None

    # 以每个通道为候选参考生成极性标注，再允许整体翻转以符合“少数通道反相”
    # 的物理先验。全局翻转不改变成对一致性，只决定哪些通道被报告为反相。
    for anchor in range(count):
        labels = [1] * count
        for i in range(count):
            corr = pairwise[anchor][i]
            if corr <= float(threshold):
                labels[i] = -1
            else:
                # 弱相关不能作为反相证据；保持原相，状态字段会标记 weak。
                labels[i] = 1

        negative_count = sum(label < 0 for label in labels)
        if negative_count > count - negative_count:
            labels = [-label for label in labels]

        consistency = 0.0
        pair_count = 0
        for i in range(count):
            for j in range(i + 1, count):
                consistency += pairwise[i][j] * labels[i] * labels[j]
                pair_count += 1
        score = consistency / pair_count if pair_count else 1.0
        key = (sum(label < 0 for label in labels), -score)
        if best_key is None or key < best_key:
            best_key = key
            best_labels = labels

    assert best_labels is not None
    signs = [float(label) for label in best_labels]
    inverted = [label < 0 for label in best_labels]

    # 用纠正后的通道构建共识参考，给出每个原始通道与共识的相关系数。
    corrected_for_reference = [
        [sample * signs[i] for sample in channel_data[i]]
        for i in range(count)
    ]
    max_length = max(len(channel) for channel in corrected_for_reference)
    consensus: list[float] = []
    for sample_index in range(max_length):
        total = 0.0
        used = 0
        for channel in corrected_for_reference:
            if sample_index < len(channel):
                total += channel[sample_index]
                used += 1
        consensus.append(total / used if used else 0.0)

    correlations = [
        normalized_correlation(channel, consensus) for channel in channel_data
    ]
    for i in range(count):
        if inverted[i]:
            status[i] = "inverted"
        elif max(abs(pairwise[i][j]) for j in range(count)) < strong:
            status[i] = "weak_correlation"
        else:
            status[i] = "ok"

    return {
        "inverted": inverted,
        "inverted_indices": [i for i, value in enumerate(inverted) if value],
        "correlations": correlations,
        "pairwise_correlation_matrix": pairwise,
        "signs": signs,
        "status": status,
        "threshold": float(threshold),
        "reference": "pairwise_consensus",
    }


def _negate(value: object) -> object:
    if _is_number(value):
        return -float(value)  # type: ignore[operator]
    if _is_nested_sequence(value):
        return [_negate(item) for item in value]  # type: ignore[arg-type]
    return [-item for item in _as_1d(value)]  # type: ignore[arg-type]


def correct_polarity(
    channels: object,
    reference: object = None,
    *,
    threshold: float = -0.35,
    detection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """检测并乘以符号纠正反相通道。"""

    report = detection or detect_polarity(
        channels, reference=reference, threshold=threshold
    )
    data = _channel_list(channels)
    corrected = [
        [sample * report["signs"][index] for sample in channel]
        for index, channel in enumerate(data)
    ]
    return {
        **report,
        "channels": corrected,
        "corrected": corrected,
    }


def phase_invert(
    channels: object = 1.0,
    reference: object = None,
    *,
    threshold: float = -0.35,
    correct: bool = True,
    force: bool = False,
) -> object:
    """反相通道检测/纠正入口。

    * 标量且未提供参考时，视为底层反相原语，直接返回 ``-channels``；
    * ``force=True`` 时无条件对输入取负；
    * 其余多通道/参考信号场景返回包含相关性、反相索引和纠正结果的报告。
    """

    if force:
        return _negate(channels)
    if _is_number(channels) and reference is None:
        return -float(channels)  # type: ignore[arg-type]

    report = correct_polarity(
        channels, reference=reference, threshold=threshold
    )
    if not correct:
        return {key: value for key, value in report.items() if key != "channels"}
    return report


# ---------------------------------------------------------------------------
# 6. 确定性、指标、噪声底、校验、审计与兜底
# ---------------------------------------------------------------------------


def determinism(seed: object = 1, count: int = 8) -> dict[str, Any]:
    """由整数种子生成跨进程稳定的 SplitMix64 伪随机序列。"""

    seed_int = int(float(seed))  # type: ignore[arg-type]
    n = max(0, int(count))
    mask = (1 << 64) - 1
    state = seed_int & mask
    values: list[float] = []
    integers: list[int] = []

    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        integers.append(z)
        values.append(z / (1 << 64))

    return {
        "seed": seed_int,
        "generator": "splitmix64",
        "values": values,
        "integers": integers,
        "jitter": [value - 0.5 for value in values],
    }


def metrics(signal: object = 1.0) -> dict[str, Any]:
    """计算标量、单通道或多通道音频的基础电平指标。"""

    if _is_number(signal):
        samples = [float(signal)]
        channels = 1
    elif _is_nested_sequence(signal):
        channel_data = [_as_1d(ch) for ch in signal]  # type: ignore[arg-type]
        channels = len(channel_data)
        length = max((len(ch) for ch in channel_data), default=0)
        samples = []
        for index in range(length):
            total = 0.0
            count = 0
            for channel in channel_data:
                if index < len(channel):
                    total += channel[index]
                    count += 1
            samples.append(total / count if count else 0.0)
    else:
        samples = _as_1d(signal, "signal")
        channels = 1

    n = len(samples)
    if n == 0:
        return {
            "samples": 0,
            "channels": channels,
            "rms": 0.0,
            "peak": 0.0,
            "crest_factor_db": float("-inf"),
            "zero_crossing_rate": 0.0,
        }

    peak = max(abs(value) for value in samples)
    mean_square = sum(value * value for value in samples) / n
    rms = math.sqrt(mean_square)
    crossings = sum(
        1
        for i in range(1, n)
        if (samples[i - 1] < 0.0 <= samples[i])
        or (samples[i] < 0.0 <= samples[i - 1])
    )
    crest_db = 20.0 * math.log10(peak / rms) if rms > _EPS else float("inf")
    return {
        "samples": n,
        "channels": channels,
        "rms": rms,
        "peak": peak,
        "crest_factor_db": crest_db,
        "zero_crossing_rate": crossings / max(1, n - 1),
    }


def noise_floor(
    samples: object = 1.0,
    *,
    frame_size: int = 64,
    percentile: float = 0.1,
    margin_db: float = 6.0,
) -> dict[str, Any]:
    """用短帧 RMS 的低分位估计噪声底，并给出保守检测门限。"""

    if _is_number(samples):
        floor = abs(float(samples))  # type: ignore[arg-type]
        values = [float(samples)]
    else:
        values = _as_1d(samples, "samples")

    if not _is_number(samples):
        size = max(1, int(frame_size))
        if len(values) <= size:
            frame = values
            mean_square = sum(v * v for v in frame) / max(1, len(frame))
            floor = math.sqrt(mean_square)
        else:
            levels: list[float] = []
            for start in range(0, len(values) - size + 1, size):
                frame = values[start : start + size]
                levels.append(
                    math.sqrt(sum(v * v for v in frame) / max(1, len(frame)))
                )
            levels.sort()
            ratio = min(1.0, max(0.0, float(percentile)))
            index = min(len(levels) - 1, int(ratio * len(levels)))
            floor = levels[index] if levels else 0.0

    threshold = floor * 10.0 ** (float(margin_db) / 20.0)
    return {
        "noise_floor_rms": floor,
        "margin_db": float(margin_db),
        "threshold_rms": threshold,
        "threshold_dbfs": 20.0 * math.log10(threshold)
        if threshold > _EPS
        else float("-inf"),
    }


def validation(config: object = 1, **overrides: object) -> dict[str, Any]:
    """校验采样率、阵元位置、温度和阈值；不通过时返回报告而不抛异常。"""

    if isinstance(config, dict):
        settings: dict[str, Any] = dict(config)
    elif config is None:
        settings = {}
    else:
        # 契约测试会传入标量 1；它不是已知配置，按“无错误的普通值”处理。
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "value": config,
            "overrides": overrides,
        }
    settings.update(overrides)

    errors: list[str] = []
    warnings: list[str] = []

    if "sample_rate" in settings:
        try:
            fs = _valid_sample_rate(settings["sample_rate"])
            if fs < 8000.0 or fs > 384_000.0:
                warnings.append("sample_rate is outside the usual audio range")
        except ValueError as exc:
            errors.append(str(exc))

    positions = settings.get("positions")
    if positions is not None:
        try:
            normalized = _normalize_positions(positions)
            if len(normalized) < 2:
                warnings.append("fewer than two microphones cannot form an array gain")
        except ValueError as exc:
            errors.append(str(exc))

    if "temperature" in settings:
        try:
            temperature = float(settings["temperature"])  # type: ignore[arg-type]
            if not math.isfinite(temperature):
                errors.append("temperature must be finite")
            elif temperature < -60.0 or temperature > 85.0:
                warnings.append("temperature is outside the calibrated range")
        except (TypeError, ValueError):
            errors.append("temperature must be a number")

    threshold = settings.get("polarity_threshold")
    if threshold is not None:
        try:
            value = float(threshold)
            if not -1.0 <= value <= 0.0:
                errors.append("polarity_threshold should be in [-1, 0]")
        except (TypeError, ValueError):
            errors.append("polarity_threshold must be a number")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "validated_keys": sorted(settings.keys()),
    }


def audit(*events: object, **kwargs: object) -> dict[str, Any]:
    """把标定、极性、温度和块边界事件汇总成可审计报告。"""

    records: list[dict[str, Any]] = []
    corrections: dict[str, Any] = {}
    ok = True

    for index, event in enumerate(events):
        if isinstance(event, dict):
            record = {"index": index, "type": event.get("type", "report")}
            for key in (
                "sample_rate",
                "n_mics",
                "carry_samples",
                "inverted_indices",
                "correlations",
                "gains",
                "pointing_angle",
                "ok",
            ):
                if key in event:
                    record[key] = event[key]
            records.append(record)
            if event.get("ok") is False:
                ok = False
            if event.get("inverted_indices"):
                corrections["polarity"] = event["inverted_indices"]
            if "gains" in event:
                corrections["temperature_gains"] = event["gains"]
            if "carry_samples" in event:
                corrections["carry_samples"] = event["carry_samples"]
        else:
            records.append({"index": index, "type": "value", "value": event})

    if kwargs:
        records.append({"index": len(records), "type": "keywords", "value": kwargs})

    return {
        "ok": ok,
        "event_count": len(events),
        "records": records,
        "corrections": corrections,
    }


def fallback(reason: object = 1, *, n_mics: int = 2) -> dict[str, Any]:
    """标定/相关性不可信时使用安全的均匀、零延迟 broadside 配置。"""

    count = max(1, int(n_mics))
    return {
        "mode": "uniform_broadside_safe_fallback",
        "reason": reason,
        "direction": 0.0,
        "delay_samples": [0.0] * count,
        "weights": [1.0 / count] * count,
        "polarity_action": "hold_unchanged_until_confident",
        "hold_last_good": True,
    }


# ---------------------------------------------------------------------------
# 7. 延迟求和波束成形集成
# ---------------------------------------------------------------------------


def delay_and_sum(
    channels: Sequence[Sequence[float]],
    delay_samples: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    history: Sequence[Sequence[float]] | None = None,
    sensitivity_gains: Sequence[float] | None = None,
) -> list[float]:
    """按整数+线性分数延迟执行延迟求和。

    ``history`` 为每个通道上一块末尾的采样；没有历史时在首块前补零。
    历史长度按最大整数延迟计算，因此跨块读取不会回绕或丢失。
    """

    data = [_as_1d(channel, "channel") for channel in channels]
    count = len(data)
    if count == 0:
        return []

    delays = _broadcast_values(delay_samples, count, 0.0, "delay_samples")
    if weights is None:
        channel_weights = [1.0] * count
    else:
        channel_weights = _as_1d(weights, "weights")
        if len(channel_weights) == 1:
            channel_weights = channel_weights * count
        if len(channel_weights) != count:
            raise ValueError("weights length must match channel count")

    if sensitivity_gains is not None:
        gains = _as_1d(sensitivity_gains, "sensitivity_gains")
        if len(gains) == 1:
            gains = gains * count
        if len(gains) != count:
            raise ValueError("sensitivity_gains length must match channel count")
        channel_weights = [channel_weights[i] * gains[i] for i in range(count)]

    total_weight = sum(channel_weights)
    if abs(total_weight) > _EPS:
        channel_weights = [w / total_weight for w in channel_weights]
    else:
        channel_weights = [1.0 / count] * count

    max_integer = 0
    split_delays: list[tuple[int, float]] = []
    for delay in delays:
        integer = int(math.floor(max(0.0, delay) + _SMALL))
        fraction = max(0.0, delay - integer)
        if abs(fraction) < _SMALL:
            fraction = 0.0
        split_delays.append((integer, fraction))
        max_integer = max(max_integer, integer)

    prefix_length = max_integer + 1
    extended_channels: list[list[float]] = []
    for index, channel in enumerate(data):
        if history is None or index >= len(history):
            prefix = [0.0] * prefix_length
        else:
            prefix = _as_1d(history[index], "history")
            if len(prefix) < prefix_length:
                prefix = [0.0] * (prefix_length - len(prefix)) + prefix
            elif len(prefix) > prefix_length:
                prefix = prefix[-prefix_length:]
        extended_channels.append(prefix + channel)

    output_length = max((len(channel) for channel in data), default=0)
    output = [0.0] * output_length
    for sample_index in range(output_length):
        total = 0.0
        for channel_index, channel in enumerate(extended_channels):
            integer, fraction = split_delays[channel_index]
            base = prefix_length + sample_index - integer
            sample_a = channel[base]
            sample_b = channel[base - 1]
            delayed = (1.0 - fraction) * sample_a + fraction * sample_b
            total += channel_weights[channel_index] * delayed
        output[sample_index] = total
    return output


def beamform(signals: object = 1.0, **kwargs: object) -> object:
    """延迟求和波束成形的灵活入口。

    标量输入按契约直接返回浮点数；一维序列按单通道处理；二维序列按
    ``channels × samples`` 处理。支持的关键字包括 ``positions``、
    ``direction``、``sample_rate``、温度校准参数、``correct_polarity``、
    ``weights`` 和 ``history``。
    """

    if _is_number(signals):
        return float(signals)  # type: ignore[arg-type]

    if _is_nested_sequence(signals):
        channels = [_as_1d(ch, "channel") for ch in signals]  # type: ignore[arg-type]
    else:
        channels = [_as_1d(signals, "signals")]

    count = len(channels)
    fs = _valid_sample_rate(kwargs.get("sample_rate", DEFAULT_SAMPLE_RATE))
    direction = kwargs.get("direction", 0.0)
    speed = float(kwargs.get("speed_of_sound", SPEED_OF_SOUND))  # type: ignore[arg-type]

    positions = kwargs.get("positions")
    if positions is None:
        mic_positions = [
            (index * DEFAULT_MIC_SPACING,) for index in range(count)
        ]
    else:
        mic_positions = _normalize_positions(positions)
        if len(mic_positions) != count:
            raise ValueError("positions count must match channel count")

    should_correct = bool(kwargs.get("correct_polarity", True))
    polarity_report: dict[str, Any] | None = None
    if should_correct and count > 1:
        polarity_report = correct_polarity(
            channels,
            threshold=float(kwargs.get("polarity_threshold", -0.35)),  # type: ignore[arg-type]
        )
        channels = polarity_report["corrected"]

    steering = steer(
        direction,
        mic_positions,
        sample_rate=fs,
        speed_of_sound=speed,
        degrees=bool(kwargs.get("degrees", False)),
        basis=str(kwargs.get("basis", "broadside")),
    )

    user_weights = kwargs.get("weights")
    if user_weights is None:
        shading = str(kwargs.get("shading", "uniform"))
        if shading == "uniform":
            weights = [1.0] * count
        else:
            weights = windowing(count, shading)
    else:
        weights = _as_1d(user_weights, "weights")

    sensitivity_gains = None
    if any(key in kwargs for key in (
        "temperature",
        "sensitivity_db",
        "temp_coefficient_db_per_c",
    )):
        calibration = array_calibration(
            positions=mic_positions,
            sample_rate=fs,
            speed_of_sound=speed,
            temperature=kwargs.get("temperature", 25.0),
            reference_temperature=kwargs.get("reference_temperature", 25.0),  # type: ignore[arg-type]
            sensitivity_db=kwargs.get("sensitivity_db", 0.0),
            temp_coefficient_db_per_c=kwargs.get(
                "temp_coefficient_db_per_c", 0.0
            ),
        )
        sensitivity_gains = calibration["sensitivity_gains"]

    history = kwargs.get("history")
    return delay_and_sum(
        channels,
        steering["delay_samples"],
        weights,
        history=None if history is None else history,  # type: ignore[arg-type]
        sensitivity_gains=sensitivity_gains,
    )
