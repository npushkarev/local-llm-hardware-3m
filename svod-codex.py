# -*- coding: utf-8 -*-
"""Единая сводная таблица закупки LLM-железа.

По умолчанию скрипт печатает Markdown:

    python3 svod.py

HTML-фрагмент строится из тех же данных:

    python3 svod.py --format html

Зависимостей, кроме стандартной библиотеки Python, нет. Число без пометы
происхождения в производительных метриках считается ошибкой модели данных.
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from typing import Iterable


BUDGET_RUB = 3_000_000
CONTEXT_TOKENS = 120_000
ANSWER_TOKENS = 200
AGENT_STEPS_PER_HOUR = 20
CHAT_MIN_TPS = 5.0
CHAT_REFERENCE_CONCURRENCY = 96
CHAT_REFERENCE_AVG_TPS = 5.11
FIT_DISPLAY_HEADROOM_GIB = 5.0
GIB = 1024**3
GB_TO_GIB = 10**9 / GIB

LOCAL = "local"
EXTERNAL = "external"
CALCULATED = "calculated"
UNAVAILABLE = "unavailable"

TAG = {
    LOCAL: "[З]",
    EXTERNAL: "[В]",
    CALCULATED: "[Р]",
    UNAVAILABLE: "н/д",
}


@dataclass(frozen=True)
class Metric:
    """Число вместе с обязательным происхождением."""

    value: float | None
    kind: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in TAG:
            raise ValueError(f"Неизвестный вид метрики: {self.kind}")
        if self.kind == UNAVAILABLE and self.value is not None:
            raise ValueError("Недоступная метрика не может содержать число")
        if self.kind != UNAVAILABLE and self.value is None:
            raise ValueError("Числовая метрика должна содержать значение")
        if not self.source:
            raise ValueError("Для каждой метрики нужен источник или формула")


NO_PREFILL = Metric(None, UNAVAILABLE, "замера prefill нет")
NO_GENERATION = Metric(None, UNAVAILABLE, "замера генерации нет")


@dataclass(frozen=True)
class Node:
    name: str
    short: str
    price_rub: int
    memory_gb: float
    bandwidth_gbs: float | None
    power: str
    price_status: str
    qualification: str = ""

    @property
    def screening_capacity_gib(self) -> float:
        """Консервативный перевод паспортных GB в GiB для проверки вместимости."""

        return self.memory_gb * GB_TO_GIB


NODES = {
    "spark": Node(
        "NVIDIA DGX Spark",
        "Spark",
        583_588,
        128,
        273,
        "TDP GB10 140 Вт; БП 240 Вт; потребление из розетки н/д",
        "розница NIX",
    ),
    "pro6000": Node(
        "RTX PRO 6000 Blackwell WE",
        "PRO 6000",
        1_550_000,
        96,
        1792,
        "GPU: макс. 600 Вт; хост н/д",
        "рыночный снимок",
    ),
    "pro5000": Node(
        "RTX PRO 5000 Blackwell 72 ГБ",
        "PRO 5000",
        979_800,
        72,
        1344,
        "GPU: TBP 300 Вт; хост н/д",
        "рыночный снимок; НДС не уточнён",
    ),
    "rtx4090": Node(
        "RTX 4090 48 ГБ, модифицированная",
        "4090 мод.",
        450_000,
        48,
        None,
        "450 Вт TGP — только штатная 24 ГБ; модификация/хост н/д",
        "рыночная оценка модификации",
        "48 ГБ не являются официальной конфигурацией NVIDIA",
    ),
    "halo": Node(
        "мини-ПК AMD Strix Halo 128 ГБ",
        "Halo",
        263_350,
        128,
        256,
        "TDP референсной Halo-платформы 120 Вт; конкретный мини-ПК/розетка н/д",
        "229 000 ₽ + 15%, расчёт",
        "256 ГБ/с и TDP 120 Вт относятся к референсной AMD-платформе; точный OEM SKU не задан",
    ),
    "mac": Node(
        "Mac Studio M3 Ultra 256 ГБ",
        "Mac",
        756_900,
        256,
        819,
        "предельная длительная мощность 480 Вт; LLM-нагрузка н/д",
        "оценка; точный SKU не задан",
    ),
    "epyc": Node(
        "сервер EPYC 768 ГБ без GPU",
        "EPYC",
        835_000,
        768,
        None,
        "н/д: нет BOM и замера из розетки",
        "оценка; BOM не задан",
        "SKU CPU, память и фактическая полоса не заданы",
    ),
}


@dataclass(frozen=True)
class Model:
    name: str
    short: str
    weights_gib: float
    parameters: str
    full_attention_layers: int
    sliding_attention_layers: int
    sliding_window: int | None
    kv_heads: int
    head_dim: int
    architecture_note: str
    weights_note: str


MODELS = {
    "qwen": Model(
        "Qwen3.8-27B Q4_K_M",
        "Qwen3.8",
        15.66,
        "27B, плотная гибридная",
        16,
        0,
        None,
        4,
        256,
        "KV: 16 слоёв полного внимания, 4 головы × 256; ещё 48 слоёв DeltaNet",
        "размер локального GGUF Q4_K_M; визуальный проектор не включён",
    ),
    "laguna": Model(
        "Laguna S 2.1 Q4_K_M",
        "Laguna",
        66.28,
        "118B / около 8B активных, MoE",
        12,
        36,
        512,
        8,
        128,
        "KV: 12 полных + 36 SWA 512, 8 голов × 128",
        "размер публичной конверсии AtomicChat Q4_K_M",
    ),
    "gptoss": Model(
        "gpt-oss-120b MXFP4",
        "gpt-oss",
        60.8,
        "116,8B / 5,1B активных, MoE",
        18,
        18,
        128,
        8,
        64,
        "KV: 18 полных + 18 SWA 128, 8 голов × 64",
        "официальный чекпойнт; MXFP4 — веса MoE-экспертов, не все тензоры",
    ),
}


def kv_cache_range_gib(model: Model, tokens: int = CONTEXT_TOKENS) -> tuple[float, float]:
    """KV одного потока в FP16/BF16.

    Нижняя граница предполагает, что движок освобождает историю за пределами
    sliding window. Верхняя граница предполагает полный размер кэша и нужна для
    осторожной проверки вместимости. Фиксированное состояние линейного внимания,
    vision-projector и runtime-буферы сюда не входят.
    """

    bytes_per_layer_token = 2 * model.kv_heads * model.head_dim * 2  # K+V, 2 байта
    retained_tokens = model.full_attention_layers * tokens
    if model.sliding_attention_layers:
        retained_tokens += model.sliding_attention_layers * min(
            tokens, model.sliding_window or tokens
        )
    full_tokens = (
        model.full_attention_layers + model.sliding_attention_layers
    ) * tokens
    low = retained_tokens * bytes_per_layer_token / GIB
    high = full_tokens * bytes_per_layer_token / GIB
    return low, high


def memory_range_gib(model: Model) -> tuple[float, float]:
    low, high = kv_cache_range_gib(model)
    return model.weights_gib + low, model.weights_gib + high


# Только прямые замеры точной пары «узел + модель». Никакой подстановки
# prefill между родственными моделями нет.
DIRECT_RATES: dict[tuple[str, str], dict[str, Metric]] = {
    ("spark", "qwen"): {
        "pre": Metric(835.61, LOCAL, "zamery-src.md, 2026-08-16 19:47"),
        "gen": Metric(12.34, LOCAL, "zamery-src.md, 2026-08-16 19:47"),
    },
    ("halo", "qwen"): {
        "pre": Metric(214.26, LOCAL, "zamery-src.md, 2026-08-16 19:47"),
        "gen": Metric(12.76, LOCAL, "zamery-src.md, 2026-08-16 19:47"),
    },
    ("spark", "laguna"): {
        "pre": Metric(683.50, LOCAL, "zamery-src.md, 2026-08-15 21:05"),
        "gen": Metric(30.19, LOCAL, "zamery-src.md, 2026-08-15 21:05"),
    },
    ("halo", "laguna"): {
        "pre": Metric(353.82, LOCAL, "zamery-src.md, 2026-08-15 21:08"),
        "gen": Metric(40.00, LOCAL, "zamery-src.md, 2026-08-15 21:08"),
    },
    ("spark", "gptoss"): {
        "pre": NO_PREFILL,
        "gen": Metric(
            33.53,
            EXTERNAL,
            "Dendro Logic, тест конкурентности DGX Spark, 2026-04-21",
        ),
    },
}

CUDA_CARDS = {"pro6000", "pro5000", "rtx4090"}


def rates_for(node_key: str, model_key: str) -> tuple[Metric, Metric]:
    """Prefill и одиночная генерация в токенах в секунду.

    Для CUDA-карт оставлена только явно помеченная оценка одиночной генерации
    по паспортной полосе памяти. Она информационная и не используется для чатов
    или агентов без prefill/concurrency-замера.
    """

    direct = DIRECT_RATES.get((node_key, model_key))
    if direct:
        return direct["pre"], direct["gen"]

    if node_key in CUDA_CARDS:
        reference = DIRECT_RATES.get(("spark", model_key))
        bandwidth = NODES[node_key].bandwidth_gbs
        if reference and reference["gen"].value is not None and bandwidth is not None:
            estimate = reference["gen"].value * bandwidth / NODES["spark"].bandwidth_gbs
            return NO_PREFILL, Metric(
                estimate,
                CALCULATED,
                f"одиночная генерация Spark {TAG[reference['gen'].kind]} "
                "× паспортная полоса карты / 273",
                "оценка порядка, не бенчмарк",
            )
    return NO_PREFILL, NO_GENERATION


# При целевом среднем не менее 5 т/с внешний нагрузочный прогон даёт 96 потоков по
# 5,11 т/с; 128 потоков уже ниже цели (4,52 т/с). Это не p95/SLA и не
# переносится с пары Spark+gpt-oss на другие модели или платформы.
CHAT_SWEEP = {
    ("spark", "gptoss"): Metric(
        CHAT_REFERENCE_CONCURRENCY,
        EXTERNAL,
        "Dendro Logic: около 1,5 тыс. входных, до 400 выходных токенов, "
        "vLLM 26.03, в среднем 5,11 т/с",
    )
}


@dataclass(frozen=True)
class CheckpointFacts:
    name: str
    parameters: str
    repository_gb: int


@dataclass(frozen=True)
class GlmSparkProfile:
    checkpoint: CheckpointFacts
    nodes: int
    stack: str
    prefill_tps: int
    generation_tps: tuple[int, int]
    generation_context_tokens: int
    short_generation_peak_tps: float
    source: str


GLM_QUANTTRIO = CheckpointFacts(
    "QuantTrio GLM-5.2 Int4/Int8Mix",
    "744B / около 40B активных",
    406,
)
GLM_NVIDIA = CheckpointFacts(
    "NVIDIA GLM-5.2 NVFP4",
    "753B / 40B активных",
    465,
)
GLM_FOUR_SPARK = GlmSparkProfile(
    checkpoint=GLM_QUANTTRIO,
    nodes=4,
    stack="модифицированный vLLM, NVFP4 KV",
    prefill_tps=819,
    generation_tps=(29, 33),
    generation_context_tokens=53_000,
    short_generation_peak_tps=42.3,
    source="0xdfi, публичный профиль одного стенда",
)


def glm_four_spark_summary() -> str:
    """Краткая подпись без смешения измеренного и официального чекпойнтов."""

    profile = GLM_FOUR_SPARK
    gen_low, gen_high = profile.generation_tps
    return (
        f"{profile.checkpoint.name} {profile.checkpoint.repository_gb} ГБ, "
        f"{profile.stack}: prefill {profile.prefill_tps} т/с; генерация "
        f"{gen_low}–{gen_high} т/с при ≈"
        f"{profile.generation_context_tokens // 1_000} тыс., отдельный короткий "
        f"пик {str(profile.short_generation_peak_tps).replace('.', ',')} т/с [В]"
    )


@dataclass(frozen=True)
class Part:
    node: str
    count: int
    mode: str = "independent"  # independent, pcie_tp, network_tp


@dataclass(frozen=True)
class ExtraCost:
    name: str
    rub: int
    status: str


HOST_ONE = ExtraCost("хост под одну карту", 400_000, "оценка, КП нет")
HOST_FOUR = ExtraCost("хост под четыре карты", 735_000, "оценка, КП нет")
CABLE = ExtraCost("кабель 200 Гбит/с", 30_000, "оценка, КП нет")


@dataclass(frozen=True)
class Build:
    name: str
    parts: tuple[Part, ...]
    extras: tuple[ExtraCost, ...]
    topology: str
    max_pool_gb: float
    power: str
    purpose: str
    unknown_cost: str = ""

    @property
    def price_rub(self) -> int:
        return sum(NODES[p.node].price_rub * p.count for p in self.parts) + sum(
            item.rub for item in self.extras
        )

    @property
    def total_memory_gb(self) -> float:
        return sum(NODES[p.node].memory_gb * p.count for p in self.parts)


BUILDS = (
    Build(
        "RTX PRO 6000 + хост + 1 Spark + 1 Halo",
        (Part("pro6000", 1), Part("spark", 1), Part("halo", 1)),
        (HOST_ONE,),
        "96 + 128 + 128 ГБ раздельно (паспортно)",
        128,
        "PRO 6000: макс. 600 Вт; Spark: БП 240 Вт; Halo: TDP референсной "
        "платформы 120 Вт; конкретный Halo/хост/розетка н/д",
        "кандидат на раздельный парк до 96/128 ГБ; решение после замера карты",
    ),
    Build(
        "4 × RTX 4090 48 ГБ, модиф. + хост",
        (Part("rtx4090", 4, "pcie_tp"),),
        (HOST_FOUR,),
        "4 × 48 ГБ; теоретически шардируются по PCIe, единый пул не подтверждён; NVLink нет",
        192,
        "4 × 450 Вт TGP штатных 24 ГБ; модификация/хост н/д",
        "вариант с риском: неофициальная память, нет гарантии производителя и TP-замера",
    ),
    Build(
        "RTX PRO 5000 + хост + 2 Spark + кабель",
        (Part("pro5000", 1), Part("spark", 2, "network_tp")),
        (HOST_ONE, CABLE),
        "72 ГБ отдельно + теоретический пул 2 × 128 ГБ по сети",
        256,
        "PRO 5000: 300 Вт TBP; 2 × Spark: БП 240 Вт; хост н/д",
        "кандидат для класса 128–256 ГБ; запуск и скорость сетевого TP не измерены",
    ),
    Build(
        "RTX PRO 6000 + хост",
        (Part("pro6000", 1),),
        (HOST_ONE,),
        "96 ГБ паспортно",
        96,
        "GPU: макс. 600 Вт; хост н/д",
        "кандидат с высокой паспортной полосой до 96 ГБ; скорость агента не измерена",
    ),
    Build(
        "4 × DGX Spark в сетевом пуле",
        (Part("spark", 4, "network_tp"),),
        (),
        "теоретический пул 4 × 128 ГБ; топология сети не определена",
        512,
        "4 × БП 240 Вт; потребление из розетки н/д",
        "класс свыше 256 ГБ; внешний профиль только для "
        f"{glm_four_spark_summary()}; {GLM_NVIDIA.name} "
        f"{GLM_NVIDIA.repository_gb} ГБ — скорость н/д",
        "сеть, кабели или коммутатор",
    ),
    Build(
        "2 × DGX Spark + кабель",
        (Part("spark", 2, "network_tp"),),
        (CABLE,),
        "теоретический пул 2 × 128 ГБ по 200 Гбит/с",
        256,
        "2 × БП 240 Вт; потребление из розетки н/д",
        "класс 128–256 ГБ; линк покупает вместимость, запуск и ускорение не доказаны",
    ),
    Build(
        "5 × DGX Spark независимыми",
        (Part("spark", 5),),
        (),
        "5 × 128 ГБ раздельно (паспортно)",
        128,
        "5 × БП 240 Вт; потребление из розетки н/д",
        "агрегированная пропускная способность реплик; одна модель не больше 128 ГБ паспортно",
    ),
    Build(
        "11 × мини-ПК Strix Halo",
        (Part("halo", 11),),
        (),
        "11 × 128 ГБ раздельно (паспортно)",
        128,
        "11 × TDP 120 Вт референсной платформы; конкретные мини-ПК/розетка н/д",
        "много независимых реплик; длинный агентный шаг даже по локальным замерам",
    ),
    Build(
        "3 × Mac Studio M3 Ultra 256 ГБ",
        (Part("mac", 3),),
        (),
        "3 × 256 ГБ раздельно (паспортно)",
        256,
        "3 × 480 Вт предельной длительной мощности; LLM-нагрузка н/д",
        "много памяти на узел; производительность выбранных моделей не измерена",
    ),
    Build(
        "EPYC 768 ГБ без GPU",
        (Part("epyc", 1),),
        (),
        "768 ГБ в одном хосте (состав не задан)",
        768,
        "н/д: нет BOM и замера",
        "максимальная вместимость; без BOM, полосы и бенчмарка производительность н/д",
    ),
)


@dataclass(frozen=True)
class PartEvaluation:
    part: Part
    placement: str
    viable: bool
    memory_not_borderline: bool
    mode: str
    replicas: int
    devices_per_replica: int
    pre: Metric
    gen: Metric


def _fit_label(capacity_gib: float, model: Model) -> str:
    low, high = memory_range_gib(model)
    if capacity_gib < low:
        return "не влезает"
    if capacity_gib < high:
        if capacity_gib - low < FIT_DISPLAY_HEADROOM_GIB:
            return "погранично даже при освобождении SWA; буферы движка н/д"
        return "зависит от освобождения SWA; буферы движка н/д"
    if capacity_gib - high < FIT_DISPLAY_HEADROOM_GIB:
        return "впритык; буферы движка н/д"
    return "веса+KV влезают; буферы движка н/д"


def evaluate_part(part: Part, model_key: str) -> PartEvaluation:
    node = NODES[part.node]
    model = MODELS[model_key]
    need_low, need_high = memory_range_gib(model)
    capacity_gib = node.screening_capacity_gib

    if capacity_gib >= need_low:
        pre, gen = rates_for(part.node, model_key)
        fit = _fit_label(capacity_gib, model)
        memory_not_borderline = (
            capacity_gib - need_high >= FIT_DISPLAY_HEADROOM_GIB
        )
        launch_note = (
            "; запуск на конкретной модификации не проверен"
            if part.node == "rtx4090"
            else ""
        )
        if memory_not_borderline:
            placement = (
                f"{node.short}: по номинальному объёму {part.count} репл.; "
                f"{fit}{launch_note}"
            )
        else:
            placement = (
                f"{node.short}: размещение не подтверждено ({fit}){launch_note}"
            )
        return PartEvaluation(
            part,
            placement,
            True,
            memory_not_borderline,
            "replicas",
            part.count,
            1,
            pre,
            gen,
        )

    if part.mode in {"pcie_tp", "network_tp"}:
        # Берём осторожную верхнюю границу KV. Runtime всё равно неизвестен.
        devices = max(2, math.ceil(need_high / capacity_gib))
        if devices <= part.count:
            replicas = part.count // devices
            link = "PCIe TP" if part.mode == "pcie_tp" else "сетевой TP"
            placement = (
                f"{node.short}: по номинальному объёму возможно "
                f"{replicas} × {link}{devices}; "
                f"{_fit_label(capacity_gib * devices, model)}; "
                "запуск и скорость не проверены"
            )
            return PartEvaluation(
                part,
                placement,
                True,
                capacity_gib * devices - need_high >= FIT_DISPLAY_HEADROOM_GIB,
                "tp",
                replicas,
                devices,
                NO_PREFILL,
                NO_GENERATION,
            )

        if capacity_gib * part.count >= need_low:
            link = "PCIe TP" if part.mode == "pcie_tp" else "сетевой TP"
            return PartEvaluation(
                part,
                f"{node.short}: по номинальному объёму возможен {link}{part.count}, "
                "но вместимость зависит от SWA и буферов; запуск и скорость не проверены",
                True,
                False,
                "tp",
                1,
                part.count,
                NO_PREFILL,
                NO_GENERATION,
            )

    return PartEvaluation(
        part,
        f"{node.short}: не влезает",
        False,
        False,
        "none",
        0,
        0,
        NO_PREFILL,
        NO_GENERATION,
    )


def step_seconds(pre: Metric, gen: Metric) -> float | None:
    if pre.value is None or gen.value is None:
        return None
    return CONTEXT_TOKENS / pre.value + ANSWER_TOKENS / gen.value


def _fmt_number(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}".replace(",", " ")
    return text.replace(".", ",")


def _fmt_rate(metric: Metric) -> str:
    if metric.value is None:
        return "н/д"
    if metric.kind == CALCULATED:
        if "[З]" in metric.source:
            tag = "[Р из З]"
        elif "[В]" in metric.source:
            tag = "[Р из В]"
        else:
            tag = TAG[CALCULATED]
        return f"≈{_fmt_number(metric.value)} {tag}"
    return f"{_fmt_number(metric.value, 1)} {TAG[metric.kind]}"


def _rate_pair(evaluation: PartEvaluation) -> str:
    node = NODES[evaluation.part.node]
    if not evaluation.viable:
        return f"{node.short}: не влезает"
    if evaluation.mode == "tp":
        return f"{node.short} TP{evaluation.devices_per_replica}: н/д"
    prefix = f"{node.short} (если запустится)" if not evaluation.memory_not_borderline else node.short
    if evaluation.pre.kind == evaluation.gen.kind and evaluation.pre.value is not None:
        return (
            f"{prefix}: {_fmt_number(evaluation.pre.value, 1)}/"
            f"{_fmt_number(evaluation.gen.value, 1)} {TAG[evaluation.pre.kind]}"
        )
    return f"{prefix}: {_fmt_rate(evaluation.pre)}/{_fmt_rate(evaluation.gen)}"


def _derived_tag(pre: Metric, gen: Metric) -> str:
    origins = {pre.kind, gen.kind}
    origins.discard(UNAVAILABLE)
    if origins == {LOCAL}:
        return "[Р из З]"
    if origins == {EXTERNAL}:
        return "[Р из В]"
    return "[Р]"


def model_cell(build: Build, model_key: str) -> list[str]:
    evaluations = [evaluate_part(part, model_key) for part in build.parts]
    placement = "; ".join(item.placement for item in evaluations)
    rates = "; ".join(_rate_pair(item) for item in evaluations)

    known: list[tuple[PartEvaluation, float]] = []
    unknown_agent_pool = False
    for item in evaluations:
        if not item.viable:
            continue
        if not item.memory_not_borderline:
            unknown_agent_pool = True
            continue
        step = step_seconds(item.pre, item.gen) if item.mode == "replicas" else None
        if step is None:
            unknown_agent_pool = True
        else:
            known.append((item, step))

    if known:
        best_item, best_step = min(known, key=lambda pair: pair[1])
        best_prefix = "лучший известный " if unknown_agent_pool else ""
        step_line = (
            f"линейная оценка шага: {best_prefix}≈{_fmt_number(best_step, 1)} с "
            f"{_derived_tag(best_item.pre, best_item.gen)} ({NODES[best_item.part.node].short})"
        )
        agent_equivalent = sum(
            item.replicas * 3600 / (AGENT_STEPS_PER_HOUR * seconds)
            for item, seconds in known
        )
        pinned_sessions = sum(
            item.replicas * math.floor(3600 / (AGENT_STEPS_PER_HOUR * seconds))
            for item, seconds in known
        )
        if unknown_agent_pool:
            pinned = f"{pinned_sessions} на известных узлах"
            scope = " по известным узлам; остальные н/д"
        else:
            pinned = f"{pinned_sessions}"
            scope = ""
        agent_line = (
            f"агентная нагрузка: ≈{_fmt_number(agent_equivalent, 2)} экв. [Р]{scope}; "
            f"закреплённых сессий {pinned} [Р по точечной оценке]"
        )
    else:
        step_line = "линейная оценка шага: н/д (нет prefill/TP-замера)"
        agent_line = "агентная нагрузка: н/д"

    chat_known = 0.0
    chat_sources = 0
    chat_unknown_pool = False
    for item in evaluations:
        if not item.viable:
            continue
        if not item.memory_not_borderline:
            chat_unknown_pool = True
            continue
        reference = CHAT_SWEEP.get((item.part.node, model_key))
        if item.mode == "replicas" and reference and reference.value is not None:
            chat_known += reference.value * item.replicas
            chat_sources += item.replicas
        else:
            chat_unknown_pool = True

    if chat_known:
        # Sweep подтверждает одну рабочую точку, но не SLA и не точную
        # границу ёмкости между 96 (5,11 т/с) и 128 (4,52 т/с).
        suffix = " (только известные пулы)" if chat_unknown_pool else ""
        if chat_sources == 1:
            observed = f"{_fmt_number(chat_known)} активных в измеренной точке [В]"
        else:
            observed = (
                f"{CHAT_REFERENCE_CONCURRENCY} × {chat_sources} = "
                f"{_fmt_number(chat_known)} активных "
                "[Р из В]"
            )
        chat_line = (
            f"чат (1,5k/≤400; целевое среднее ≥{_fmt_number(CHAT_MIN_TPS)} т/с): "
            f"{observed}{suffix}; в опорном прогоне "
            f"{_fmt_number(CHAT_REFERENCE_AVG_TPS, 2)} т/с"
        )
    else:
        chat_line = (
            f"чат (1,5k/≤400; среднее ≥{_fmt_number(CHAT_MIN_TPS, 0)} т/с): н/д"
        )

    return [
        f"размещение: {placement}",
        f"чтение/ген., т/с: {rates}",
        step_line,
        agent_line,
        chat_line,
    ]


def _model_header(model_key: str) -> list[str]:
    model = MODELS[model_key]
    kv_low, kv_high = kv_cache_range_gib(model)
    mem_low, mem_high = memory_range_gib(model)
    if math.isclose(kv_low, kv_high):
        kv = _fmt_number(kv_low, 2)
        memory = _fmt_number(mem_low, 2)
    else:
        kv = f"{_fmt_number(kv_low, 2)}–{_fmt_number(kv_high, 2)}"
        memory = f"{_fmt_number(mem_low, 2)}–{_fmt_number(mem_high, 2)}"
    return [
        model.name,
        model.parameters,
        f"веса { _fmt_number(model.weights_gib, 2) } ГиБ",
        model.weights_note,
        model.architecture_note,
        f"KV 120k {kv} ГиБ [Р]",
        f"веса+KV {memory} ГиБ; буферы движка н/д",
    ]


def _cost_cell(build: Build) -> list[str]:
    balance = BUDGET_RUB - build.price_rub
    if build.unknown_cost:
        price = f"{_fmt_number(build.price_rub)} ₽ + {build.unknown_cost} н/д"
        remainder = f"остаток до {_fmt_number(balance)} ₽"
    else:
        price = f"{_fmt_number(build.price_rub)} ₽ [Р]"
        remainder = f"остаток {_fmt_number(balance)} ₽"
    statuses = [
        f"{NODES[part.node].short}: {NODES[part.node].price_status}"
        for part in build.parts
    ]
    statuses.extend(f"{item.name}: {item.status}" for item in build.extras)
    return [
        price,
        remainder,
        "основания: " + "; ".join(statuses),
        "срез 18.08.2026; ни одна цена не является КП",
    ]


def _memory_cell(build: Build) -> list[str]:
    price_per_pool_gb = build.price_rub / build.max_pool_gb
    prefix = "от " if build.unknown_cost else ""
    seen: set[str] = set()
    nodes = []
    for part in build.parts:
        if part.node not in seen:
            seen.add(part.node)
            nodes.append(NODES[part.node])
    bandwidths = "; ".join(
        f"{node.short}: "
        + (f"{_fmt_number(node.bandwidth_gbs)} ГБ/с" if node.bandwidth_gbs else "н/д")
        for node in nodes
    )
    screening_capacities = "; ".join(
        f"{node.short}: {_fmt_number(node.memory_gb)} ГБ → "
        f"{_fmt_number(node.screening_capacity_gib, 2)} ГиБ"
        for node in nodes
    )
    result = [
        build.topology,
        f"паспортно всего {_fmt_number(build.total_memory_gb)} ГБ; "
        f"наибольшая логическая ёмкость {_fmt_number(build.max_pool_gb)} ГБ",
        f"для консервативной проверки до резервов: {screening_capacities}",
        f"паспортная полоса: {bandwidths}",
        f"{prefix}{_fmt_number(price_per_pool_gb)} ₽/ГБ логической ёмкости [Р]",
    ]
    qualifications = [node.qualification for node in nodes if node.qualification]
    if qualifications:
        result.append("оговорка: " + "; ".join(qualifications))
    return result


def _markdown_cell(lines: Iterable[str]) -> str:
    return "<br>".join(str(line).replace("|", "\\|") for line in lines)


def _html_cell(lines: Iterable[str], header: bool = False) -> str:
    tag = "th" if header else "td"
    content = "<br>".join(html.escape(str(line)) for line in lines)
    return f"<{tag}>{content}</{tag}>"


METHOD_NOTES = (
    "[З] — замер на собственном стенде; [В] — внешний замер точной пары модель/платформа; [Р] — расчёт. «н/д» означает, что число не подставлялось.",
    "Линейная оценка шага агента [Р] = 120 000 / prefill + 200 / скорость генерации при полном промахе кэша префикса. Это экстраполяция однопоточного llama-bench, а не замер шага на глубине 120 тыс. токенов.",
    "Агентный эквивалент — агрегированная пропускная способность при 20 шагах/ч и 100% загрузке. Дробь не является человеком. «Закреплённая сессия» требует, чтобы один узел по точечной линейной оценке укладывался в 180 с; результат около порога неустойчив, общая очередь и запас SLA не учтены.",
    "Измеренная рабочая точка чатовой нагрузки приводится только для Spark + gpt-oss: 96 одновременно активных ответов в среднем по 5,11 т/с; при 128 было 4,52 т/с. Это не гарантия на пользователя, p95 или точная граница ёмкости. Нагрузка — около 1,5 тыс. входных и до 400 выходных токенов, контейнер NVIDIA vLLM 26.03. Переноса по одиночной генерации нет.",
    "В том же тесте плотная Nemotron 49B выдержала 32 потока по 5,08 т/с, но Qwen имеет гибридное внимание, а Laguna — другую MoE-архитектуру. Поэтому даже эта плотная опора на них не переносится.",
    "KV рассчитан для одного потока и FP16/BF16. Нижняя граница освобождает историю за скользящим окном, верхняя хранит полный кэш. Буферы движка, MTP, визуальный проектор, фиксированные состояния и дополнительные одновременные последовательности не включены.",
    "Железо указано в паспортных GB, файлы и KV — в GiB. Для предварительной проверки вместимости GB консервативно переведены по 10^9 / 2^30; память ОС, ECC/резерв платформы и аллокации движка затем не вычтены. Запас менее 5 GiB помечен как пограничный; это редакционный порог, а не гарантия запуска.",
    "Локальные llama-bench могли пересекаться по времени с другой нагрузкой. Их следует читать как нижние границы для конкретного запуска, но не как сопоставимый гарантированный минимум между платформами.",
    "Чтение контекста (prefill) видеокарт, Mac и EPYC не выводится из полосы памяти: без бенчмарка стоит «н/д». Оценка генерации CUDA-карт по полосе показана только как порядок и не участвует в расчёте агентов или чатов.",
    f"Универсального КПД TP нет. На 4 × Spark измерен только профиль {glm_four_spark_summary()}; 213 т/с на 2 × Spark относится к llama.cpp/RPC/IQ1. Это разные профили, они не задают коэффициент ни для сети, ни для PCIe.",
    f"GLM-5.2 не смешивается между чекпойнтами: измерен {GLM_QUANTTRIO.name}, {GLM_QUANTTRIO.parameters}, репозиторий {GLM_QUANTTRIO.repository_gb} ГБ; официальная карточка {GLM_NVIDIA.name} указывает {GLM_NVIDIA.parameters} и около {GLM_NVIDIA.repository_gb} ГБ, но его скорость на 4 × Spark не измерена.",
    "Питание — паспортные TDP/TBP/мощность БП, а не измеренное потребление сборки из розетки. Для модифицированной 4090 48 ГБ значения штатной 24-ГБ карты не гарантированы.",
    "₽/ГБ делит цену всей сборки на наибольшую логическую ёмкость. Для PCIe/сетевого TP она теоретическая до подтверждения запуска. Метрика показывает стоимость вместимости, но не скорость, доступный движку объём или совокупную стоимость владения.",
    "₽/агент и ₽/чат намеренно не выводятся: почти все знаменатели не измерены, а агентный эквивалент не равен числу сотрудников.",
)


OPEN_ITEMS = (
    "чтение контекста, шаг 120k и тест конкурентности на RTX PRO 6000/5000 и конкретной модифицированной RTX 4090 48 ГБ;",
    "TP-бенчмарки той же модели и кванта: отдельно PCIe внутри хоста и сеть 200 Гбит/с между Spark;",
    "фактические буферы и политика SWA/KV каждого выбранного движка, включая несколько одновременных контекстов;",
    "фактически доступная движку память после ОС/ECC/резервов и резидентный размер каждого GGUF/чекпойнта;",
    "профиль реальной длины контекста, доля попаданий в кэш префикса и проверка допущения 20 агентных шагов/ч;",
    "чтение gpt-oss на Spark; нагрузочный прогон Qwen/Laguna на Spark и Halo; gpt-oss на Halo; выбранные модели на Mac;",
    "потребление полных сборок из розетки под длительной LLM-нагрузкой;",
    "КП с точными SKU, НДС, гарантией и сроком: карты, хосты, Halo, Mac, EPYC, кабели и сеть четырёх Spark;",
    "состав (BOM) EPYC и STREAM-замер памяти; паспорт и гарантия модификатора RTX 4090 48 ГБ;",
    "точная ревизия, формат и загруженный объём чекпойнта GLM-5.2 до выбора сети на четыре Spark.",
)


SOURCES = (
    ("исходные замеры стенда", "zamery-src.md"),
    ("основания цен из готового документа", "README-src.md"),
    ("DGX Spark — цена NIX на дату среза", "https://www.nix.ru/autocatalog/other_computers/NVIDIA-DGX-Spark-4TB-940-54242-0006-000-ARM-v92-A-GB10-128-4TbSSD-NVIDIA-Grace-Blackwell-WiFi-BT-DGX-OS_960944.html"),
    ("DGX Spark — спецификация NVIDIA", "https://docs.nvidia.com/dgx/dgx-spark/hardware.html"),
    ("DGX Spark — ConnectX-7 200 Гбит/с", "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"),
    ("RTX PRO 6000 — спецификация NVIDIA", "https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/"),
    ("RTX PRO 5000 — datasheet NVIDIA", "https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf"),
    ("RTX 4090 24 ГБ — штатная спецификация NVIDIA", "https://www.nvidia.com/en-eu/geforce/graphics-cards/40-series/rtx-4090/"),
    ("Ryzen AI Max+ 395 — спецификация AMD", "https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html"),
    ("Mac Studio M3 Ultra — спецификация Apple", "https://support.apple.com/en-us/122211"),
    ("Qwen3.8-27B — карточка модели", "https://huggingface.co/Qwen/Qwen3.8-27B"),
    ("Qwen3.8-27B — конфигурация внимания", "https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json"),
    ("Laguna S 2.1 — карточка модели", "https://huggingface.co/poolside/Laguna-S-2.1"),
    ("Laguna S 2.1 — конфигурация внимания", "https://huggingface.co/poolside/Laguna-S-2.1/blob/main/config.json"),
    ("Laguna S 2.1 Q4_K_M — публичная конверсия GGUF", "https://huggingface.co/AtomicChat/Laguna-S-2.1-GGUF"),
    ("gpt-oss-120b — карточка OpenAI", "https://openai.com/index/gpt-oss-model-card/"),
    ("gpt-oss-120b — официальный чекпойнт", "https://huggingface.co/openai/gpt-oss-120b"),
    ("gpt-oss-120b — конфигурация внимания", "https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json"),
    ("тест конкурентности DGX Spark", "https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/"),
    ("GLM-5.2 на 4 × DGX Spark — публичный профиль одного стенда", "https://github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark"),
    ("точная карточка GLM-бенчмарка 0xdfi", "https://huggingface.co/0xdfi/GLM-5.2-1M-context-NVFP4-4x-DGX-Spark"),
    ("QuantTrio GLM-5.2 Int4/Int8Mix — чекпойнт бенчмарка", "https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix/tree/main"),
    ("GLM-5.2 — официальная базовая модель Z.ai", "https://huggingface.co/zai-org/GLM-5.2"),
    ("GLM-5.2 NVFP4 — официальный чекпойнт NVIDIA", "https://huggingface.co/nvidia/GLM-5.2-NVFP4"),
    ("GLM-5.2 на 2 × DGX Spark — другой llama.cpp/RPC-профиль", "https://forums.developer.nvidia.com/t/academic-glm-5-2-on-2x-dgx-spark-gb10-nodes-crazy-1-bit-ud-iq1-s-rpc-llama-cpp-256k-context-8-tok-s/374523"),
)


def validate() -> None:
    """Встроенные проверки ключевой арифметики и запрета фиктивных данных."""

    qwen_kv = kv_cache_range_gib(MODELS["qwen"])
    laguna_kv = kv_cache_range_gib(MODELS["laguna"])
    gpt_kv = kv_cache_range_gib(MODELS["gptoss"])
    assert math.isclose(qwen_kv[0], 7.32421875, rel_tol=1e-9)
    assert math.isclose(laguna_kv[0], 5.5634765625, rel_tol=1e-9)
    assert math.isclose(laguna_kv[1], 21.97265625, rel_tol=1e-9)
    assert math.isclose(gpt_kv[0], 4.124267578125, rel_tol=1e-9)
    assert math.isclose(gpt_kv[1], 8.23974609375, rel_tol=1e-9)

    expected_prices = (
        2_796_938,
        2_535_000,
        2_576_976,
        1_950_000,
        2_334_352,
        1_197_176,
        2_917_940,
        2_896_850,
        2_270_700,
        835_000,
    )
    assert tuple(build.price_rub for build in BUILDS) == expected_prices
    assert BUILDS[4].unknown_cost
    assert NODES["halo"].price_rub == round(229_000 * 1.15)
    assert CHAT_SWEEP[("spark", "gptoss")].value == 96
    assert math.isclose(NODES["pro5000"].screening_capacity_gib, 67.055225, rel_tol=1e-8)
    assert GLM_FOUR_SPARK.checkpoint is GLM_QUANTTRIO
    assert GLM_FOUR_SPARK.checkpoint is not GLM_NVIDIA
    assert GLM_FOUR_SPARK.generation_context_tokens == 53_000
    assert GLM_FOUR_SPARK.generation_tps == (29, 33)
    assert "NVIDIA" not in glm_four_spark_summary()

    # Граничные размещения не должны незаметно стать подтверждёнными после
    # изменения конфигов моделей или единиц памяти.
    assert not evaluate_part(Part("pro5000", 1), "laguna").viable
    assert not evaluate_part(Part("pro5000", 1), "gptoss").memory_not_borderline
    assert not evaluate_part(Part("pro6000", 1), "laguna").memory_not_borderline
    assert step_seconds(*rates_for("spark", "laguna")) > 180
    assert step_seconds(*rates_for("spark", "qwen")) < 180

    for card in CUDA_CARDS:
        for model_key in MODELS:
            pre, _ = rates_for(card, model_key)
            assert pre.value is None and pre.kind == UNAVAILABLE
            if card == "rtx4090":
                _, gen = rates_for(card, model_key)
                assert gen.value is None and gen.kind == UNAVAILABLE


def render_markdown(rows: Iterable[Build] | None = None) -> str:
    """Вернуть готовый для вставки Markdown-блок с одной сводной таблицей."""

    validate()
    builds = tuple(rows) if rows is not None else BUILDS
    headers = [
        ["Сборка и назначение"],
        ["Цена и бюджет"],
        ["Пулы памяти"],
        ["Питание, паспорт / н/д"],
        _model_header("qwen"),
        _model_header("laguna"),
        _model_header("gptoss"),
    ]
    lines = [
        "## Единая сводная таблица",
        "",
        f"Агентный расчётный сценарий: {_fmt_number(CONTEXT_TOKENS)} токенов контекста, "
        f"{ANSWER_TOKENS} токенов ответа. Измеренная рабочая точка чата использует "
        f"отдельный профиль около 1,5k входных / до 400 выходных токенов. "
        f"Бюджет {_fmt_number(BUDGET_RUB)} ₽. "
        + "Все производительные числа помечены по происхождению.",
        "",
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for build in builds:
        cells = [
            [build.name, build.purpose],
            _cost_cell(build),
            _memory_cell(build),
            [build.power],
            model_cell(build, "qwen"),
            model_cell(build, "laguna"),
            model_cell(build, "gptoss"),
        ]
        lines.append("| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |")

    lines.extend(["", "### Как читать расчёт", ""])
    lines.extend(f"- {note}" for note in METHOD_NOTES)
    lines.extend(["", "### Что требует замера или коммерческого предложения", ""])
    lines.extend(f"- {item}" for item in OPEN_ITEMS)
    lines.extend(["", "### Источники", ""])
    lines.extend(f"- [{label}]({url})" for label, url in SOURCES)
    return "\n".join(lines)


def render_html(rows: Iterable[Build] | None = None) -> str:
    """Вернуть семантический HTML-фрагмент из тех же данных, что и Markdown."""

    validate()
    builds = tuple(rows) if rows is not None else BUILDS
    headers = [
        ["Сборка и назначение"],
        ["Цена и бюджет"],
        ["Пулы памяти"],
        ["Питание, паспорт / н/д"],
        _model_header("qwen"),
        _model_header("laguna"),
        _model_header("gptoss"),
    ]
    out = [
        '<section class="llm-procurement-summary">',
        "<h2>Единая сводная таблица</h2>",
        (
            f"<p>Агентный расчётный сценарий: {_fmt_number(CONTEXT_TOKENS)} токенов контекста, "
            f"{ANSWER_TOKENS} токенов ответа. Измеренная рабочая точка чата использует "
            f"отдельный профиль около 1,5k входных / до 400 выходных токенов. "
            f"Бюджет {_fmt_number(BUDGET_RUB)} ₽. "
            "Все производительные числа помечены по происхождению.</p>"
        ),
        "<table>",
        "<caption>Сборки, бюджет, память и проверяемые метрики трёх моделей</caption>",
        "<thead><tr>" + "".join(_html_cell(header, True) for header in headers) + "</tr></thead>",
        "<tbody>",
    ]
    for build in builds:
        cells = [
            [build.name, build.purpose],
            _cost_cell(build),
            _memory_cell(build),
            [build.power],
            model_cell(build, "qwen"),
            model_cell(build, "laguna"),
            model_cell(build, "gptoss"),
        ]
        out.append("<tr>" + "".join(_html_cell(cell) for cell in cells) + "</tr>")
    out.extend(["</tbody>", "</table>", "<h3>Как читать расчёт</h3>", "<ul>"])
    out.extend(f"<li>{html.escape(note)}</li>" for note in METHOD_NOTES)
    out.extend(["</ul>", "<h3>Что требует замера или коммерческого предложения</h3>", "<ul>"])
    out.extend(f"<li>{html.escape(item)}</li>" for item in OPEN_ITEMS)
    out.extend(["</ul>", "<h3>Источники</h3>", "<ul>"])
    for label, url in SOURCES:
        out.append(
            f'<li><a href="{html.escape(url, quote=True)}">{html.escape(label)}</a></li>'
        )
    out.extend(["</ul>", "</section>"])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "html"),
        default="markdown",
        help="формат вывода; по умолчанию markdown",
    )
    args = parser.parse_args()
    print(render_html() if args.format == "html" else render_markdown())


if __name__ == "__main__":
    main()
