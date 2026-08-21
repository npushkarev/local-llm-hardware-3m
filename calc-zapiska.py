# -*- coding: utf-8 -*-
"""Расчёт для записки на полную сборку (4 карты, 288 ГБ).
[П] прайс serverflow.ru 21.08.2026 · [В] замер из обзора ServerFlow 20.08.2026
[С] Хабр Карьера H1'2026 + НК РФ ст.427 · [Р] расчёт из перечисленного."""

# ── СПЕЦИФИКАЦИЯ [П]
ITEMS = [
 ("ASUS ESC4000A-E12: шасси 2U, 2×2600 Вт, рельсы, BMC", 498_700 - 159_700),
 ("AMD EPYC 9374F, 32 ядра / 64 потока, до 4,3 ГГц",     224_900),
 ("ОЗУ 4 × 32 ГБ DDR5 ECC RDIMM 4800",                   4 * 99_800),
 ("NVMe Intel DC-P4510 4 ТБ (под несколько моделей)",     66_800),
 ("NVIDIA RTX PRO 5000 Blackwell 72 ГБ × 4",             4 * 979_800),
]
TOTAL   = sum(c for _, c in ITEMS)
VRAM    = 4 * 72
PRO6000 = 1_900_000              # [В] розница РФ 1,8–2,0 млн, середина
ALT     = TOTAL - 4*979_800 + 2*PRO6000
ALT_VRAM = 2 * 96

# ── МОДЕЛЬ [В]
WEIGHTS = 167                    # ГБ, DeepSeek V4-Flash-0731, 284 млрд / 13 млрд активных
FREE_OUR = VRAM - WEIGHTS
FREE_ALT = ALT_VRAM - WEIGHTS

# ── ЗАМЕРЫ [В] 4 карты, vLLM + патч + спекулятивное декодирование
PREFILL = 190_000 / 21           # 190к токенов промпта за 21 с
DECODE  = 2_741                  # суммарно т/с при 32 параллельных
PER_CLI = 86                     # т/с на клиента при 32 параллельных
IDLE_W, LOAD_W, PEAK_W = 380, 1_300, 2_240
NOISE_IDLE, NOISE_LOAD = 75, 83
GPU_TEMP_INFER, GPU_W_INFER = (50, 60), 200

# ── ПРОФИЛИ НАГРУЗКИ
PROF = {'копайлот': (120, 100, 50), 'чат': (30, 2_000, 500), 'агент': (20, 120_000, 200)}
PEAK = 3.0

def cap(pre_tps, dec_tps):
    out = {}
    for n,(ph,ctx,o) in PROF.items():
        need_p, need_d = ph*ctx/3600, ph*o/3600
        sim = min(pre_tps/need_p, dec_tps/need_d)
        out[n] = (sim, sim/PEAK, 'чтение' if pre_tps/need_p < dec_tps/need_d else 'генерация')
    return out
CAP = cap(PREFILL, DECODE)

# агенты: сессия либо идёт, либо нет. Доля рабочего дня под агентом:
AGENT_DUTY = (1.0/8, 2.0/8)      # 1–2 часа из 8

# ── СТАВКА [С]
SAL_M, LIM = 270_000, 2_979_000
SAL_Y = SAL_M*12
def payroll(a,b): return SAL_Y + min(SAL_Y,LIM)*a + max(0,SAL_Y-LIM)*b
ST_IT, ST_GEN = payroll(.15,.076), payroll(.30,.151)

# ── ЭКСПЛУАТАЦИЯ
LIFE, PUE, TARIFF = 4, 1.5, 7.0
ELEC = LOAD_W/1000 * PUE * 8760 * TARIFF

# ── ОБЛАКО ДЛЯ СРАВНЕНИЯ
SUB_M = 3_000                    # ₽/чел/мес, середина 2–4 тыс

def m(x): return f"{x:,.0f}".replace(',',' ')

if __name__ == '__main__':
    print("── СПЕЦИФИКАЦИЯ")
    for n,c in ITEMS: print(f"  {n:52} {m(c):>10} ₽")
    print(f"  {'ИТОГО':52} {m(TOTAL):>10} ₽   ({VRAM} ГБ видеопамяти)")
    print(f"\n  Альтернатива 2× RTX PRO 6000:              {m(ALT):>10} ₽   ({ALT_VRAM} ГБ)")
    print(f"\n── ПАМЯТЬ ПОД РАБОТУ (модель весит {WEIGHTS} ГБ)")
    print(f"  наш вариант  {VRAM} ГБ − {WEIGHTS} = {FREE_OUR} ГБ свободно")
    print(f"  альтернатива {ALT_VRAM} ГБ − {WEIGHTS} = {FREE_ALT} ГБ свободно  → в {FREE_OUR/FREE_ALT:.1f} раза меньше")
    print(f"\n── ЁМКОСТЬ [В] чтение {PREFILL:.0f} т/с, генерация {DECODE} т/с, {PER_CLI} т/с на клиента")
    for n,(s,p,l) in CAP.items():
        print(f"  {n:9} одновременно {s:7.0f} | сотрудников {p:6.0f} | упор в {l}")
    a = CAP['агент'][0]
    print(f"  агент при 1–2 ч работы из 8: {a/AGENT_DUTY[1]:.0f}–{a/AGENT_DUTY[0]:.0f} разработчиков")
    print(f"\n── СТАВКА (год, полная стоимость работодателю)")
    print(f"  IT-аккредитация 15%/7,6%   {m(ST_IT):>10} ₽")
    print(f"  общий тариф 30%/15,1%      {m(ST_GEN):>10} ₽")
    print(f"  сервер = {TOTAL/ST_IT:.2f} ставки | две ставки покрывают с остатком {m(2*ST_IT-TOTAL)} ₽")
    print(f"  в год при {LIFE}-летнем сроке: {m(TOTAL/LIFE)} ₽ = {TOTAL/LIFE/ST_IT:.2f} ставки")
    print(f"  электричество+охлаждение:   {m(ELEC)} ₽/год = {ELEC/ST_IT*100:.1f}% ставки")
    print(f"\n── ОБЛАКО ПРОТИВ СЕРВЕРА ({m(SUB_M)} ₽/чел/мес)")
    for ppl in (50,100,200):
        y = ppl*SUB_M*12
        print(f"  {ppl:4} чел: облако {m(y):>10} ₽/год | сервер окупается за {TOTAL/y*12:.1f} мес")
