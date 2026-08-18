# -*- coding: utf-8 -*-
"""Сводная таблица закупки: три режима работы с порогами комфорта.

Каждое число помечено происхождением: [З] собственный замер, [В] внешний
замер точной пары модель/платформа, [Р] расчёт. Расчёт генерации — масштаб
по полосе памяти от нашего замера на Spark; расчёт чтения контекста у карт
опирается на соотношение вычислителей и является самым слабым местом.
"""
BUDGET = 3_000_000

# ── ПОРОГИ КОМФОРТА. Задаются здесь, а не подгоняются под результат.
COPILOT = dict(fresh=100, out=20,   limit=0.5,  per_hour=120)   # ≤0,5 с на подсказку
AGENT   = dict(fresh=120_000, out=200, limit=60, per_hour=20)   # ≤60 с на шаг
CHAT    = dict(fresh=2_000, out=500, min_tps=5.0, per_hour=30)  # ≥5 т/с на человека

NODE = {
 'pro6000':dict(n='RTX PRO 6000',      c=1_550_000, mem=96,  bw=1792),
 'pro5000':dict(n='RTX PRO 5000',      c=979_800,   mem=72,  bw=1344),
 'rtx4090':dict(n='RTX 4090 48 ГБ',    c=450_000,   mem=48,  bw=1008),
 'spark':  dict(n='DGX Spark',         c=583_588,   mem=128, bw=273),
 'halo':   dict(n='мини-ПК Strix Halo',c=263_000,   mem=128, bw=256),
 'mac':    dict(n='Mac Studio M3 Ultra',c=756_900,  mem=256, bw=819),
 'epyc':   dict(n='EPYC 768 ГБ',       c=835_000,   mem=768, bw=350),
}
HOST1,HOST4,CABLE = 400_000,735_000,30_000

# веса + KV на 120к (диапазон по codex) → верхняя граница потребности
MODEL = {
 'coder': ('Qwen3-Coder-30B-A3B Q6', 24.53, 30.0, 'MoE'),
 'qwen':  ('Qwen3.8-27B Q4',         15.66, 23.0, 'dense'),
 'laguna':('Laguna S 2.1 Q4',        66.28, 88.3, 'MoE'),
 'gptoss':('gpt-oss-120b MXFP4',     60.80, 69.0, 'MoE'),
}

# замеры: (чтение, генерация). None = нет
M = {
 'spark': {'coder':(2004.6,69.0),'qwen':(835.6,12.34),'laguna':(683.5,30.19),'gptoss':(None,33.53)},
 'halo':  {'coder':(None,None),  'qwen':(214.3,12.76),'laguna':(353.8,40.00),'gptoss':(None,None)},
}
SRC = {('spark','gptoss'):'В'}                    # внешний замер
PRE_CARD = {'pro6000':8000,'pro5000':6000,'rtx4090':2273,'mac':180,'epyc':250}
KIN = {k:v[3] for k,v in MODEL.items()}

def _sib(k,m,idx):
    for m2,v in M.get(k,{}).items():
        if m2!=m and v[idx] is not None and KIN[m2]==KIN[m]: return v[idx]
    return None

def rate(k,m):
    """(чтение, генерация, метка)"""
    v = M.get(k,{}).get(m,(None,None))
    tag = SRC.get((k,m),'З')
    pre,gen = v
    if gen is None:
        gen = M['spark'][m][1] * NODE[k]['bw']/NODE['spark']['bw']; tag='Р'
    if pre is None:
        pre = PRE_CARD.get(k) or _sib(k,m,0) or M['spark']['laguna'][0]*NODE[k]['bw']/NODE['spark']['bw']
        tag = 'Р' if tag!='З' else 'Р'
    return pre,gen,tag

def fits(k,m,q=1,pool=False):  return MODEL[m][2] <= NODE[k]['mem']*(q if pool else 1)

TP_EFF = 0.85   # КПД тензорного параллелизма; замером не подтверждён

def mode(k,m,cfg,tp=1):
    """(время_запроса_с, комфортно_ли, пользователей_на_группу)"""
    pre,gen,_ = rate(k,m)
    if tp > 1: pre, gen = pre*tp*TP_EFF, gen*tp*TP_EFF
    t = cfg['fresh']/pre + cfg['out']/gen
    if 'limit' in cfg:  ok = t <= cfg['limit']
    else:               ok = gen >= cfg['min_tps']
    return t, ok, (3600/t)/cfg['per_hour']

BUILDS = [
 ('RTX PRO 6000 + хост + Spark + Halo',[('pro6000',1,0),('spark',1,0),('halo',1,0)],HOST1),
 ('4 × RTX 4090 48 ГБ + хост',         [('rtx4090',4,1)],                           HOST4),
 ('RTX PRO 5000 + хост + 2 Spark',     [('pro5000',1,0),('spark',2,1)],             HOST1+CABLE),
 ('RTX PRO 6000 + хост',               [('pro6000',1,0)],                           HOST1),
 ('4 × DGX Spark в кластере',          [('spark',4,1)],                             CABLE*3),
 ('5 × DGX Spark независимыми',        [('spark',5,0)],                             0),
 ('11 × мини-ПК Strix Halo',           [('halo',11,0)],                             0),
 ('3 × Mac Studio M3 Ultra',           [('mac',3,0)],                               0),
 ('EPYC 768 ГБ без карты',             [('epyc',1,0)],                              0),
]

def build(name,parts,extra):
    cena=sum(NODE[k]['c']*q for k,q,_ in parts)+extra
    maxp=max(NODE[k]['mem']*(q if p else 1) for k,q,p in parts)
    r=dict(name=name,cena=cena,ost=BUDGET-cena,
           mem=sum(NODE[k]['mem']*q for k,q,_ in parts),maxp=maxp)
    for mk,cfg in (('copilot',COPILOT),('agent',AGENT),('chat',CHAT)):
        for m in MODEL:
            tot=0.0; ok_any=False; best=None
            for k,q,p in parts:
                if not fits(k,m,q,p): continue
                pooled = p and not fits(k,m,1,False)
                n = 1 if pooled else q
                t,ok,u = mode(k,m,cfg, q if pooled else 1)
                tot+=u*n; ok_any = ok_any or ok
                best = t if best is None else min(best,t)
            r[(mk,m)] = (best,ok_any,tot)
    return r

ROWS=[build(*b) for b in BUILDS]
def f(x): return f'{x:,.0f}'.replace(',',' ')

if __name__=='__main__':
    print(f'ПОРОГИ: копилот ≤{COPILOT["limit"]} с на подсказку · '
          f'агент ≤{AGENT["limit"]} с на шаг · чат ≥{CHAT["min_tps"]:.0f} т/с на человека\n')
    for mk,label,m in (('copilot','КОПИЛОТ  (Qwen3-Coder-30B-A3B)','coder'),
                       ('agent','АГЕНТ    (Laguna S 2.1)','laguna'),
                       ('chat','ЧАТ      (Laguna S 2.1)','laguna')):
        print(f'== {label}')
        for r in ROWS:
            t,ok,u = r[(mk,m)]
            if t is None: print(f'   {r["name"]:38} не влезает'); continue
            mark = 'комфортно' if ok else 'НЕ комфортно'
            print(f'   {r["name"]:38} {t:7.2f} с  {mark:12} {u:6.0f} чел.')
        print()

# ──────────────────────────── РЕНДЕР ────────────────────────────
MODES = (('copilot','Копилот','coder','≤0,5 с на подсказку'),
         ('agent','Агент','laguna','≤60 с на шаг'),
         ('chat','Чат','laguna','≥5 т/с на человека'))

def cell(r,mk,m):
    t,ok,u = r[(mk,m)]
    if t is None: return '—', 'не влезает'
    znak = '' if ok else '⚠ '
    tt = f'{t:.2f} с' if t < 10 else (f'{t:.0f} с' if t < 120 else f'{t/60:.1f} мин')
    return f'{znak}{u:.0f}', tt

def markdown():
    L=['| Сборка | Цена | Остаток | Крупнейший пул | ' +
       ' | '.join(f'{n}<br><span title="{th}">{th}</span>' for _,n,_,th in MODES) + ' |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in ROWS:
        cells=[]
        for mk,_,m,_ in MODES:
            u,tt = cell(r,mk,m)
            cells.append(f'**{u}**<br><span>{tt}</span>' if u!='—' else '—')
        L.append(f'| {r["name"]} | {f(r["cena"])} ₽ | {f(r["ost"])} ₽ | {r["maxp"]} ГБ | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L)

def html():
    L=['<div class="card scroller">','  <table>','    <thead><tr><th>Сборка</th><th class="r">Цена</th>'
       '<th class="r">Остаток</th><th class="r">Крупнейший<br>пул</th>' +
       ''.join(f'<th class="r">{n}<br><span class="dim">{th}</span></th>' for _,n,_,th in MODES) +
       '</tr></thead>','    <tbody>']
    for r in ROWS:
        tds=[]
        for mk,_,m,_ in MODES:
            u,tt = cell(r,mk,m)
            if u=='—': tds.append('<td class="r dim">не влезает</td>'); continue
            warn = u.startswith('⚠')
            num = u.lstrip('⚠ ')
            cls = 'stop' if warn else 'ok'
            tds.append(f'<td class="r mono {cls}"><strong>{num}</strong>'
                       f'<br><span class="dim">{tt}{" · не комфортно" if warn else ""}</span></td>')
        L.append(f'      <tr><td>{r["name"]}</td><td class="r mono">{f(r["cena"])} ₽</td>'
                 f'<td class="r mono">{f(r["ost"])} ₽</td><td class="r mono">{r["maxp"]} ГБ</td>'
                 + ''.join(tds) + '</tr>')
    L += ['    </tbody>','  </table>','</div>']
    return '\n'.join(L)
