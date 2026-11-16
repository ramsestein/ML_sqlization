"""Entrenar un gradient boosting (el algoritmo de XGBoost con histogramas) con consultas SQL en FakeNex.

Sin permiso de escritura y sin descargar filas de pacientes: el estado del entrenamiento (cortes de cada variable y
árboles ya entrenados) viaja DENTRO de cada consulta como arrays binarios float32 (binario.py), igual que en la
inferencia, y lo único que sale del servidor son sumas agregadas por nodo.

Ciclo:
  0. una consulta: ingresos, tasa base y cortes de cada variable (approx_percentile en float32 de los ingresos de
     entrenamiento; como max_bin de XGBoost).
  1. por cada árbol y nivel, UNA consulta que, en el servidor:
       recalcula las variables desde las tablas crudas (contrato.sql_variables, las mismas que en la inferencia),
       aplica el modelo entrenado hasta ahora (arrays binarios; hojas sumadas en real, como XGBoost),
       compara con lo que pasó a cada ingreso: g = p - y, h = p (1 - p) (pérdida logística),
       asigna cada ingreso a su nodo del árbol en curso y hace histogramas (nodo, variable, intervalo),
       sumas acumuladas con una ventana y ganancia de cada corte, con los ausentes a cada lado:
         ganancia = GL^2 / (HL + lambda) + GR^2 / (HR + lambda) - G^2 / (H + lambda)
       y devuelve una fila por nodo: el mejor corte y sus sumas, más la pérdida de entrenamiento y la de prueba.
     Las hojas salen de las sumas del corte que las crea (-eta G / (H + lambda)): `profundidad` consultas por árbol.
  2. una consulta final: la pérdida del modelo completo.

Cada CTE se usa una sola vez (cadena lineal): Trino no materializa los CTE y, si los histogramas se usaran dos veces,
recalcularía las variables (medido en FakeNex: de 25-55 s a 6-10 s por consulta).

Empates: el mejor corte es el máximo de (ganancia redondeada a 9 decimales, -variable, -intervalo, ausentes a la izquierda),
igual en SQL y en local; así el orden (no determinista) de las sumas de Trino no decide entre cortes equivalentes.

Partición entrenamiento / prueba por paciente con un hash de patient_ref, el mismo en SQL y en Python (es_prueba).
`entrenar_local` hace exactamente las mismas cuentas con numpy sobre los datos descargados: sirve para comprobar que el
SQL calcula lo previsto. El modelo se exporta como documento if-then (núcleo de árboles float32, como XGBoost; lo
compila sql.py) y como JSON de XGBoost (lo carga la librería y lo evalúa `fnxsql evaluar`).
"""
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import binario as B, cliente, config as C, contrato as K, ir
from .binario import f32
from .contrato import lit

HASH_MULT, HASH_MOD = 2654435761, 2 ** 32  # hash multiplicativo de Knuth: patient_ref -> [0, 2^32)
EPS_GANANCIA = 1e-6  # kRtEps de XGBoost: un corte tiene que mejorar más que esto
DECIMALES_EMPATE = 9


@dataclass
class Parametros:
    arboles: int = 100       # n_estimators
    profundidad: int = 3     # max_depth
    eta: float = 0.1         # learning_rate
    lambda_: float = 1.0     # reg_lambda
    min_hijo: float = 1.0    # min_child_weight (suma de hessianos en cada hijo)
    min_ingresos: int = 0    # mínimo de ingresos en cada hijo (0 = como XGBoost); sube el tamaño mínimo de cada agregado
    bins: int = 256          # max_bin: hasta bins - 1 cortes por variable
    prueba: float = 0.2      # fracción de pacientes de prueba (hash de patient_ref)
    paciencia: int = 0       # parada temprana: árboles sin mejorar la pérdida de prueba (0 = no)


def corte_prueba(p):
    return int(round(p.prueba * HASH_MOD))


def es_prueba(pids, p):
    """Mismo criterio que el SQL: mod(patient_ref * 2654435761, 2^32) < prueba * 2^32."""
    pids = np.asarray(pids, dtype=np.int64)
    assert pids.max(initial=0) < 2 ** 33, "patient_ref demasiado grande para el hash en int64"
    return (pids * HASH_MULT) % HASH_MOD < corte_prueba(p)


def margen_base(positivos, n):
    """(tasa base en float32, logit en float32) como XGBoost con base_score = tasa de entrenamiento."""
    b = f32(positivos / n)
    return b, f32(math.log(b / (1 - b)))


def _real(x):
    return "REAL '" + repr(float(np.float32(x))) + "'"


# ------------------------------------------------------------------ SQL
def sql_base(variables, contrato, p):
    """CTEs `vars`, `etiqueta` y `t` (variables + y + prueba) de toda la base original de FakeNex."""
    return (K.sql_variables(variables, contrato, None) + ",\n" + K.sql_etiqueta(None) + ",\n"
            "t AS (  -- y = muerte en el ingreso; prueba = paciente de prueba (hash de patient_ref)\n"
            f"  SELECT v.*, CAST(coalesce(e.muere, 0) AS double) AS y,\n"
            f"         mod(v.patient_ref * {HASH_MULT}, {HASH_MOD}) < {corte_prueba(p)} AS prueba\n"
            "  FROM vars v LEFT JOIN etiqueta e ON e.ep_et = v.episode_ref\n)")


def sql_inicio(variables, contrato, p):
    """Ingresos, positivos y cortes (cuantiles i / bins en float32) de cada variable, solo con los de entrenamiento."""
    qs = f"transform(sequence(1, {p.bins - 1}), i -> i / {p.bins}E0)"
    cols = ",\n       ".join(f"approx_percentile(CAST(CAST(v{i:03d} AS real) AS double), {qs}) FILTER (WHERE NOT prueba) "
                             f"AS q{i:03d}" for i in range(len(variables)))
    return (sql_base(variables, contrato, p) + "\nSELECT count(*) FILTER (WHERE NOT prueba) AS n, "
            "count(*) FILTER (WHERE prueba) AS n_prueba, sum(y) FILTER (WHERE NOT prueba) AS positivos,\n"
            f"       count(DISTINCT patient_ref) AS pacientes,\n       {cols}\nFROM t")


def cortes_de(valores):
    """Cortes float32 únicos y ordenados de una variable (lista vacía si no tiene datos)."""
    return sorted({f32(v) for v in valores if v is not None and not math.isnan(v)}) if valores is not None else []


def sql_cortes(cortes):
    """CTE `cortes` (una fila): c = array de arrays double con los cortes float32 de cada variable, desde UN binario.
    width_bucket no admite arrays vacíos: una variable sin cortes lleva uno de relleno y no se puede cortar (kv = 0)."""
    trozos = [c or [0.0] for c in cortes]
    todos = [x for c in trozos for x in c]
    offs = np.cumsum([0] + [len(c) for c in trozos])[:-1].tolist()
    return (f"cortes AS (  -- cortes float32 de cada variable ({len(todos)} en binario)\n"
            f"  SELECT transform(sequence(1, {len(cortes)}), j -> slice(c, element_at(o, j) + 1, element_at(n, j))) AS c\n"
            f"  FROM (SELECT transform({B.vector('bc', len(todos), 'f4')}, z -> CAST(z AS double)) AS c,\n"
            f"               ARRAY[{', '.join(map(str, offs))}] AS o, ARRAY[{', '.join(str(len(c)) for c in trozos)}] AS n\n"
            f"        FROM (SELECT from_base64({B.bin_float(todos, 4)}) AS bc))\n)")


def sql_nodo(parcial, frontera, n=0):
    """Nodo del árbol en curso (en anchura: hijos 2n+1 y 2n+2) al que llega cada ingreso, desde sus intervalos `bb`
    (-1 = ausente); NULL si llega a una hoja ya cerrada. Corte (j, b): a la izquierda si intervalo <= b (x < corte b)."""
    if n in frontera:
        return str(n)
    x = parcial.get(n)
    if x is None:
        return "NULL"
    bj = f"element_at(bb, {x['j'] + 1})"
    cond = f"{bj} <= {x['b']}" if x["izq"] else f"{bj} BETWEEN 0 AND {x['b']}"  # ausente = -1
    return f"(CASE WHEN {cond} THEN {sql_nodo(parcial, frontera, 2 * n + 1)} ELSE {sql_nodo(parcial, frontera, 2 * n + 2)} END)"


PASO = """,
q AS (SELECT y, prueba, ARRAY[{x}] AS x FROM t),
s0 AS (  -- margen del modelo hasta ahora (hojas sumadas en real) e intervalo de cada variable (-1 = ausente)
  SELECT q.y, q.prueba, CAST({f} AS double) AS f,
         transform(sequence(1, {J}), j -> coalesce(width_bucket(CAST(element_at(q.x, j) AS double), element_at(k.c, j)), -1)) AS bb
  FROM q{junta} CROSS JOIN cortes k
),
s AS (SELECT y, prueba, f, bb, {nodo} AS nodo FROM s0),
gh AS (  -- gradiente, hessiano y pérdida; los de prueba van todos al nodo -1 con un solo intervalo (solo cuentan su pérdida)
  SELECT if(prueba, -1, nodo) AS nodo, if(prueba, ARRAY[BIGINT '0'], bb) AS b, p - y AS g, p * (1E0 - p) AS h,
         if(f > 0, f + ln(1E0 + exp(-f)), ln(1E0 + exp(f))) - y * f AS l
  FROM (SELECT y, prueba, f, bb, nodo, 1E0 / (1E0 + exp(-f)) AS p FROM s)
  WHERE prueba OR nodo IS NOT NULL
),
hist AS (SELECT nodo, j, bin, sum(g) AS g, sum(h) AS h, count(*) AS n, sum(l) AS l
         FROM gh CROSS JOIN UNNEST(b) WITH ORDINALITY AS u(bin, j) GROUP BY 1, 2, 3),
win AS (  -- totales del nodo, de sus ausentes y acumulados por intervalo (una sola lectura de hist)
  SELECT nodo, j, bin, l,
         sum(g) OVER p AS gt, sum(h) OVER p AS ht, sum(n) OVER p AS nt,
         sum(if(bin = -1, g, 0E0)) OVER p AS gm, sum(if(bin = -1, h, 0E0)) OVER p AS hm, sum(if(bin = -1, n, 0)) OVER p AS nm,
         sum(if(bin >= 0, g, 0E0)) OVER a AS ga, sum(if(bin >= 0, h, 0E0)) OVER a AS ha, sum(if(bin >= 0, n, 0)) OVER a AS na
  FROM hist WINDOW p AS (PARTITION BY nodo, j), a AS (PARTITION BY nodo, j ORDER BY bin ROWS UNBOUNDED PRECEDING)
),
cand AS (  -- corte "x < corte[bin]" con los ausentes a la izquierda (izq) o a la derecha
  SELECT w.*, m.izq, w.nodo >= 0 AND w.bin >= 0 AND w.bin < element_at(ARRAY[{kv}], CAST(w.j AS integer)) AS corte,
         w.ga + if(m.izq, w.gm, 0E0) AS gl, w.ha + if(m.izq, w.hm, 0E0) AS hl, w.na + if(m.izq, w.nm, 0) AS nl
  FROM win w CROSS JOIN UNNEST(ARRAY[true, false]) AS m(izq)
),
gan AS (
  SELECT *, if(corte AND hl >= {mh} AND ht - hl >= {mh} AND nl >= {mn} AND nt - nl >= {mn},
               gl * gl / (hl + {lam}) + (gt - gl) * (gt - gl) / (ht - hl + {lam}) - gt * gt / (ht + {lam})) AS ganancia
  FROM cand
)
SELECT nodo, min_by(gt, j) AS gt, min_by(ht, j) AS ht, min_by(nt, j) AS nt, sum(if(j = 1 AND izq, l, 0E0)) AS perdida,
       max_by(ROW(j, bin, izq, gl, hl, nl, ganancia),
              if(ganancia IS NULL, NULL, ROW(round(ganancia, {dec}), -j, -bin, izq))) AS r
FROM gan GROUP BY nodo ORDER BY nodo"""


def sql_paso(base, cortes_cte, kv, nombres, arboles, margen, parcial, frontera, p):
    """Consulta de un nivel del árbol en curso: una fila por nodo con (nodo, gt, ht, nt, perdida, r)."""
    J = len(nombres)
    if arboles:
        cte, rec, _ = B.sql_arboles({"precision": "float32", "arboles": arboles}, {n: i + 1 for i, n in enumerate(nombres)},
                                    "float32", "<")
        f, junta = f"reduce({rec}, {_real(margen)}, (s, h) -> s + h, s -> s)", " CROSS JOIN arboles a"
    else:
        cte, f, junta = "", _real(margen), ""
    x = ", ".join(f"CAST(v{i:03d} AS real)" for i in range(J))
    return (base + ",\n" + cte + cortes_cte + PASO.format(
        x=x, f=f, J=J, junta=junta, nodo=sql_nodo(parcial, set(frontera)), kv=", ".join(map(str, kv)),
        mh=lit(p.min_hijo), mn=int(p.min_ingresos), lam=lit(p.lambda_), dec=DECIMALES_EMPATE))


# ------------------------------------------------------------------ bucle común (servidor o local)
def _hoja(sumas, p):
    G, H, N = sumas
    return {"hoja": f32(-p.eta * G / (H + p.lambda_)), "info": {"hessiano": H, "ingresos": int(N)}}


def _arbol(parcial, sumas, cortes, nombres, p, n=0):
    x = parcial.get(n)
    if x is None:
        return _hoja(sumas[n], p)
    G, H, N = sumas[n]
    return {"si": {"variable": nombres[x["j"]], "op": "<", "valor": cortes[x["j"]][x["b"]], "nulo_a_entonces": x["izq"]},
            "entonces": _arbol(parcial, sumas, cortes, nombres, p, 2 * n + 1),
            "si_no": _arbol(parcial, sumas, cortes, nombres, p, 2 * n + 2),
            "info": {"ganancia": x["ganancia"], "hessiano": H, "ingresos": int(N), "peso": f32(-p.eta * G / (H + p.lambda_))}}


def _aviso(*a):
    print(*a, flush=True)


def crecer(evaluar_nivel, cortes, nombres, p, aviso=_aviso):
    """Bucle de boosting. evaluar_nivel(arboles, parcial, frontera) -> {nodo: {"gt", "ht", "nt", "perdida", "r"}} (nodo -1:
    los de prueba). Devuelve (árboles en formato if-then, historial de pérdidas por número de árboles)."""
    arboles, historial, mejor = [], [], (math.inf, 0)
    for t in range(p.arboles + 1):  # la última vuelta solo mide la pérdida del modelo completo
        parcial, sumas, frontera = {}, {}, [0]
        for nivel in range(p.profundidad):
            filas = evaluar_nivel(arboles, parcial, frontera)
            if nivel == 0:
                tr, te = filas[0], filas.get(-1)
                h = {"arboles": t, "perdida_entrenamiento": tr["perdida"] / tr["nt"],
                     "perdida_prueba": te["perdida"] / te["nt"] if te else None}
                historial.append(h)
                aviso(f"  {t:4d} árboles: pérdida entrenamiento {h['perdida_entrenamiento']:.5f}"
                      + (f", prueba {h['perdida_prueba']:.5f}" if te else ""))
                if te and h["perdida_prueba"] < mejor[0]:
                    mejor = (h["perdida_prueba"], t)
                if t == p.arboles or (p.paciencia and te and t - mejor[1] >= p.paciencia):
                    if p.paciencia and t != p.arboles:
                        aviso(f"  parada temprana: el mejor fue con {mejor[1]} árboles")
                        arboles = arboles[:mejor[1]]
                    return arboles, historial
                sumas[0] = (tr["gt"], tr["ht"], tr["nt"])
            nueva = []
            for n in frontera:
                fila = filas.get(n)
                r = fila and fila["r"]
                if r is not None and r[6] is not None and float(r[6]) > EPS_GANANCIA:
                    j, b, izq, gl, hl, nl, gan = int(r[0]) - 1, int(r[1]), bool(r[2]), float(r[3]), float(r[4]), int(r[5]), float(r[6])
                    parcial[n] = {"j": j, "b": b, "izq": izq, "ganancia": gan}
                    sumas[2 * n + 1] = (gl, hl, nl)
                    sumas[2 * n + 2] = (fila["gt"] - gl, fila["ht"] - hl, fila["nt"] - nl)
                    nueva += [2 * n + 1, 2 * n + 2]
            frontera = nueva
            if not frontera:
                break
        arboles.append(_arbol(parcial, sumas, cortes, nombres, p))
    return arboles, historial


# ------------------------------------------------------------------ entrenamiento en el servidor
def entrenar(nombres, contrato=None, p=None, forzar=False, aviso=_aviso):
    """Entrena con consultas a FakeNex. Devuelve dict con el documento if-then, el JSON de XGBoost, los cortes, el
    historial de pérdidas y el registro de cada consulta (tiempos, bytes, filas devueltas)."""
    contrato, p = contrato or K.Contrato(), p or Parametros()
    variables, errores = K.variables_del_modelo(nombres, contrato)
    if errores:
        raise ValueError("; ".join(errores))
    registro, t0 = [], time.time()

    def ejecutar(sql, que):
        nb = len(sql.encode("utf-8"))
        if nb > C.MAX_SQL_BYTES and not forzar:
            raise ValueError(f"consulta de {nb:,} bytes: no cabe en AWS Athena ({C.MAX_SQL_BYTES:,})")
        df, reloj, st = cliente.ejecutar(sql)
        registro.append({"que": que, "bytes_sql": nb, "filas_devueltas": len(df), **cliente.resumen_stats(st, reloj)})
        return df

    base = sql_base(variables, contrato, p)
    d = ejecutar(sql_inicio(variables, contrato, p), "inicio")
    n, positivos = int(d.n[0]), float(d.positivos[0])
    cortes = [cortes_de(d[f"q{i:03d}"][0]) for i in range(len(nombres))]
    tasa, margen = margen_base(positivos, n)
    aviso(f"  {n} ingresos de entrenamiento ({positivos:.0f} positivos, tasa {tasa:.4f}), {int(d.n_prueba[0])} de prueba; "
          f"{sum(map(len, cortes))} cortes; {registro[-1]['reloj_s']} s")
    ccte, kv = sql_cortes(cortes), [len(c) for c in cortes]

    def evaluar_nivel(arboles, parcial, frontera):
        sql = sql_paso(base, ccte, kv, nombres, arboles, margen, parcial, frontera, p)
        df = ejecutar(sql, f"árbol {len(arboles)} nivel {(min(frontera) + 1).bit_length() - 1}")
        registro[-1].update(arboles=len(arboles), nodos=len(frontera))
        return {int(r.nodo): {"gt": float(r.gt), "ht": float(r.ht), "nt": int(r.nt), "perdida": float(r.perdida), "r": r.r}
                for r in df.itertuples()}

    arboles, historial = crecer(evaluar_nivel, cortes, nombres, p, aviso)
    doc = documento(arboles, margen, variables, contrato, "boosting_sql", p)
    return {"documento": doc, "xgboost": a_xgboost(doc, tasa), "cortes": cortes, "tasa": tasa, "margen": margen,
            "historial": historial, "consultas": registro, "segundos": time.time() - t0,
            "ingresos": {"entrenamiento": n, "prueba": int(d.n_prueba[0]), "pacientes": int(d.pacientes[0])}}


# ------------------------------------------------------------------ el mismo algoritmo en local (verificación)
def intervalos(X, cortes):
    """Intervalo de cada valor como width_bucket de Trino sobre float32(x): nº de cortes <= x; -1 si es ausente."""
    X32 = np.asarray(X, dtype=np.float64).astype(np.float32).astype(np.float64)
    out = np.empty(X32.shape, dtype=np.int64)
    for j, c in enumerate(cortes):
        c = np.array(c or [0.0])
        out[:, j] = np.where(np.isnan(X32[:, j]), -1, np.searchsorted(c, np.nan_to_num(X32[:, j]), side="right"))
    return out


def recorrer(arbol, X, nombres):
    """Hoja (float32) de cada fila de X (numpy) en un árbol del documento: float32(x) < corte, ausente según la rama."""
    pos = {n: i for i, n in enumerate(nombres)}
    X32 = np.asarray(X, dtype=np.float64).astype(np.float32)
    out = np.empty(len(X32), dtype=np.float32)

    def rec(nodo, idx):
        if "hoja" in nodo:
            out[idx] = np.float32(nodo["hoja"])
            return
        c = nodo["si"]
        x = X32[idx, pos[c["variable"]]]
        izq = np.where(np.isnan(x), c["nulo_a_entonces"], x < np.float32(c["valor"]))
        rec(nodo["entonces"], idx[izq])
        rec(nodo["si_no"], idx[~izq])
    rec(arbol, np.arange(len(X32)))
    return out


def margen_modelo(arboles, margen, X, nombres):
    """Margen float32 como en el SQL: base y luego cada árbol sumado en real, en orden."""
    F = np.full(len(X), np.float32(margen), dtype=np.float32)
    for a in arboles:
        F = (F + recorrer(a, X, nombres)).astype(np.float32)
    return F


def entrenar_local(X, y, prueba, cortes, tasa_margen, nombres, p):
    """Mismas cuentas que `entrenar`, con numpy sobre los datos descargados y los mismos cortes y margen base."""
    X, y, prueba = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(prueba, dtype=bool)
    Bn, kv, margen = intervalos(X, cortes), [len(c) for c in cortes], tasa_margen[1]
    ent = ~prueba
    cache = {0: np.full(len(y), np.float32(margen), dtype=np.float32)}

    def evaluar_nivel(arboles, parcial, frontera):
        t = len(arboles)
        if t not in cache:
            cache[t] = (cache[t - 1] + recorrer(arboles[-1], X, nombres)).astype(np.float32)
        f = cache[t].astype(np.float64)
        pr = 1 / (1 + np.exp(-f))
        g, h = pr - y, pr * (1 - pr)
        l = np.where(f > 0, f + np.log(1 + np.exp(-f)), np.log(1 + np.exp(f))) - y * f
        nodo = np.zeros(len(y), dtype=np.int64)
        for n in sorted(parcial):  # en anchura: el padre antes que sus hijos
            x = parcial[n]
            en = nodo == n
            bj = Bn[:, x["j"]]
            va = (bj <= x["b"]) if x["izq"] else ((bj >= 0) & (bj <= x["b"]))
            nodo[en & va], nodo[en & ~va] = 2 * n + 1, 2 * n + 2
        filas = {-1: {"gt": 0.0, "ht": 0.0, "nt": int(prueba.sum()), "perdida": float(l[prueba].sum()), "r": None}} if prueba.any() else {}
        for n in frontera:
            idx = ent & (nodo == n)
            G, H, N = g[idx].sum(), h[idx].sum(), int(idx.sum())
            if N == 0:
                continue
            cands = []
            for j in range(len(cortes)):
                bj = Bn[idx, j]
                m = len(cortes[j] or [0.0]) + 2
                cnt = np.bincount(bj + 1, minlength=m)
                gs, hs = np.bincount(bj + 1, weights=g[idx], minlength=m), np.bincount(bj + 1, weights=h[idx], minlength=m)
                ga, ha, na = np.cumsum(gs[1:]), np.cumsum(hs[1:]), np.cumsum(cnt[1:])
                b = np.arange(len(ga))
                ok = (cnt[1:] > 0) & (b < kv[j])
                for izq in (True, False):
                    gl, hl, nl = ga + gs[0] * izq, ha + hs[0] * izq, na + cnt[0] * izq
                    v = ok & (hl >= p.min_hijo) & (H - hl >= p.min_hijo) & (nl >= p.min_ingresos) & (N - nl >= p.min_ingresos)
                    gan = gl ** 2 / (hl + p.lambda_) + (G - gl) ** 2 / (H - hl + p.lambda_) - G ** 2 / (H + p.lambda_)
                    for k in np.flatnonzero(v):
                        cands.append((round(float(gan[k]), DECIMALES_EMPATE), -(j + 1), -int(k), izq,
                                      (j + 1, int(k), izq, float(gl[k]), float(hl[k]), int(nl[k]), float(gan[k]))))
            r = max(cands, key=lambda c: c[:4])[4] if cands else None
            filas[n] = {"gt": float(G), "ht": float(H), "nt": N, "perdida": float(l[idx].sum()), "r": r}
        return filas

    arboles, historial = crecer(evaluar_nivel, cortes, nombres, p, aviso=lambda *_: None)
    return arboles, historial


def comparar_arboles(a, b):
    """(nodos distintos en estructura, diferencia máxima de hojas, nodos comparados) entre dos listas de árboles."""
    distintos, dmax, total = 0, 0.0, 0

    def rec(x, y):
        nonlocal distintos, dmax, total
        total += 1
        if ("hoja" in x) != ("hoja" in y):
            distintos += 1
            return
        if "hoja" in x:
            dmax = max(dmax, abs(x["hoja"] - y["hoja"]))
            return
        if (x["si"]["variable"], x["si"]["valor"], x["si"]["nulo_a_entonces"]) != (y["si"]["variable"], y["si"]["valor"],
                                                                                   y["si"]["nulo_a_entonces"]):
            distintos += 1
            return
        rec(x["entonces"], y["entonces"])
        rec(x["si_no"], y["si_no"])
    for x, y in zip(a, b):
        rec(x, y)
    return distintos + abs(len(a) - len(b)), dmax, total


# ------------------------------------------------------------------ exportación
def documento(arboles, margen, variables, contrato, nombre, p):
    """Documento if-then (formato de ir.py) con núcleo de árboles float32 como XGBoost."""
    return {"formato": ir.FORMATO,
            "modelo": {"nombre": nombre, "origen": "fnxsql entrenar", "clase": "XGBoost (hist) entrenado con consultas SQL",
                       "parametros": asdict(p)},
            "contrato": contrato.a_dict(),
            "variables": [{k: getattr(v, k) for k in ("nombre", "fuente", "codigo", "valor", "estadistico", "constante")
                           if getattr(v, k) is not None} for v in variables],
            "preproceso": [], "entrada_nucleo": [v.nombre for v in variables],
            "nucleo": {"tipo": "arboles", "precision": "float32", "combinacion": "suma_sigmoide", "base": margen,
                       "arboles": arboles}}


def a_xgboost(doc, tasa):
    """JSON de XGBoost (save_model) equivalente al documento: lo carga xgboost.Booster y lo admite `fnxsql evaluar`."""
    nombres = doc["entrada_nucleo"]
    pos = {n: i for i, n in enumerate(nombres)}
    trees = []
    for t, raiz in enumerate(doc["nucleo"]["arboles"]):
        orden, padres, izq, der = [raiz], [2147483647], [], []
        k = 0
        while k < len(orden):
            n = orden[k]
            if "si" in n:
                izq.append(len(orden))
                der.append(len(orden) + 1)
                orden += [n["entonces"], n["si_no"]]
                padres += [k, k]
            else:
                izq.append(-1)
                der.append(-1)
            k += 1
        es_hoja = ["hoja" in n for n in orden]
        trees.append({
            "base_weights": [n["hoja"] if h else n["info"]["peso"] for n, h in zip(orden, es_hoja)],
            "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
            "default_left": [0 if h else int(n["si"]["nulo_a_entonces"]) for n, h in zip(orden, es_hoja)],
            "id": t, "left_children": izq, "right_children": der, "parents": padres,
            "loss_changes": [0.0 if h else n["info"]["ganancia"] for n, h in zip(orden, es_hoja)],
            "split_conditions": [n["hoja"] if h else n["si"]["valor"] for n, h in zip(orden, es_hoja)],
            "split_indices": [0 if h else pos[n["si"]["variable"]] for n, h in zip(orden, es_hoja)],
            "split_type": [0] * len(orden),
            "sum_hessian": [n["info"]["hessiano"] for n in orden],
            "tree_param": {"num_deleted": "0", "num_feature": str(len(nombres)), "num_nodes": str(len(orden)),
                           "size_leaf_vector": "1"}})
    T = len(trees)
    return {"learner": {
        "attributes": {}, "feature_names": nombres, "feature_types": ["float"] * len(nombres),
        "gradient_booster": {"model": {"gbtree_model_param": {"num_parallel_tree": "1", "num_trees": str(T)},
                                       "iteration_indptr": list(range(T + 1)), "tree_info": [0] * T, "trees": trees},
                             "name": "gbtree"},
        "learner_model_param": {"base_score": f"[{repr(float(np.float32(tasa)))}]", "boost_from_average": "0",
                                "num_class": "0", "num_feature": str(len(nombres)), "num_target": "1"},
        "objective": {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}},
        "version": [3, 1, 0]}


def guardar(res, carpeta):
    """documento if-then, modelo XGBoost (.json), cortes, historial y consultas en la carpeta."""
    from pathlib import Path
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "if_then.json").write_text(json.dumps(res["documento"], ensure_ascii=False), encoding="utf-8")
    (carpeta / "boosting_sql.json").write_text(json.dumps(res["xgboost"]), encoding="utf-8")
    (carpeta / "entrenamiento.json").write_text(json.dumps(
        {k: res[k] for k in ("cortes", "tasa", "margen", "historial", "consultas", "segundos", "ingresos")}, indent=1,
        ensure_ascii=False, default=str), encoding="utf-8")
    return carpeta
