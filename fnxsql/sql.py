"""Compilador: documento if-then (ir.py) -> UNA consulta SQL de Trino / Athena que puntúa los ingresos en FakeNex.

Estructura de la consulta:
  1. `vars`: variables del contrato, una fila por ingreso (contrato.sql_variables).
  2. `vec`: preproceso (imputar, indicador de nulo, estandarizar) y entrada del núcleo, como array o como columnas.
  3. núcleo, según `tipo`:
       puntuacion / terminos   logit = base + suma de CASE WHEN regla THEN valor ELSE 0 END
       arboles / arrays        nodos aplanados en literales JSON (variable, umbral, hijos, nulo, hoja) y cada árbol
                               recorrido con reduce(sequence(1, profundidad), raíz, ...): sin bucles, pasos fijos
       arboles / case          un CASE anidado por árbol, cada uno en su columna y usado dos veces (suma y array) para que
                               Trino lo compile por separado
       red / arrays            matrices de pesos como literales JSON; capa = zip_with(pesos, sesgos,
                               (w, b) -> greatest(0, reduce(zip_with(w, x, ...)) + b))
       red / terminos          cada neurona escrita término a término
  4. probabilidad: sigmoide del logit (o media de las hojas en un random forest).
Precisión: float32 (XGBoost) -> arrays `real` y comparación float32(x) < umbral, suma en real; float64 -> double y x <= umbral.

Salvaguardas (config): tamaño máximo del texto y, para "case" / "terminos", número máximo de términos: la red grande
escrita término a término (~20.000 operaciones) tumbó el coordinador de Trino; con arrays planifica en ~1 s.
"""
import json

from . import config as C
from .contrato import interpretar, lit, sql_variables, Contrato

TIPOS = {"puntuacion": ["terminos"], "arboles": ["arrays", "case"], "red": ["arrays", "terminos"]}


class SQLDemasiadoGrande(ValueError):
    pass


def tipos_admitidos(doc):
    return TIPOS[doc["nucleo"]["tipo"]]


def _contrato(doc):
    d = doc["contrato"]
    return Contrato(momento_horas=d["momento_horas"], ventana_horas=d["ventana_horas"], si_falta=d["si_falta"],
                    constantes=d.get("constantes", {}))


def _expresiones(doc):
    """Nombre -> expresión SQL (double) sobre las columnas de `vars`, para variables y salidas del preproceso."""
    ex = {v["nombre"]: f"v{i:03d}" for i, v in enumerate(doc["variables"])}
    for p in doc["preproceso"]:
        e = ex[p["variable"]]
        if p["tipo"] == "imputar":
            ex[p["salida"]] = f"coalesce({e}, {lit(p['valor'])})"
        elif p["tipo"] == "indicador_nulo":
            ex[p["salida"]] = f"(CASE WHEN {e} IS NULL THEN 1E0 ELSE 0E0 END)"
        else:
            ex[p["salida"]] = f"(({e} - {lit(p['media'])}) / {lit(p['escala'])})"
    return ex


def compilar(doc, pids, tipo="auto", forzar=False):
    """(sql, info) para los pacientes `pids`. tipo: "auto" (el primero admitido) o uno de TIPOS[núcleo]."""
    nuc = doc["nucleo"]
    tipo = TIPOS[nuc["tipo"]][0] if tipo in (None, "auto") else tipo
    if tipo not in TIPOS[nuc["tipo"]]:
        raise ValueError(f"tipo {tipo!r} no admitido para un núcleo {nuc['tipo']!r} (admitidos: {TIPOS[nuc['tipo']]})")
    contrato = _contrato(doc)
    variables = [interpretar(v["nombre"], contrato) for v in doc["variables"]]
    ex = _expresiones(doc)
    cab = (f"-- {doc['modelo']['nombre']} ({doc['modelo'].get('clase', '')}), núcleo {nuc['tipo']} / {tipo}; generado por fnxsql\n"
           f"-- desde el documento if-then: una fila por ingreso con la probabilidad del modelo.\n")
    cuerpo, info = {"puntuacion": _puntuacion, "arboles": _arboles, "red": _red}[nuc["tipo"]](doc, ex, tipo)
    sql = cab + sql_variables(variables, contrato, pids) + ",\n" + cuerpo
    info.update({"tipo_sql": tipo, "bytes": len(sql.encode("utf-8"))})
    if info["bytes"] > C.MAX_SQL_BYTES:
        raise SQLDemasiadoGrande(f"SQL de {info['bytes'] / 1024:.0f} KB (máximo {C.MAX_SQL_BYTES / 1024:.0f} KB)")
    if info.get("terminos", 0) > C.MAX_TERMINOS_SQL and not forzar:
        raise SQLDemasiadoGrande(f"{info['terminos']} términos en la representación {tipo!r} (máximo {C.MAX_TERMINOS_SQL}): "
                                 "riesgo de agotar la memoria del coordinador de Trino al planificar; usa --sql arrays")
    return sql, info


FINAL = ("SELECT patient_ref, episode_ref, t0 AS ingreso, t_eval AS momento, probabilidad\n"
         "FROM nucleo\nORDER BY patient_ref, ingreso")


def _cond(c, ex):
    v = ex[c["variable"]]
    op = c["op"]
    if op == "es_nulo":
        return f"{v} IS NULL"
    if op == "no_es_nulo":
        return f"{v} IS NOT NULL"
    base = {"<=": f"{v} <= {lit(c.get('valor', 0))}", ">": f"{v} > {lit(c.get('valor', 0))}",
            "==": f"{v} = {lit(c.get('valor', 0))}",
            "en": f"{v} IN ({', '.join(lit(x) for x in c.get('valores', []))})"}[op]
    return f"({base} OR {v} IS NULL)" if c.get("o_nulo") else base


def _puntuacion(doc, ex, tipo):
    nuc = doc["nucleo"]
    terms = []
    for r in nuc["reglas"]:
        cond = " AND ".join(_cond(c, ex) for c in r["si"]) or "TRUE"
        e = r["entonces"]
        val = lit(e["suma"]) if "suma" in e else f"{lit(e['suma_por'][1])} * {ex[e['suma_por'][0]]}"
        terms.append(f"CASE WHEN {cond} THEN {val} ELSE 0E0 END")
    cuerpo = ("nucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval,\n         1E0 / (1E0 + exp(-(" + lit(nuc["base"])
              + "\n           + " + "\n           + ".join(terms) + "))) AS probabilidad\n  FROM vars\n)\n" + FINAL)
    return cuerpo, {"terminos": len(terms)}


# ------------------------------------------------------------------ árboles
def _aplanar(arboles, entrada):
    """Nodos de todos los árboles en arrays (índices desde 1; l = 0 en las hojas)."""
    pos = {n: i + 1 for i, n in enumerate(entrada)}
    A = {k: [] for k in ("f", "t", "l", "r", "mg", "v")}
    raices, prof = [], 0

    def nuevo():
        for k in A:
            A[k].append(0)
        return len(A["f"])

    def rec(n, d):
        nonlocal prof
        i = nuevo()
        prof = max(prof, d)
        if "hoja" in n:
            A["v"][i - 1] = n["hoja"]
            return i
        c = n["si"]
        A["f"][i - 1], A["t"][i - 1], A["mg"][i - 1] = pos[c["variable"]], c["valor"], int(c["nulo_a_entonces"])
        A["l"][i - 1] = rec(n["entonces"], d + 1)
        A["r"][i - 1] = rec(n["si_no"], d + 1)
        return i

    for a in arboles:
        raices.append(rec(a, 0))
    return A, raices, prof


def _num(x, f32):
    if f32:
        import numpy as np
        return repr(np.float32(x)).replace("np.float32(", "").rstrip(")")
    return repr(float(x))


def _arboles(doc, ex, tipo):
    nuc = doc["nucleo"]
    f32 = nuc["precision"] == "float32"
    tnum = "real" if f32 else "double"
    entrada = doc["entrada_nucleo"]
    comb = nuc["combinacion"]
    if tipo == "case":
        cols, nodos = [], [0]
        col = {n: f"n{i:03d}" for i, n in enumerate(entrada)}

        def rec(n):
            nodos[0] += 1
            if "hoja" in n:
                return (f"REAL '{_num(n['hoja'], True)}'" if f32 else lit(n["hoja"]))
            c = n["si"]
            x = f"CAST({col[c['variable']]} AS real)" if f32 else col[c["variable"]]
            t = f"REAL '{_num(c['valor'], True)}'" if f32 else lit(c["valor"])
            cmp = f"{x} {'<' if c['op'] == '<' else '<='} {t}"
            cond = f"({cmp} OR {col[c['variable']]} IS NULL)" if c["nulo_a_entonces"] else cmp
            return f"CASE WHEN {cond} THEN {rec(n['entonces'])} ELSE {rec(n['si_no'])} END"
        for a in nuc["arboles"]:
            cols.append(rec(a))
        names = [f"a{i:04d}" for i in range(len(cols))]
        vec = ("vec AS (\n  SELECT patient_ref, episode_ref, t0, t_eval,\n         "
               + ",\n         ".join(f"{ex[n]} AS {col[n]}" for n in entrada) + "\n  FROM vars\n),\n")
        arb = ("arboles AS (\n  SELECT patient_ref, episode_ref, t0, t_eval,\n" + ",\n".join(f"    {c} AS {n}" for c, n in zip(cols, names))
               + "\n  FROM vec\n),\n")
        suma = " + ".join(names)
        cont = f"ARRAY[{', '.join(names)}] AS contribuciones"  # segunda referencia: evita que Trino funda los árboles
        prob = (f"({suma}) / {lit(len(names))}" if comb == "media" else f"1E0 / (1E0 + exp(-CAST({_base(nuc, f32)} + {suma} AS double)))")
        cuerpo = (vec + arb + f"nucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval, {prob} AS probabilidad, {cont}\n"
                  "  FROM arboles\n)\n" + FINAL)
        return cuerpo, {"terminos": nodos[0], "arboles": len(cols)}
    A, raices, prof = _aplanar(nuc["arboles"], entrada)
    lits = {k: json.dumps(A[k], separators=(",", ":")) for k in ("f", "l", "r", "mg")}
    lits["t"] = "[" + ",".join(_num(x, f32) for x in A["t"]) + "]"
    lits["v"] = "[" + ",".join(_num(x, f32) for x in A["v"]) + "]"
    tipos = {"f": "array(integer)", "l": "array(integer)", "r": "array(integer)", "mg": "array(integer)",
             "t": f"array({tnum})", "v": f"array({tnum})"}
    arb = ("arboles AS (  -- todos los nodos aplanados como literales JSON (índices desde 1; l = 0 en las hojas)\n  SELECT "
           + ",\n         ".join(f"CAST(json_parse('{lits[k]}') AS {tipos[k]}) AS {k}" for k in tipos)
           + f",\n         CAST(json_parse('{json.dumps(raices, separators=(',', ':'))}') AS array(integer)) AS raices\n),\n")
    x = ", ".join(f"CAST({ex[n]} AS {tnum})" for n in entrada)
    vec = f"vec AS (SELECT patient_ref, episode_ref, t0, t_eval, ARRAY[{x}] AS x FROM vars),\n"
    op = "<" if f32 else "<="
    xv = "element_at(q.x, element_at(a.f, n))"
    paso = (f"(n, i) -> IF(element_at(a.l, n) = 0, n, IF(CASE WHEN {xv} IS NULL THEN element_at(a.mg, n) = 1 "
            f"ELSE {xv} {op} element_at(a.t, n) END, element_at(a.l, n), element_at(a.r, n)))")
    hojas = (f"hojas AS (\n  SELECT q.patient_ref, q.episode_ref, q.t0, q.t_eval,\n         transform(a.raices, rz -> reduce("
             f"sequence(1, {max(prof, 1)}), rz,\n           {paso},\n           n -> element_at(a.v, n))) AS c\n"
             "  FROM vec q CROSS JOIN arboles a\n),\n")
    if comb == "media":
        prob = f"reduce(c, 0E0, (s, h) -> s + h, s -> s) / {lit(len(raices))}"
    else:
        cero = f"CAST({_num(nuc['base'], f32)} AS {tnum})" if f32 else lit(nuc["base"])
        prob = f"1E0 / (1E0 + exp(-CAST(reduce(c, {cero}, (s, h) -> s + h, s -> s) AS double)))"
    cuerpo = (arb + vec + hojas + f"nucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval, {prob} AS probabilidad FROM hojas\n)\n"
              + FINAL)
    return cuerpo, {"nodos": len(A["f"]), "arboles": len(raices), "profundidad": prof}


def _base(nuc, f32):
    return f"REAL '{_num(nuc['base'], True)}'" if f32 else lit(nuc["base"])


# ------------------------------------------------------------------ red
def _red(doc, ex, tipo):
    capas = doc["nucleo"]["capas"]
    entrada = doc["entrada_nucleo"]
    if tipo == "terminos":
        prev = [ex[n] for n in entrada]
        ctes, fuente, terms = [], "vars", 0
        for li, capa in enumerate(capas, start=1):
            exprs = []
            for w, b in zip(capa["pesos"], capa["sesgos"]):
                s = lit(b) + "".join(f" + {lit(wi)} * {xi}" for wi, xi in zip(w, prev))
                terms += len(w)
                exprs.append(s if capa["activacion"] == "identidad" else f"greatest(0E0, {s})")
            nombres = [f"h{li}_{j:03d}" for j in range(len(exprs))]
            extra = f", ARRAY[{', '.join(nombres)}] AS act" if li == len(capas) - 1 else ""
            ctes.append(f"capa{li} AS (\n  SELECT patient_ref, episode_ref, t0, t_eval,\n         "
                        + ",\n         ".join(f"{e} AS {n}" for e, n in zip(exprs, nombres)) + extra + f"\n  FROM {fuente}\n)")
            prev, fuente = nombres, f"capa{li}"
        cuerpo = (",\n".join(ctes) + ",\nnucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval, "
                  f"1E0 / (1E0 + exp(-{prev[0]})) AS probabilidad FROM {fuente}\n)\n" + FINAL)
        return cuerpo, {"terminos": terms, "capas": [len(c["sesgos"]) for c in capas]}
    lits = []
    for li, capa in enumerate(capas, start=1):
        lits.append(f"CAST(json_parse('{json.dumps(capa['pesos'], separators=(',', ':'))}') AS array(array(double))) AS w{li}")
        lits.append(f"CAST(json_parse('{json.dumps(capa['sesgos'], separators=(',', ':'))}') AS array(double)) AS b{li}")
    ctes = ["pesos AS (  -- pesos y sesgos como literales JSON (un token por matriz)\n  SELECT " + ",\n         ".join(lits) + "\n)",
            f"x0 AS (SELECT patient_ref, episode_ref, t0, t_eval, ARRAY[{', '.join(ex[n] for n in entrada)}] AS x FROM vars)"]
    dot = "reduce(zip_with(w, v.x, (p, e) -> p * e), 0E0, (s, t) -> s + t, s -> s)"
    for li, capa in enumerate(capas, start=1):
        act = "greatest(0E0, {d} + b)" if capa["activacion"] == "relu" else "{d} + b"
        ctes.append(f"x{li} AS (\n  SELECT v.patient_ref, v.episode_ref, v.t0, v.t_eval,\n         zip_with(p.w{li}, p.b{li}, (w, b) -> "
                    f"{act.format(d=dot)}) AS x\n  FROM x{li - 1} v CROSS JOIN pesos p\n)")
    n = len(capas)
    cuerpo = (",\n".join(ctes) + ",\nnucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval, "
              f"1E0 / (1E0 + exp(-element_at(x, 1))) AS probabilidad FROM x{n}\n)\n" + FINAL)
    return cuerpo, {"pesos": sum(len(w) for c in capas for w in c["pesos"]), "capas": [len(c["sesgos"]) for c in capas]}
