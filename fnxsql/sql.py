"""Compilador: documento if-then (ir.py) -> UNA consulta SQL de Trino / Athena que puntúa los ingresos en FakeNex.

Estructura de la consulta:
  1. `vars`: variables del contrato, una fila por ingreso, todas double (contrato.sql_variables).
  2. `vec`: preproceso (imputar, indicador de nulo, estandarizar) y entrada del núcleo, como array o como columnas.
  3. núcleo, según `tipo`:
       puntuacion / terminos   logit = base + suma de CASE WHEN regla THEN valor ELSE 0 END
       arboles / arrays        nodos aplanados en binario (binario.py) y cada árbol recorrido con
                               reduce(sequence(1, profundidad), raíz, ...): sin bucles, pasos fijos
       arboles / case          un CASE anidado por árbol, cada uno en su columna y usado dos veces (suma y array) para que
                               Trino lo compile por separado
       red / arrays            matrices de pesos en binario; capa = zip_with(pesos, sesgos,
                               (w, b) -> greatest(0, reduce(zip_with(w, x, ...)) + b))
       red / terminos          cada neurona escrita término a término
  4. probabilidad: sigmoide del logit (o media de las hojas en un random forest).

Precisión (binario.py): float32 por defecto, float64 (exacto) o int8. El documento se cuantiza primero
(binario.cuantizar) y el SQL escribe exactamente esos valores; las cuentas por ingreso van en double. Un modelo float32
de por sí (XGBoost) compara float32(x) < umbral y suma las hojas en real con cualquier precisión.

Salvaguardas (config): tamaño máximo del texto (el de AWS Athena, 262.144 bytes; con --forzar, el del proxy de FakeNex)
y, para "case" / "terminos", número máximo de términos: la red grande escrita término a término (~20.000 operaciones)
tumbó el coordinador de Trino; con arrays planifica en ~1 s.
"""
import numpy as np

from . import binario as B
from . import config as C
from .contrato import interpretar, lit, sql_variables, Contrato

TIPOS = {"puntuacion": ["terminos"], "arboles": ["arrays", "case"], "red": ["arrays", "terminos"]}
PRECISIONES = B.PRECISIONES


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


def compilar(doc, pids, tipo="auto", forzar=False, precision=None):
    """(sql, info) para los pacientes `pids`. tipo: "auto" (el primero admitido) o uno de TIPOS[núcleo].
    precision: "float32" (por defecto, config.PRECISION), "float64" o "int8"."""
    nuc = doc["nucleo"]
    tipo = TIPOS[nuc["tipo"]][0] if tipo in (None, "auto") else tipo
    if tipo not in TIPOS[nuc["tipo"]]:
        raise ValueError(f"tipo {tipo!r} no admitido para un núcleo {nuc['tipo']!r} (admitidos: {TIPOS[nuc['tipo']]})")
    precision = precision or C.PRECISION
    dq = B.cuantizar(doc, precision)  # los valores exactos que escribe el SQL (literales double o binario)
    contrato = _contrato(dq)
    variables = [interpretar(v["nombre"], contrato) for v in dq["variables"]]
    ex = _expresiones(dq)
    cab = (f"-- {dq['modelo']['nombre']} ({dq['modelo'].get('clase', '')}), núcleo {nuc['tipo']} / {tipo}, {precision}; "
           "generado por fnxsql\n-- desde el documento if-then: una fila por ingreso con la probabilidad del modelo.\n")
    cuerpo, info = {"puntuacion": _puntuacion, "arboles": _arboles, "red": _red}[nuc["tipo"]](dq, ex, tipo, precision)
    sql = cab + sql_variables(variables, contrato, pids) + ",\n" + cuerpo
    info.update({"tipo_sql": tipo, "precision": precision, "bytes": len(sql.encode("utf-8"))})
    if info["bytes"] > C.MAX_SQL_BYTES_FAKENEX or (info["bytes"] > C.MAX_SQL_BYTES and not forzar):
        raise SQLDemasiadoGrande(
            f"SQL de {info['bytes']:,} bytes: no cabe en AWS Athena ({C.MAX_SQL_BYTES:,} bytes por consulta)"
            + ("" if info["bytes"] > C.MAX_SQL_BYTES_FAKENEX else
               "; usa --sql arrays, --precision int8, menos pacientes por lote, o --forzar para probarla en FakeNex")
            + (f" ni en el proxy de FakeNex ({C.MAX_SQL_BYTES_FAKENEX:,} bytes)" if info["bytes"] > C.MAX_SQL_BYTES_FAKENEX else ""))
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


def _puntuacion(doc, ex, tipo, precision):
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
def _real(x):
    """Literal REAL con el float32 de x (modelos float32 de por sí, como XGBoost)."""
    return "REAL '" + repr(np.float32(x)).replace("np.float32(", "").rstrip(")") + "'"


def _ops(arboles):
    ops = set()

    def rec(n):
        if "si" in n:
            ops.add(n["si"]["op"])
            rec(n["entonces"])
            rec(n["si_no"])
    for a in arboles:
        rec(a)
    if len(ops) > 1:
        raise ValueError(f"comparaciones mezcladas en los árboles: {sorted(ops)}")
    return ops.pop() if ops else "<="


def _arboles(doc, ex, tipo, precision):
    nuc = doc["nucleo"]
    f32n = nuc["precision"] == "float32"  # el modelo compara y suma en float32 (XGBoost)
    ef32 = bool(nuc.get("entrada_float32")) and not f32n  # árboles de scikit-learn: comparan float32(x)
    tnum = "real" if (f32n or ef32) else "double"
    entrada = doc["entrada_nucleo"]
    comb = nuc["combinacion"]
    op = _ops(nuc["arboles"])
    if tipo == "case":
        cols, nodos = [], [0]
        col = {n: f"n{i:03d}" for i, n in enumerate(entrada)}

        def rec(n):
            nodos[0] += 1
            if "hoja" in n:
                return _real(n["hoja"]) if f32n else lit(n["hoja"])
            c = n["si"]
            x = f"CAST({col[c['variable']]} AS real)" if (f32n or ef32) else col[c["variable"]]
            cmp = f"{x} {c['op']} {_real(c['valor']) if f32n else lit(c['valor'])}"
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
        if comb == "media":
            prob = f"({suma}) / {lit(len(names))}"
        elif f32n:
            prob = f"1E0 / (1E0 + exp(-CAST({_real(nuc['base'])} + {suma} AS double)))"
        else:
            prob = f"1E0 / (1E0 + exp(-({lit(nuc['base'])} + {suma})))"
        cuerpo = (vec + arb + f"nucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval, {prob} AS probabilidad, {cont}\n"
                  "  FROM arboles\n)\n" + FINAL)
        return cuerpo, {"terminos": nodos[0], "arboles": len(cols)}
    pos = {n: i + 1 for i, n in enumerate(entrada)}
    # un modelo float32 de por sí guarda sus valores en 4 bytes también con float64 (son float32 exactos)
    arb, rec, info = B.sql_arboles(nuc, pos, "float32" if (f32n and precision == "float64") else precision, op)
    x = ", ".join(f"CAST({ex[n]} AS {tnum})" for n in entrada)
    vec = f"vec AS (SELECT patient_ref, episode_ref, t0, t_eval, ARRAY[{x}] AS x FROM vars),\n"
    hojas = (f"hojas AS (\n  SELECT q.patient_ref, q.episode_ref, q.t0, q.t_eval,\n         {rec} AS c\n"
             "  FROM vec q CROSS JOIN arboles a\n),\n")
    if comb == "media":
        prob = f"reduce(c, 0E0, (s, h) -> s + h, s -> s) / {lit(len(nuc['arboles']))}"
    elif f32n:
        prob = f"1E0 / (1E0 + exp(-CAST(reduce(c, {_real(nuc['base'])}, (s, h) -> s + h, s -> s) AS double)))"
    else:
        prob = f"1E0 / (1E0 + exp(-reduce(c, {lit(nuc['base'])}, (s, h) -> s + h, s -> s)))"
    cuerpo = (arb + vec + hojas + f"nucleo AS (\n  SELECT patient_ref, episode_ref, t0, t_eval, {prob} AS probabilidad FROM hojas\n)\n"
              + FINAL)
    return cuerpo, info


# ------------------------------------------------------------------ red
def _red(doc, ex, tipo, precision):
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
    ancho = 8 if precision == "float64" else 4
    tf = f"f{ancho}"
    blobs, decs = [], []
    for li, capa in enumerate(capas, start=1):
        filas, cols = len(capa["pesos"]), len(capa["pesos"][0])
        if precision == "int8":
            blobs += [f"from_base64({B.bin_uint([q + 128 for fila in capa['q'] for q in fila], 1)}) AS bw{li}",
                      f"from_base64({B.bin_float(capa['escalas'], 4)}) AS bs{li}"]
            decs.append(f"{B.matriz_int8(f'bw{li}', f'bs{li}', filas, cols)} AS w{li}")
        else:
            blobs.append(f"from_base64({B.bin_float([w for fila in capa['pesos'] for w in fila], ancho)}) AS bw{li}")
            decs.append(f"{B.matriz(f'bw{li}', filas, cols, tf)} AS w{li}")
        blobs.append(f"from_base64({B.bin_float(capa['sesgos'], ancho)}) AS bb{li}")
        decs.append(f"{B.vector(f'bb{li}', filas, tf)} AS b{li}")
    ctes = [f"pesos AS (  -- pesos (una fila por neurona) y sesgos de cada capa, en binario {precision}\n  SELECT "
            + ", ".join(decs) + "\n  FROM (SELECT " + ", ".join(blobs) + ")\n)",
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
