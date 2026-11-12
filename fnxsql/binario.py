"""Precisión de los parámetros (float64 / float32 / int8) y arrays en binario (base64) dentro de la consulta.

AWS Athena, donde corre DataNex, admite consultas de hasta 262.144 bytes. Un double escrito en texto ocupa ~20 bytes; en
binario y base64, 10,7 (float64), 5,3 (float32) o 1,3 (int8). Cada array va en UN literal base64 que la consulta
descodifica una sola vez, en un CTE de una fila, con from_base64 + from_ieee754_32 / from_ieee754_64 / from_big_endian_32
(funciones de Trino que también tiene Athena v3). Las cuentas por ingreso son en double (en real si el modelo es float32
de por sí, como XGBoost).

Dos pasos: cuantizar(doc, precision) devuelve el documento if-then con los valores que usará el SQL (así el validador
comprueba que el SQL calcula exactamente ese documento y mide aparte el error que añade la precisión frente al modelo
original), y el compilador los escribe en binario.

Precisión (config.PRECISION, opción --precision):
  float64            exacto (como el modelo); arrays de 8 bytes.
  float32 (defecto)  todo parámetro (coeficientes, pesos, sesgos, umbrales, hojas, medias, escalas, imputaciones) en
                     float32; arrays de 4 bytes. Umbrales, según cómo compara el modelo original:
                       árboles de scikit-learn (comparan float32(x) <= t; núcleo con "entrada_float32"): mayor float32
                         <= t y CAST(x AS real) <= t32: exactamente las mismas ramas que el original;
                       x <= t en double (HistGradientBoosting): menor float32 >= t (un x igual a t sigue a la izquierda;
                         sus umbrales salen de cuantiles y suelen ser valores observados);
                       x < t (XGBoost, ya float32): mayor float32 <= t, es decir, el mismo umbral.
                     En el caso double solo cambiaría de rama un valor a menos de un ulp de float32 (~6e-8 relativo) por
                     encima del umbral: imposible con datos clínicos decimales.
  int8               pesos de la red, coeficientes de la puntuación y hojas de los árboles como enteros q en [-127, 127]
                     con UNA escala float32 por neurona, por modelo (coeficientes) o por árbol: valor = q x escala
                     (escala = máx |valor| / 127). Umbrales, sesgos y preproceso como en float32: un umbral cuantizado
                     mandaría pacientes enteros a otra rama. Error medido en el experimento de mortalidad (620 ingresos):
                     de 4e-4 a 8e-2 en la probabilidad y hasta 3e-3 en el AUC; float32: <= 2e-6 y el mismo AUC.
Árboles: nodos en anchura por árbol, de modo que los dos hijos de un nodo quedan seguidos (derecho = izquierdo + 1) y el
izquierdo se guarda como distancia al nodo. Variable (posición en x), dirección del nulo y esa distancia van en un entero
de 1-4 bytes (0 = hoja). En float32 / float64, el umbral o la hoja en un float por nodo; en int8, un entero por nodo: en
los internos, el índice en la tabla de umbrales float32 del modelo; en las hojas, q + 128.
"""
import base64
import copy
import math

import numpy as np

PRECISIONES = ("float32", "float64", "int8")
TROZO = 8192  # sequence() de Trino admite como mucho 10.000 elementos: los arrays largos se descodifican a trozos
QMAX = 127


# ------------------------------------------------------------------ redondeos
def f32(x):
    """float32 más cercano (como float de Python, exacto)."""
    with np.errstate(over="ignore"):
        return float(np.float32(x))


def f32_arriba(t):
    """Menor float32 >= t."""
    if math.isinf(t):
        return float(t)
    with np.errstate(over="ignore"):
        f = np.float32(t)
    if float(f) < t:
        f = np.nextafter(f, np.float32(np.inf))
    return float(f)


def f32_abajo(t):
    """Mayor float32 <= t."""
    if math.isinf(t):
        return float(t)
    with np.errstate(over="ignore"):
        f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def umbral(t, op, entrada_f32=False):
    """Umbral en float32 tal como lo guarda el SQL (ver la cabecera del módulo)."""
    if entrada_f32 or op == "<":
        return f32_abajo(t)
    return f32_arriba(t)


def escala_int8(valores):
    """Escala float32 de un grupo: máx |valor| / 127 (1 si todos son 0)."""
    m = max((abs(float(v)) for v in valores), default=0.0)
    return f32(m / QMAX) if m > 0 else 1.0


def q_int8(v, s):
    """Entero q en [-127, 127] con q x s ~ v."""
    return int(max(-QMAX, min(QMAX, round(float(v) / s))))


# ------------------------------------------------------------------ cuantización del documento
def cuantizar(doc, precision):
    """Copia del documento if-then con los parámetros en `precision` (los valores exactos que usa el SQL). En int8 añade
    las escalas ("escalas" en el núcleo de árboles y en cada capa; "escala_coeficientes" en la puntuación) y los enteros
    ("q" en hojas y capas)."""
    if precision not in PRECISIONES:
        raise ValueError(f"precisión {precision!r} no admitida (admitidas: {PRECISIONES})")
    if precision == "float64":
        return doc
    d = copy.deepcopy(doc)
    d["precision_parametros"] = precision
    int8 = precision == "int8"
    for p in d["preproceso"]:
        for k in ("valor", "media", "escala"):
            if k in p:
                p[k] = f32(p[k])
    nuc = d["nucleo"]
    if nuc["tipo"] == "puntuacion":
        nuc["base"] = f32(nuc["base"])
        coefs = [r["entonces"]["suma_por"][1] for r in nuc["reglas"] if "suma_por" in r["entonces"]]
        s = escala_int8(coefs) if int8 else None
        if int8:
            nuc["escala_coeficientes"] = s
        for r in nuc["reglas"]:
            for c in r["si"]:
                if c["op"] in ("<=", ">") and "valor" in c:
                    c["valor"] = f32_arriba(c["valor"])  # x = t queda del mismo lado en <= y en >
            e = r["entonces"]
            if "suma" in e:
                e["suma"] = f32(e["suma"])
            else:
                e["suma_por"][1] = q_int8(e["suma_por"][1], s) * s if int8 else f32(e["suma_por"][1])
    elif nuc["tipo"] == "arboles":
        ef32, f32n = bool(nuc.get("entrada_float32")), nuc["precision"] == "float32"
        nuc["base"] = f32(nuc["base"])
        escalas = []
        for a in nuc["arboles"]:
            hojas = []

            def rec(n):
                if "hoja" in n:
                    hojas.append(n)
                    return
                n["si"]["valor"] = umbral(n["si"]["valor"], n["si"]["op"], ef32)
                rec(n["entonces"])
                rec(n["si_no"])
            rec(a)
            if int8:
                s = escala_int8([h["hoja"] for h in hojas])
                escalas.append(s)
                for h in hojas:
                    h["q"] = q_int8(h["hoja"], s)
                    h["hoja"] = f32(h["q"] * s) if f32n else h["q"] * s  # XGBoost: el producto en real
            else:
                for h in hojas:
                    h["hoja"] = f32(h["hoja"])
        if int8:
            nuc["escalas"] = escalas
    else:
        for capa in nuc["capas"]:
            capa["sesgos"] = [f32(b) for b in capa["sesgos"]]
            if int8:
                capa["escalas"] = [escala_int8(w) for w in capa["pesos"]]
                capa["q"] = [[q_int8(x, s) for x in w] for w, s in zip(capa["pesos"], capa["escalas"])]
                capa["pesos"] = [[q * s for q in qs] for qs, s in zip(capa["q"], capa["escalas"])]
            else:
                capa["pesos"] = [[f32(x) for x in w] for w in capa["pesos"]]
    return d


# ------------------------------------------------------------------ binario
def b64(datos):
    return "'" + base64.b64encode(datos).decode("ascii") + "'"


def bin_float(vals, ancho):
    return b64(np.asarray(vals, dtype=np.float64).astype(">f4" if ancho == 4 else ">f8").tobytes())


def ancho_entero(maximo):
    if maximo >= 2 ** 31:
        raise ValueError("entero demasiado grande para from_big_endian_32")
    return 1 if maximo < 2 ** 8 else 2 if maximo < 2 ** 16 else 4


def bin_uint(vals, B):
    return b64(np.asarray(vals, dtype={1: ">u1", 2: ">u2", 4: ">u4"}[B]).tobytes())


def lee(col, i, tipo):
    """Elemento i (expresión SQL, desde 0) del varbinary `col`. tipo: "f4", "f8" o "u1" / "u2" / "u4"."""
    if tipo == "f4":
        return f"from_ieee754_32(substr({col}, 4 * {i} + 1, 4))"
    if tipo == "f8":
        return f"from_ieee754_64(substr({col}, 8 * {i} + 1, 8))"
    B = int(tipo[1:])
    s = f"substr({col}, {B} * {i} + 1, {B})" if B > 1 else f"substr({col}, {i} + 1, 1)"
    return f"from_big_endian_32({s})" if B == 4 else f"from_big_endian_32(X'{'00' * (4 - B)}' || {s})"


def vector(col, n, tipo):
    """Array SQL con los n elementos del binario `col`."""
    if n <= 10_000:
        return f"transform(sequence(0, {n - 1}), i -> {lee(col, 'i', tipo)})"
    k = math.ceil(n / TROZO)
    return (f"flatten(transform(sequence(0, {k - 1}), k -> transform(sequence(k * {TROZO}, least(k * {TROZO} + {TROZO - 1}, "
            f"{n - 1})), i -> {lee(col, 'i', tipo)})))")


def matriz(col, filas, cols, tipo):
    """array(array(...)) filas x cols desde un binario guardado por filas."""
    return (f"transform(sequence(0, {filas - 1}), k -> transform(sequence(0, {cols - 1}), j -> "
            f"{lee(col, f'(k * {cols} + j)', tipo)}))")


def matriz_int8(col, escalas, filas, cols):
    """array(array(double)) filas x cols desde enteros q + 128 (1 byte) por la escala float32 de cada fila."""
    return (f"transform(sequence(0, {filas - 1}), k -> transform(sequence(0, {cols - 1}), j -> "
            f"({lee(col, f'(k * {cols} + j)', 'u1')} - 128) * CAST({lee(escalas, 'k', 'f4')} AS double)))")


# ------------------------------------------------------------------ árboles
def aplanar(arboles, pos):
    """Nodos del documento (ya cuantizado) en anchura. pos: nombre de variable -> posición en x (desde 1).
    Devuelve (filas, nodos, raices, profundidad, bits_distancia); filas[i] = None en las hojas."""
    filas, nodos, raices, prof = [], [], [], 0
    for raiz in arboles:
        orden, izq, nivel, k = [raiz], {}, [0], 0
        while k < len(orden):
            n = orden[k]
            if "si" in n:
                izq[k] = len(orden)
                orden += [n["entonces"], n["si_no"]]
                nivel += [nivel[k] + 1] * 2
            k += 1
        prof = max(prof, max(nivel))
        raices.append(len(nodos) + 1)
        for p, n in enumerate(orden):
            nodos.append(n)
            filas.append(None if "hoja" in n else
                         (pos[n["si"]["variable"]], int(bool(n["si"]["nulo_a_entonces"])), izq[p] - p))
    bd = max([x[2] for x in filas if x] or [1]).bit_length()
    return filas, nodos, raices, prof, bd


def sql_arboles(nuc, pos, precision, op):
    """(CTEs `arb_bin`, `arb_cod`, `arboles`; expresión del array de hojas de cada ingreso sobre q.x y a.*; info).
    nuc: núcleo de árboles YA cuantizado (cuantizar)."""
    filas, nodos, raices, prof, bd = aplanar(nuc["arboles"], pos)
    cod = [0 if x is None else (x[0] * 2 + x[1]) * 2 ** bd + x[2] for x in filas]
    B, Br, n, T = ancho_entero(max(cod)), ancho_entero(max(raices)), len(cod), len(raices)
    f32n = nuc["precision"] == "float32"
    comunes = (f"transform(c, z -> z / {2 ** (bd + 1)}) AS f, transform(c, z -> (z / {2 ** bd}) % 2) AS mg, "
               f"transform(c, z -> z % {2 ** bd}) AS d")
    if precision == "int8":
        tabla = sorted({n_["si"]["valor"] for n_ in nodos if "si" in n_})
        idx = {t: i for i, t in enumerate(tabla)}
        u = [n_["q"] + 128 if "hoja" in n_ else idx[n_["si"]["valor"]] for n_ in nodos]
        Bu = ancho_entero(max(u))
        cte = (f"arb_bin AS (SELECT from_base64({bin_uint(cod, B)}) AS bc, from_base64({bin_uint(u, Bu)}) AS bu, "
               f"from_base64({bin_float(tabla or [0.0], 4)}) AS bt, from_base64({bin_float(nuc['escalas'], 4)}) AS be, "
               f"from_base64({bin_uint(raices, Br)}) AS br),\n"
               f"arb_cod AS (SELECT {vector('bc', n, f'u{B}')} AS c, {vector('bu', n, f'u{Bu}')} AS u, "
               f"{vector('bt', max(len(tabla), 1), 'f4')} AS ut, {vector('be', T, 'f4')} AS esc, "
               f"{vector('br', T, f'u{Br}')} AS raices FROM arb_bin),\n"
               "arboles AS (  -- f = variable (0 en las hojas), mg = nulo a la izquierda, d = distancia al hijo izquierdo,\n"
               "              -- v = umbral (tabla float32); hojas: (u - 128) x escala del árbol\n"
               f"  SELECT {comunes}, zip_with(c, u, (ci, ui) -> IF(ci = 0, NULL, element_at(ut, ui + 1))) AS v, u, esc, raices "
               "FROM arb_cod\n),\n")
        hoja = ("CAST(element_at(a.u, n) - 128 AS real) * element_at(a.esc, t)" if f32n else
                "(element_at(a.u, n) - 128) * CAST(element_at(a.esc, t) AS double)")
    else:
        ancho = 8 if precision == "float64" else 4
        val = [n_["hoja"] if "hoja" in n_ else n_["si"]["valor"] for n_ in nodos]
        cte = (f"arb_bin AS (SELECT from_base64({bin_uint(cod, B)}) AS bc, from_base64({bin_float(val, ancho)}) AS bv, "
               f"from_base64({bin_uint(raices, Br)}) AS br),\n"
               f"arb_cod AS (SELECT {vector('bc', n, f'u{B}')} AS c, {vector('bv', n, f'f{ancho}')} AS v, "
               f"{vector('br', T, f'u{Br}')} AS raices FROM arb_bin),\n"
               "arboles AS (  -- f = variable (0 en las hojas), mg = nulo a la izquierda, d = distancia al hijo izquierdo,\n"
               "              -- v = umbral o valor de la hoja\n"
               f"  SELECT {comunes}, v, raices FROM arb_cod\n),\n")
        hoja = "element_at(a.v, n)"
    xv = "element_at(q.x, element_at(a.f, n))"
    paso = (f"(n, i) -> IF(element_at(a.f, n) = 0, n, IF(CASE WHEN {xv} IS NULL THEN element_at(a.mg, n) = 1 "
            f"ELSE {xv} {op} element_at(a.v, n) END, n + element_at(a.d, n), n + element_at(a.d, n) + 1))")
    rec = (f"transform(sequence(1, {T}), t -> reduce(sequence(1, {max(prof, 1)}), element_at(a.raices, t), {paso}, "
           f"n -> {hoja}))")
    return cte, rec, {"nodos": n, "arboles": T, "profundidad": prof, "bytes_por_codigo": B}
