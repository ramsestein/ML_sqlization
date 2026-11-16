"""Entrenar el gradient boosting en UNA consulta SQL: el bucle de boosting corre dentro de Trino con funciones de arrays.

entrenar.py hace una consulta por árbol y nivel (300 para 100 árboles de profundidad 3): demasiadas para que una persona
las lance. Aquí todo va en una sola consulta, sin permiso de escritura y sin que salga ninguna fila de pacientes:

  agg   una fila con los ingresos de entrenamiento y de prueba como arrays (variables en float32, desenlace), ordenados
        por episode_ref, y los cortes de cada variable (approx_percentile en float32, como max_bin de XGBoost)
  dat   intervalo de cada valor (width_bucket; -1 = ausente), por variable
  bucle reduce(sequence(1, árboles), estado, (estado, t) -> árbol t) con estado = (margen de cada ingreso en real,
        árboles hechos, pérdidas). Cada árbol, por niveles con otro reduce:
          g = p - y, h = p (1 - p)
          histograma de cada variable: multimap_from_entries(nodo * KM + intervalo -> (g, h)) y suma de cada clave
          sumas acumuladas por intervalo (suma por duplicación: log2(bins) pasos de zip_with)
          mejor corte de cada nodo: máximo de (ganancia redondeada, -variable, -intervalo, ausentes a la izquierda)
          nuevo nodo de cada ingreso; hojas = float32(-eta G / (H + lambda)); margen += hoja (en real)
  salida una fila: los árboles (por nodos en anchura), las pérdidas por número de árboles y los cortes.

Las sumas se hacen en el orden de los ingresos (reduce secuencial) y las acumuladas por duplicación: `local` repite
exactamente las mismas operaciones en numpy, así que el resultado debe coincidir bit a bit.
Si el bucle no cabe en el tiempo de una consulta, `arboles_por_consulta` lo parte en unas pocas: cada una recibe los
cortes y el modelo hecho como arrays binarios (binario.py) y sigue desde ahí.
"""
import math
import time

import numpy as np

from . import binario as B, cliente, config as C, contrato as K, entrenar as E
from .binario import f32
from .contrato import lit

CAND = "ROW(r double, nj bigint, nb bigint, izq boolean, j bigint, b bigint, gl double, hl double, nl bigint, gan double)"
PQ = "ROW(gl array(double), hl array(double), nl array(bigint), cb array(bigint), gm double, hm double, nm bigint)"
ARB = ("ROW(feat array(bigint), bin array(bigint), izq array(boolean), val array(real), g array(double), h array(double), "
       "gan array(double), n array(bigint))")
# Estado de la máquina: un solo reduce cuyo paso elige con CASE qué hacer (ver sql_bucle)
CAMPOS = [("f", "array(real)"), ("ft", "array(real)"), ("g", "array(double)"), ("h", "array(double)"), ("nd", "array(bigint)"),
          ("feat", "array(bigint)"), ("bin", "array(bigint)"), ("izq", "array(boolean)"), ("tg", "array(double)"),
          ("th", "array(double)"), ("tn", "array(bigint)"), ("ab", "array(boolean)"), ("gan", "array(double)"),
          ("cl", "array(bigint)"), ("hm", "map(bigint, row(double, double, bigint))"), ("pq", f"array({PQ})"),
          ("gt", "array(array(double))"), ("gf", "array(array(double))"), ("bs", f"array({CAND})"),
          ("va", "array(real)"), ("arb", f"array({ARB})"), ("perd", "array(ROW(e double, p double))")]
ST = "ROW(" + ", ".join(f"{n} {t}" for n, t in CAMPOS) + ")"
VACIOS = {"hm": "CAST(MAP() AS map(bigint, row(double, double, bigint)))", "pq": f"CAST(ARRAY[] AS array({PQ}))",
          "bs": f"CAST(ARRAY[] AS array({CAND}))", "cl": "CAST(ARRAY[] AS array(bigint))",
          "gt": "CAST(ARRAY[] AS array(array(double)))", "gf": "CAST(ARRAY[] AS array(array(double)))"}
MAX_SEGUNDOS = 540  # FakeNex corta a los 10 min; una consulta larga es mejor para el servidor que cientos cortas
# El proxy de FakeNex tarda exponencialmente en aceptar lambdas anidadas (medido: 8 niveles +3 s, 9 +10 s, 10 +30 s);
# la máquina de estados no pasa de 6 niveles.


def desplazamientos(bins):
    return [2 ** k for k in range(max(1, math.ceil(math.log2(bins + 1))))]


def _scan(arr, cero, bins):
    """Sumas acumuladas por duplicación (Hillis-Steele): log2(bins) pasos de zip_with."""
    return (f"reduce(ARRAY[{', '.join(map(str, desplazamientos(bins)))}], {arr}, (sa, sk) -> zip_with(sa, "
            f"slice(concat(repeat({cero}, CAST(sk AS integer)), sa), 1, cardinality(sa)), (su, sv) -> su + sv), sa -> sa)")


def _perdida(f, y):
    fd = "CAST(pf AS double)"
    return (f"(reduce(zip_with({f}, {y}, (pf, py) -> IF({fd} > 0, {fd} + ln(1E0 + exp(-{fd})), ln(1E0 + exp({fd}))) - py * {fd}), "
            f"0E0, (s, x) -> s + x, s -> s) / cardinality({y}))")


def _estado(**cambios):
    """Nuevo estado: los campos cambiados y el resto copiado de st."""
    return f"CAST(ROW({', '.join(cambios.get(n, f'st.{n}') for n, _ in CAMPOS)}) AS {ST})"


def pasos_por_arbol(D):
    return 6 * D + 3


def sql_bucle(J, p, arboles):
    """Expresión que entrena `arboles` árboles sobre las columnas de `dat` (ye, yp, be, bp, c, kv, f0e, f0p).

    Máquina de estados: reduce(sequence(1, árboles x P), estado, (st, it) -> CASE paso ...), con P = 4 x profundidad + 3:
      0          inicio del árbol: g, h, pérdida del modelo hasta ahora, árbol vacío
      por nivel  histograma (mapa clave -> sumas), sumas acumuladas por (nodo, variable), mejor corte por nodo, actualizar
      P - 2      hojas: float32(-eta G / (H + lambda))
      P - 1      cierre: margen += hoja (entrenamiento por su nodo; prueba recorriendo el árbol) y guardar el árbol
    Ninguna expresión anida más de 4 lambdas."""
    D, S, KM, NL = p.profundidad, 2 ** (p.profundidad + 1) - 1, p.bins + 1, 2 ** (p.profundidad - 1)
    lam, eta, mh, mn, eps = lit(p.lambda_), lit(p.eta), lit(p.min_hijo), int(p.min_ingresos), lit(E.EPS_GANANCIA)
    P = pasos_por_arbol(D)
    Z = "BIGINT '0'"
    N, M = "CAST(cardinality(ye) AS bigint)", "cardinality(yp)"
    pr = "(1E0 / (1E0 + exp(-CAST(fi AS double))))"
    suma = lambda a: f"reduce({a}, 0E0, (s, x) -> s + x, s -> s)"
    g_expr = f"zip_with(st.f, ye, (fi, yi) -> {pr} - yi)"
    ramas = {}
    # ---- 0: inicio del árbol
    ramas[0] = _estado(
        g=g_expr, h=f"transform(st.f, fi -> {pr} * (1E0 - {pr}))", nd=f"repeat({Z}, {N})",
        feat=f"repeat({Z}, {S})", bin=f"repeat({Z}, {S})", izq=f"repeat(false, {S})",
        tg=f"concat(ARRAY[{suma(g_expr)}], repeat(0E0, {S - 1}))",
        th=f"concat(ARRAY[{suma(f'transform(st.f, fi -> {pr} * (1E0 - {pr}))')}], repeat(0E0, {S - 1}))",
        tn=f"concat(ARRAY[{N}], repeat({Z}, {S - 1}))", ab=f"concat(ARRAY[true], repeat(false, {S - 1}))",
        gan=f"repeat(0E0, {S})",
        perd=f"st.perd || ARRAY[CAST(ROW({_perdida('st.f', 'ye')}, {_perdida('st.ft', 'yp')}) AS ROW(e double, p double))]")
    centinela = f"CAST(ROW(-infinity(), 0, 0, false, 0, 0, 0E0, 0E0, 0, -infinity()) AS {CAND})"
    for lv in range(D):
        lo, hi, n_lv = 2 ** lv - 1, 2 ** (lv + 1) - 2, 2 ** lv
        r0 = 1 + 6 * lv
        # ---- claves: ((j - 1) * NL + nodo - lo) * KM + intervalo + 1 por (variable, ingreso); -1 si el ingreso no está en el nivel
        ramas[r0] = _estado(cl=(f"flatten(transform(sequence(1, {J}), j -> zip_with(st.nd, element_at(be, j), (zn, zb) -> "
                                f"IF(zn >= {lo}, ((j - 1) * {NL} + zn - {lo}) * {KM} + zb + 1, BIGINT '-1'))))"))
        # ---- histograma: clave -> (suma g, suma h, ingresos), sumas en el orden de las filas
        ramas[r0 + 1] = _estado(cl=VACIOS["cl"], hm=(
            f"transform_values(multimap_from_entries(zip(st.cl, zip(flatten(repeat(st.g, {J})), flatten(repeat(st.h, {J}))))), "
            f"(k, v) -> ROW(reduce(v, 0E0, (s, x) -> s + x[1], s -> s), reduce(v, 0E0, (s, x) -> s + x[2], s -> s), "
            "CAST(cardinality(v) AS bigint)))"))
        # ---- sumas acumuladas por (nodo, variable): q = (nodo - lo) * J + (j - 1); una búsqueda en el mapa por intervalo
        jq, nq = f"(q % {J} + 1)", f"({lo} + q / {J})"
        base = f"(((q % {J}) * {NL} + q / {J}) * {KM})"
        cero = f"CAST(ROW(0E0, 0E0, {Z}) AS row(double, double, bigint))"
        celdas = (f"transform(sequence(1, cardinality(element_at(c, {jq})) + 1), b -> "
                  f"coalesce(element_at(st.hm, {base} + b), {cero}))")
        falta = f"coalesce(element_at(st.hm, {base}), {cero})"
        cuerpo = (f"CAST(ROW({_scan('transform(rr, x -> x[1])', '0E0', p.bins)}, {_scan('transform(rr, x -> x[2])', '0E0', p.bins)}, "
                  f"{_scan('transform(rr, x -> x[3])', Z, p.bins)}, transform(rr, x -> x[3]), {falta}[1], {falta}[2], {falta}[3]) AS {PQ})")
        pq = (f"IF(element_at(st.ab, {nq} + 1), element_at(transform(ARRAY[{celdas}], rr -> {cuerpo}), 1), "
              f"CAST(ROW(ARRAY[], ARRAY[], ARRAY[], ARRAY[], 0E0, 0E0, 0) AS {PQ}))")
        ramas[r0 + 2] = _estado(pq=f"transform(sequence(0, {n_lv * J - 1}), q -> {pq})", hm=VACIOS["hm"])
        # ---- ganancias redondeadas por intervalo, con los ausentes a la izquierda (gt) o a la derecha (gf); -inf si no vale
        G, H, C = "L[2]", "L[3]", "L[4]"
        def gan_arr(izq):
            gl = f"(e[1] + L[1].gm)" if izq else "e[1]"
            hl = f"(e[2] + L[1].hm)" if izq else "e[2]"
            nl = f"(e[3] + L[1].nm)" if izq else "e[3]"
            gan = (f"({gl} * {gl} / ({hl} + {lam}) + ({G} - {gl}) * ({G} - {gl}) / ({H} - {hl} + {lam}) - {G} * {G} / ({H} + {lam}))")
            valido = (f"e[5] <= L[5] AND e[4] > 0 AND {hl} >= {mh} AND {H} - {hl} >= {mh} AND {nl} >= {mn} AND {C} - {nl} >= {mn}")
            return (f"transform(zip(L[1].gl, L[1].hl, L[1].nl, L[1].cb, sequence(1, cardinality(L[1].gl))), "
                    f"e -> IF({valido}, round({gan}, {E.DECIMALES_EMPATE}), -infinity()))")
        datos_q = (f"ROW(element_at(st.pq, q + 1), element_at(st.tg, {nq} + 1), element_at(st.th, {nq} + 1), "
                   f"element_at(st.tn, {nq} + 1), CAST(element_at(kv, {jq}) AS bigint))")
        vacio = "ARRAY[-infinity()]"
        gq = lambda izq: (f"transform(sequence(0, {n_lv * J - 1}), q -> IF(element_at(st.ab, {nq} + 1), "
                          f"element_at(transform(ARRAY[{datos_q}], L -> {gan_arr(izq)}), 1), {vacio}))")
        ramas[r0 + 3] = _estado(gt=gq(True), gf=gq(False))
        # ---- mejor corte por (nodo, variable) y por nodo: máximo de (ganancia redondeada, -variable, -intervalo, izquierda)
        GT, GF = "element_at(st.gt, q + 1)", "element_at(st.gf, q + 1)"
        mT, mF = f"array_max({GT})", f"array_max({GF})"
        pT, pF = f"array_position({GT}, {mT})", f"array_position({GF}, {mF})"
        izq = f"({mT} > {mF} OR ({mT} = {mF} AND {pT} <= {pF}))"
        bsel = f"IF({izq}, {pT}, {pF})"
        Pq = "element_at(st.pq, q + 1)"
        Gq, Hq, Cq = (f"element_at(st.{t}, {nq} + 1)" for t in ("tg", "th", "tn"))
        gl = f"(element_at({Pq}.gl, {bsel}) + IF({izq}, {Pq}.gm, 0E0))"
        hl = f"(element_at({Pq}.hl, {bsel}) + IF({izq}, {Pq}.hm, 0E0))"
        nl = f"(element_at({Pq}.nl, {bsel}) + IF({izq}, {Pq}.nm, 0))"
        gan = f"({gl} * {gl} / ({hl} + {lam}) + ({Gq} - {gl}) * ({Gq} - {gl}) / ({Hq} - {hl} + {lam}) - {Gq} * {Gq} / ({Hq} + {lam}))"
        cand = (f"IF(greatest({mT}, {mF}) = -infinity(), {centinela}, CAST(ROW(greatest({mT}, {mF}), -{jq}, -({bsel} - 1), {izq}, "
                f"{jq}, {bsel} - 1, {gl}, {hl}, {nl}, {gan}) AS {CAND}))")
        ramas[r0 + 4] = _estado(
            bs=f"transform(sequence(0, {n_lv - 1}), o -> array_max(transform(sequence(o * {J}, o * {J} + {J - 1}), q -> {cand})))",
            pq=VACIOS["pq"], gt=VACIOS["gt"], gf=VACIOS["gf"])
        # ---- actualizar el árbol y el nodo de cada ingreso
        bk, bp_ = f"element_at(st.bs, k - {lo} + 1)", f"element_at(st.bs, (k - 1) / 2 - {lo} + 1)"
        corta = lambda nuevo, campo: (f"transform(sequence(0, {S - 1}), k -> IF(k BETWEEN {lo} AND {hi}, IF({bk}.gan > {eps}, {nuevo}, "
                                      f"element_at(st.{campo}, k + 1)), element_at(st.{campo}, k + 1)))")
        hijo = lambda izq_, der, campo: (f"transform(sequence(0, {S - 1}), k -> IF(k BETWEEN {2 * lo + 1} AND {2 * hi + 2}, "
                                         f"IF({bp_}.gan > {eps}, IF(k % 2 = 1, {izq_}, {der}), element_at(st.{campo}, k + 1)), "
                                         f"element_at(st.{campo}, k + 1)))")
        padre = "element_at(st.{t}, (k - 1) / 2 + 1)"
        ni, bn = "element_at(st.nd, i)", f"element_at(st.bs, element_at(st.nd, i) - {lo} + 1)"
        bi = f"element_at(element_at(be, {bn}.j), i)"
        nd = (f"transform(sequence(1, {N}), i -> IF({ni} >= {lo}, IF({bn}.gan > {eps}, IF({bi} <= {bn}.b AND ({bi} >= 0 OR {bn}.izq), "
              f"2 * {ni} + 1, 2 * {ni} + 2), {ni}), {ni}))")
        ramas[r0 + 5] = _estado(
            nd=nd, feat=corta(f"{bk}.j", "feat"), bin=corta(f"{bk}.b", "bin"), izq=corta(f"{bk}.izq", "izq"),
            gan=corta(f"{bk}.gan", "gan"),
            tg=hijo(f"{bp_}.gl", f"{padre.format(t='tg')} - {bp_}.gl", "tg"),
            th=hijo(f"{bp_}.hl", f"{padre.format(t='th')} - {bp_}.hl", "th"),
            tn=hijo(f"{bp_}.nl", f"{padre.format(t='tn')} - {bp_}.nl", "tn"),
            ab=hijo("true", "true", "ab"), bs=VACIOS["bs"])
    # ---- hojas y cierre
    ramas[P - 2] = _estado(va=(f"transform(sequence(1, {S}), k -> IF(element_at(st.ab, k) AND element_at(st.feat, k) = 0, "
                               f"CAST(-{eta} * element_at(st.tg, k) / (element_at(st.th, k) + {lam}) AS real), CAST(NULL AS real)))"))
    btn = "element_at(element_at(bp, element_at(st.feat, nn + 1)), i)"
    hoja_p = (f"reduce(sequence(1, {D}), {Z}, (nn, z) -> IF(element_at(st.feat, nn + 1) = 0, nn, IF({btn} <= "
              f"element_at(st.bin, nn + 1) AND ({btn} >= 0 OR element_at(st.izq, nn + 1)), 2 * nn + 1, 2 * nn + 2)), nn -> nn)")
    ramas[P - 1] = _estado(
        f="zip_with(st.f, st.nd, (pf, pn) -> pf + element_at(st.va, pn + 1))",
        ft=f"IF({M} = 0, st.ft, transform(sequence(1, {M}), i -> element_at(st.ft, i) + element_at(st.va, {hoja_p} + 1)))",
        arb=f"st.arb || ARRAY[CAST(ROW(st.feat, st.bin, st.izq, st.va, st.tg, st.th, st.gan, st.tn) AS {ARB})]")
    # cada rama en su propia lambda: Trino compila cada lambda en un método aparte (una sola expresión con todas las ramas
    # supera el tamaño máximo de un método de la JVM: "expressions are too complex")
    rama = lambda v: f"element_at(transform(ARRAY[st], st -> {v}), 1)"
    caso = "CASE (it - 1) % " + str(P) + "".join(f"\n      WHEN {k} THEN {rama(v)}" for k, v in sorted(ramas.items())) + " END"
    inicio = _estado(f="f0e", ft="f0p", **{n: (VACIOS.get(n) or f"CAST(ARRAY[] AS {t})") for n, t in CAMPOS if n not in ("f", "ft")})
    inicio = inicio.replace("st.", "")
    salida = (f"st -> ROW(st.arb, st.perd || ARRAY[CAST(ROW({_perdida('st.f', 'ye')}, {_perdida('st.ft', 'yp')}) "
              "AS ROW(e double, p double))])")
    return f"reduce(sequence(1, {arboles * P}), {inicio},\n  (st, it) -> {caso},\n  {salida})"


def sql_entrenar(variables, contrato, p, arboles, previo=None):
    """La consulta. previo = None (primera: calcula los cortes) o (cortes, margen, árboles hechos en formato if-then)."""
    J = len(variables)
    nombres = [v.nombre for v in variables]
    base = E.sql_base(variables, contrato, p)
    x = "ARRAY[" + ", ".join(f"CAST(v{i:03d} AS real)" for i in range(J)) + "]"
    vacio_d, vacio_x = "CAST(ARRAY[] AS array(double))", "CAST(ARRAY[] AS array(array(real)))"
    agg = lambda filtro, col, vacio: f"coalesce(array_agg({col} ORDER BY episode_ref) FILTER (WHERE {filtro}), {vacio})"
    if previo is None:
        qs = f"transform(sequence(1, {p.bins - 1}), i -> i / {p.bins}E0)"
        q = "ARRAY[" + ", ".join(f"approx_percentile(CAST(CAST(v{i:03d} AS real) AS double), {qs}) FILTER (WHERE NOT prueba)"
                                 for i in range(J)) + "]"
        cuerpo = (",\nagg AS (  -- una fila: entrenamiento y prueba como arrays (orden de episode_ref) y cuantiles de cada variable\n"
                  f"  SELECT {agg('NOT prueba', 'y', vacio_d)} AS ye, {agg('prueba', 'y', vacio_d)} AS yp,\n"
                  f"         {agg('NOT prueba', x, vacio_x)} AS xe, {agg('prueba', x, vacio_x)} AS xp,\n         {q} AS q\n  FROM t\n),\n"
                  "cor AS (SELECT ye, yp, xe, xp, transform(q, a -> IF(a IS NULL, NULL, array_sort(array_distinct("
                  "transform(a, v -> CAST(CAST(v AS double) AS real)))))) AS cq FROM agg),\n"
                  "pre AS (SELECT ye, yp, xe, xp, transform(cq, a -> coalesce(transform(a, v -> CAST(v AS double)), ARRAY[0E0])) AS c,\n"
                  "               transform(cq, a -> coalesce(cardinality(a), 0)) AS kv,\n"
                  "               CAST(CAST(reduce(ye, 0E0, (s, v) -> s + v, s -> s) / cardinality(ye) AS real) AS double) AS tasa FROM cor),\n"
                  "pre2 AS (SELECT ye, yp, xe, xp, c, kv, CAST(ln(tasa / (1E0 - tasa)) AS real) AS m0 FROM pre),\n"
                  "dat AS (SELECT ye, yp, c, kv, m0, repeat(m0, cardinality(ye)) AS f0e, repeat(m0, cardinality(yp)) AS f0p,\n")
        fuente = "pre2"
    else:
        cortes, margen, hechos = previo
        trozos = [c_ or [0.0] for c_ in cortes]
        ccte = E.sql_cortes(cortes)
        if hechos:
            cte, rec, _ = B.sql_arboles({"precision": "float32", "arboles": hechos}, {n: i + 1 for i, n in enumerate(nombres)},
                                        "float32", "<")
            fm, junta = f"reduce({rec}, {E._real(margen)}, (s, h) -> s + h, s -> s)", " CROSS JOIN arboles a"
        else:
            cte, fm, junta = "", E._real(margen), ""
        cuerpo = (",\n" + cte + ccte + ",\n"
                  f"q AS (SELECT episode_ref, y, prueba, {x} AS x FROM t),\n"
                  f"qf AS (SELECT q.episode_ref, q.y, q.prueba, q.x, {fm} AS f FROM q{junta}),\n"
                  "agg AS (  -- una fila: entrenamiento y prueba como arrays con el margen del modelo hecho hasta ahora\n"
                  f"  SELECT {agg('NOT prueba', 'y', vacio_d)} AS ye, {agg('prueba', 'y', vacio_d)} AS yp,\n"
                  f"         {agg('NOT prueba', 'x', vacio_x)} AS xe, {agg('prueba', 'x', vacio_x)} AS xp,\n"
                  f"         {agg('NOT prueba', 'f', 'CAST(ARRAY[] AS array(real))')} AS fe, {agg('prueba', 'f', 'CAST(ARRAY[] AS array(real))')} AS fp\n"
                  "  FROM qf\n),\n"
                  f"pre2 AS (SELECT a.ye, a.yp, a.xe, a.xp, a.fe, a.fp, k.c, ARRAY[{', '.join(str(len(c_)) for c_ in cortes)}] AS kv, "
                  f"{E._real(margen)} AS m0 FROM agg a CROSS JOIN cortes k),\n"
                  "dat AS (SELECT ye, yp, c, kv, m0, fe AS f0e, fp AS f0p,\n")
        fuente = "pre2"
    bins = lambda xs: (f"transform(sequence(1, {J}), j -> transform({xs}, xx -> coalesce(width_bucket(CAST(element_at(xx, j) AS double), "
                       "element_at(c, j)), -1)))")
    cuerpo += f"         {bins('xe')} AS be,\n         {bins('xp')} AS bp\n  FROM {fuente})\n"
    final = ("SELECT cardinality(ye) AS n, cardinality(yp) AS n_prueba, reduce(ye, 0E0, (s, v) -> s + v, s -> s) AS positivos, "
             f"m0, c, kv,\n       {sql_bucle(J, p, arboles)} AS r\nFROM dat")
    return base + cuerpo + final


def _arbol_doc(feat, bins, izq, val, gs, hes, gan, cnt, cortes, nombres, p, k=0):
    """Árbol del documento if-then desde los arrays por nodo (en anchura) que devuelve la consulta."""
    if feat[k] == 0:
        return {"hoja": f32(val[k]), "info": {"hessiano": float(hes[k]), "ingresos": int(cnt[k])}}
    j = int(feat[k]) - 1
    a = (feat, bins, izq, val, gs, hes, gan, cnt, cortes, nombres, p)
    return {"si": {"variable": nombres[j], "op": "<", "valor": float(cortes[j][int(bins[k])]), "nulo_a_entonces": bool(izq[k])},
            "entonces": _arbol_doc(*a, 2 * k + 1), "si_no": _arbol_doc(*a, 2 * k + 2),
            "info": {"ganancia": float(gan[k]), "hessiano": float(hes[k]), "ingresos": int(cnt[k]),
                     "peso": f32(-p.eta * float(gs[k]) / (float(hes[k]) + p.lambda_))}}


def entrenar(nombres, contrato=None, p=None, arboles_por_consulta=None, forzar=False, aviso=E._aviso):
    """Entrena con UNA consulta (o unas pocas, si arboles_por_consulta < p.arboles). Mismo resultado que entrenar.entrenar
    salvo los cortes (approx_percentile no es determinista entre ejecuciones)."""
    contrato, p = contrato or K.Contrato(), p or E.Parametros()
    variables, errores = K.variables_del_modelo(nombres, contrato)
    if errores:
        raise ValueError("; ".join(errores))
    por = arboles_por_consulta or p.arboles
    registro, t0, hechos, historial, previo, info = [], time.time(), [], [], None, None
    while previo is None or len(hechos) < p.arboles:
        n_t = min(por, p.arboles - len(hechos))
        sql = sql_entrenar(variables, contrato, p, n_t, previo)
        nb = len(sql.encode("utf-8"))
        if nb > C.MAX_SQL_BYTES and not forzar:
            raise ValueError(f"consulta de {nb:,} bytes: no cabe en AWS Athena ({C.MAX_SQL_BYTES:,})")
        df, reloj, st = cliente.ejecutar(sql, max_segundos=MAX_SEGUNDOS)
        registro.append({"que": f"árboles {len(hechos)}-{len(hechos) + n_t - 1}", "bytes_sql": nb, "filas_devueltas": len(df),
                         **cliente.resumen_stats(st, reloj)})
        fila = df.iloc[0]
        if previo is None:
            cortes = [list(map(float, c_)) if k_ else [] for c_, k_ in zip(fila["c"], fila["kv"])]
            tasa = f32(float(fila["positivos"]) / int(fila["n"]))
            margen = f32(float(fila["m0"]))
            info = {"entrenamiento": int(fila["n"]), "prueba": int(fila["n_prueba"]), "positivos": float(fila["positivos"])}
        arbs, perd = fila["r"]
        cortes_doc = [c_ or [0.0] for c_ in cortes]
        for a in arbs:
            hechos.append(_arbol_doc(*a, cortes_doc, nombres, p))
        nuevos = [{"arboles": len(historial) + i, "perdida_entrenamiento": float(e), "perdida_prueba": float(q_)}
                  for i, (e, q_) in enumerate(perd[:-1])]
        historial += nuevos
        aviso(f"  consulta {len(registro)}: {len(hechos)} árboles en {reloj:.1f} s (servidor {registro[-1]['servidor_s']} s, "
              f"{nb / 1024:.0f} KB); pérdida entrenamiento {float(perd[-1][0]):.5f}, prueba {float(perd[-1][1]):.5f}")
        previo = (cortes, margen, list(hechos))
    historial.append({"arboles": len(hechos), "perdida_entrenamiento": float(perd[-1][0]), "perdida_prueba": float(perd[-1][1])})
    if p.paciencia:  # parada temprana a posteriori, con la misma regla que entrenar.crecer (la consulta da la pérdida por árbol)
        mejor = (math.inf, 0)
        for h in historial:
            if h["perdida_prueba"] < mejor[0]:
                mejor = (h["perdida_prueba"], h["arboles"])
            if h["arboles"] - mejor[1] >= p.paciencia:
                break
        if mejor[1] < len(hechos):
            aviso(f"  parada temprana: el mejor fue con {mejor[1]} árboles")
            hechos, historial = hechos[:mejor[1]], historial[:mejor[1] + 1]
    doc = E.documento(hechos, margen, variables, contrato, "boosting_sql", p)
    return {"documento": doc, "xgboost": E.a_xgboost(doc, tasa), "cortes": cortes, "tasa": tasa, "margen": margen,
            "historial": historial, "consultas": registro, "segundos": time.time() - t0,
            "ingresos": {**info, "pacientes": None}}


# ------------------------------------------------------------------ las mismas cuentas en numpy (verificación bit a bit)
def _round_trino(x, d=9):
    f = 10.0 ** d
    return -(math.floor(-x * f + 0.5) / f) if x < 0 else math.floor(x * f + 0.5) / f


def _scan_np(a, bins):
    a = np.array(a)
    for k in desplazamientos(bins):
        a = a + np.concatenate([np.zeros(k, dtype=a.dtype), a])[:len(a)]
    return a


def local(X, y, prueba, orden, cortes, margen, p, arboles=None):
    """Mismo algoritmo que sql_bucle con numpy. X, y, prueba en cualquier orden; `orden` = episode_ref (la consulta
    ordena por él). Devuelve los árboles como listas (feat, bin, izq, val) por nodo en anchura y las pérdidas."""
    o = np.argsort(np.asarray(orden), kind="stable")
    X, y, prueba = np.asarray(X, float)[o], np.asarray(y, float)[o], np.asarray(prueba, bool)[o]
    Xe, ye, Xp, yp = X[~prueba], y[~prueba], X[prueba], y[prueba]
    c = [c_ or [0.0] for c_ in cortes]
    kv = [len(c_) for c_ in cortes]
    be, bp = E.intervalos(Xe, c), E.intervalos(Xp, c)
    D, S, KM, J, N = p.profundidad, 2 ** (p.profundidad + 1) - 1, p.bins + 1, len(c), len(ye)
    fe, fp = np.full(N, np.float32(margen), np.float32), np.full(len(yp), np.float32(margen), np.float32)
    arbs = []
    for _ in range(arboles if arboles is not None else p.arboles):
        f = fe.astype(np.float64)
        g = 1.0 / (1.0 + np.exp(-f)) - ye
        h = (1.0 / (1.0 + np.exp(-f))) * (1.0 - 1.0 / (1.0 + np.exp(-f)))
        nd = np.zeros(N, np.int64)
        feat, bn, iz = np.zeros(S, np.int64), np.zeros(S, np.int64), np.zeros(S, bool)
        G, H, Cn, ab, gn = np.zeros(S), np.zeros(S), np.zeros(S, np.int64), np.zeros(S, bool), np.zeros(S)
        G[0], H[0], Cn[0], ab[0] = np.cumsum(g)[-1], np.cumsum(h)[-1], N, True
        for lv in range(D):
            lo, hi = 2 ** lv - 1, 2 ** (lv + 1) - 2
            act = nd >= lo
            bests = {}
            for nn in range(lo, hi + 1):
                mejor = (-math.inf, 0, 0, False)
                datos = None
                if ab[nn]:
                    sel = act & (nd == nn)
                    for j in range(J):
                        Kj = len(c[j])
                        keys = be[sel, j] + 1  # 0 = ausente, b + 1 = intervalo b
                        gs = np.bincount(keys, weights=g[sel], minlength=Kj + 2)
                        hs = np.bincount(keys, weights=h[sel], minlength=Kj + 2)
                        cs = np.bincount(keys, minlength=Kj + 2).astype(np.int64)
                        GL, HL, NL = _scan_np(gs[1:], p.bins), _scan_np(hs[1:], p.bins), _scan_np(cs[1:], p.bins)
                        for q in range(1, 2 * kv[j] + 1):
                            b, ci = (q + 1) // 2, q % 2 == 1
                            gl = GL[b - 1] + (gs[0] if ci else 0.0)
                            hl = HL[b - 1] + (hs[0] if ci else 0.0)
                            nl = int(NL[b - 1] + (cs[0] if ci else 0))
                            ga = gl * gl / (hl + p.lambda_) + (G[nn] - gl) * (G[nn] - gl) / (H[nn] - hl + p.lambda_) - G[nn] * G[nn] / (H[nn] + p.lambda_)
                            if not (cs[b] > 0 and hl >= p.min_hijo and H[nn] - hl >= p.min_hijo and nl >= p.min_ingresos
                                    and Cn[nn] - nl >= p.min_ingresos):
                                continue
                            clave = (_round_trino(ga), -(j + 1), -(b - 1), ci)
                            if clave > mejor:
                                mejor, datos = clave, (j + 1, b - 1, ci, gl, hl, nl, ga)
                bests[nn] = datos
            nd_viejo = nd.copy()
            for nn, d in bests.items():
                if d is None or not d[6] > E.EPS_GANANCIA:
                    continue
                j1, b, ci, gl, hl, nl, ga = d
                feat[nn], bn[nn], iz[nn], gn[nn] = j1, b, ci, ga
                for k, (gg, hh, cc) in ((2 * nn + 1, (gl, hl, nl)), (2 * nn + 2, (G[nn] - gl, H[nn] - hl, Cn[nn] - nl))):
                    G[k], H[k], Cn[k], ab[k] = gg, hh, cc, True
                en = nd_viejo == nn
                bj = be[:, j1 - 1]
                va = (bj <= b) & ((bj >= 0) | ci)
                nd[en & va], nd[en & ~va] = 2 * nn + 1, 2 * nn + 2
        val = np.array([np.float32(-p.eta * G[k] / (H[k] + p.lambda_)) if (ab[k] and feat[k] == 0) else np.nan for k in range(S)],
                       dtype=np.float32)
        fe = (fe + val[nd]).astype(np.float32)
        nodo = np.zeros(len(yp), np.int64)
        for _ in range(D):
            j = feat[nodo]
            bt = np.where(j > 0, bp[np.arange(len(yp)), np.maximum(j, 1) - 1], 0)
            izq = (bt <= bn[nodo]) & ((bt >= 0) | iz[nodo])
            nodo = np.where(j == 0, nodo, np.where(izq, 2 * nodo + 1, 2 * nodo + 2))
        fp = (fp + val[nodo]).astype(np.float32)
        arbs.append((feat.tolist(), bn.tolist(), iz.tolist(), val.tolist()))
    return arbs
