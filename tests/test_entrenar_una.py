"""Entrenamiento en UNA consulta, sin servidor: la consulta generada (límites del proxy y de Athena) y el algoritmo local."""
import numpy as np

from fnxsql import config, contrato as K, entrenar as E, entrenar_una as U, ir
from fnxsql.practicar import VARIABLES
from test_entrenar import NOMBRES, datos


def profundidad_lambdas(sql):
    """Máximo de lambdas anidadas (una lambda dura hasta que se cierra el paréntesis que la contiene)."""
    nivel, abiertas, maximo = 0, [], 0
    for i, ch in enumerate(sql):
        if ch == "(":
            nivel += 1
        elif ch == ")":
            nivel -= 1
            while abiertas and abiertas[-1] > nivel:
                abiertas.pop()
        elif sql.startswith("->", i):
            abiertas.append(nivel)
            maximo = max(maximo, len(abiertas))
    return maximo


def test_consulta_dentro_de_los_limites():
    vs, _ = K.variables_del_modelo(VARIABLES)
    for prof in (3, 4):
        p = E.Parametros(arboles=100, profundidad=prof)
        sql = U.sql_entrenar(vs, K.Contrato(), p, p.arboles)
        # el proxy de FakeNex tarda exponencialmente con lambdas anidadas: 8 niveles ya añaden ~3 s, 10 niveles ~30 s
        assert profundidad_lambdas(sql) <= 6
        assert len(sql.encode()) < config.MAX_SQL_BYTES
        assert f"reduce(sequence(1, {100 * U.pasos_por_arbol(prof)})" in sql
    assert "approx_percentile" in sql and "multimap_from_entries" in sql and "array_max" in sql


def test_continuacion_lleva_cortes_y_modelo_en_binario():
    X, y, pids = datos()
    vs, _ = K.variables_del_modelo(NOMBRES)
    p = E.Parametros(arboles=4, profundidad=2, bins=16)
    prueba = E.es_prueba(pids, p)
    cortes = [E.cortes_de(np.nanquantile(X[~prueba, j], np.arange(1, 16) / 16)) for j in range(X.shape[1])]
    tm = E.margen_base(y[~prueba].sum(), (~prueba).sum())
    arbs = U.local(X, y, prueba, np.arange(len(y)), cortes, tm[1], p, arboles=2)
    docs = [U._arbol_doc(f, b, i, v, [0] * len(f), [1] * len(f), [0] * len(f), [0] * len(f), [c or [0.0] for c in cortes], NOMBRES, p)
            for f, b, i, v in arbs]
    sql = U.sql_entrenar(vs, K.Contrato(), p, 2, previo=(cortes, tm[1], docs))
    assert "approx_percentile" not in sql and "from_base64(" in sql and "CROSS JOIN arboles a" in sql
    assert profundidad_lambdas(sql) <= 6


def test_suma_por_duplicacion():
    a = np.random.default_rng(0).normal(size=300)
    assert np.allclose(U._scan_np(a, 256), np.cumsum(a), rtol=0, atol=1e-12)
    assert U._scan_np(np.arange(1, 6), 4).tolist() == [1, 3, 6, 10, 15]  # medido en FakeNex con la misma expresión


def test_redondeo_como_trino():
    assert U._round_trino(1.2345678905) == 1.234567891 and U._round_trino(-1.2345678905) == -1.234567891


def test_local_entrena_y_exporta():
    X, y, pids = datos()
    p = E.Parametros(arboles=6, profundidad=3, eta=0.3, bins=32)
    prueba = E.es_prueba(pids, p)
    cortes = [E.cortes_de(np.nanquantile(X[~prueba, j], np.arange(1, 32) / 32)) for j in range(X.shape[1])]
    tm = E.margen_base(y[~prueba].sum(), (~prueba).sum())
    arbs = U.local(X, y, prueba, np.arange(len(y)), cortes, tm[1], p)
    docs = [U._arbol_doc(f, b, i, v, [0] * len(f), [1] * len(f), [0] * len(f), [0] * len(f), [c or [0.0] for c in cortes], NOMBRES, p)
            for f, b, i, v in arbs]
    assert len(docs) == 6 and all("si" in d for d in docs)
    m = E.margen_modelo(docs, tm[1], X, NOMBRES).astype(float)
    ye = y[~prueba]
    ll = lambda f: float(np.mean(np.logaddexp(0, f) - ye * f))
    assert ll(m[~prueba]) < ll(np.full(len(ye), tm[1]))
    # el mismo algoritmo que entrenar.entrenar_local (solo cambia el orden de las sumas): mismo primer corte
    loc, _ = E.entrenar_local(X, y, prueba, cortes, tm, NOMBRES, p)
    assert (loc[0]["si"]["variable"], loc[0]["si"]["valor"]) == (docs[0]["si"]["variable"], docs[0]["si"]["valor"])
    vs, _ = K.variables_del_modelo(NOMBRES)
    doc = E.documento(docs, tm[1], vs, K.Contrato(), "x", p)
    p_ir = np.array([ir.evaluar(doc, dict(zip(NOMBRES, x))) for x in X[:50]])
    assert np.abs(p_ir - 1 / (1 + np.exp(-m[:50]))).max() < 1e-6
