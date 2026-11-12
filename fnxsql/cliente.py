"""Cliente mínimo de la API REST de Trino de FakeNex (solo biblioteca estándar + pandas) con salvaguardas.

Cada consulta: comprueba antes la salud del servidor (/v1/info), rechaza SQL de más de MAX_SQL_BYTES, reintenta mientras
el servidor arranca y devuelve el resultado, el tiempo de reloj y las estadísticas de Trino (planificación, CPU, memoria).
"""
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from . import config as C


class ErrorFakeNex(RuntimeError):
    pass


def _get(url, timeout=60):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=_cabeceras()), timeout=timeout).read())


def _cabeceras():
    return {"X-Trino-User": C.USUARIO, "X-Trino-Catalog": C.CATALOGO, "X-Trino-Schema": C.ESQUEMA,
            "X-Trino-Source": "fnxsql", "Content-Type": "text/plain; charset=utf-8"}


def salud():
    """(activo y arrancado, uptime) según /v1/info; (False, motivo) si no responde."""
    try:
        i = json.loads(urllib.request.urlopen(C.URL + "/v1/info", timeout=30).read())
        return i.get("state") == "ACTIVE" and not i.get("starting"), i.get("uptime")
    except Exception as e:  # red, 502 del proxy...
        return False, str(e)


def esperar_salud(max_segundos=300):
    """Espera a que el servidor esté activo (p. ej. tras un reinicio); ErrorFakeNex si no lo está a tiempo."""
    t = time.time()
    while True:
        ok, info = salud()
        if ok:
            return info
        if time.time() - t > max_segundos:
            raise ErrorFakeNex(f"FakeNex no está disponible ({info}); se detiene la ejecución")
        time.sleep(10)


def ejecutar(sql, timeout=180):
    """Ejecuta una consulta de solo lectura: (DataFrame, segundos de reloj, stats de Trino)."""
    datos = sql.encode("utf-8")
    if len(datos) > C.MAX_SQL_BYTES:
        raise ErrorFakeNex(f"SQL de {len(datos) / 1024:.0f} KB: supera el límite de {C.MAX_SQL_BYTES / 1024:.0f} KB")
    esperar_salud()
    t = time.time()
    for intento in range(6):
        try:
            req = urllib.request.Request(C.URL + "/v1/statement", data=datos, headers=_cabeceras(), method="POST")
            r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (502, 503) and intento < 5:
                esperar_salud()
                continue
            raise
    cols, filas = None, []
    while True:
        if "error" in r:
            msg = r["error"].get("message", "")
            if "still initializing" in msg:
                esperar_salud()
                return ejecutar(sql, timeout)
            raise ErrorFakeNex(msg)
        cols = cols or [c["name"] for c in r.get("columns", [])] or None
        filas += r.get("data", [])
        if "nextUri" not in r:
            st = r.get("stats", {})
            if st.get("elapsedTimeMillis", 0) / 1000 > C.MAX_SEGUNDOS_CONSULTA:
                raise ErrorFakeNex(f"la consulta tardó {st['elapsedTimeMillis'] / 1000:.0f} s en el servidor (máximo "
                                   f"{C.MAX_SEGUNDOS_CONSULTA} s): se detiene la batería para no sobrecargarlo")
            return pd.DataFrame(filas, columns=cols), time.time() - t, st
        r = _get(r["nextUri"], timeout)


def resumen_stats(st, reloj):
    """Diccionario plano de tiempos y recursos de una consulta."""
    return {"reloj_s": round(reloj, 2), "servidor_s": round(st.get("elapsedTimeMillis", 0) / 1000, 2),
            "planificacion_s": round(st.get("planningTimeMillis", 0) / 1000, 2),
            "cpu_s": round(st.get("cpuTimeMillis", 0) / 1000, 2),
            "memoria_mb": round(st.get("peakMemoryBytes", 0) / 2**20, 1)}
