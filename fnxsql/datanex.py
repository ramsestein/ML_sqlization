"""DataNex (AWS Athena): consultas autocontenidas en .txt para lanzarlas a mano y lectura de los CSV que devuelve Athena.

En DataNex se pega cada consulta en Athena y se descarga su resultado como CSV. Cada consulta es AISLADA: no necesita el
resultado de otra ni ningún paso intermedio. Los pacientes se eligen dentro de la propia consulta por un hash de
patient_ref (contrato.Cohorte), así que todas ven los mismos pacientes.

  preparar   escribe datanex/consultas/ (y su zip):
               00_variables_10_pacientes.txt        variables de los ingresos de 10 pacientes al azar
               NN_inferencia_<modelo>.txt           probabilidad de cada modelo para esos mismos ingresos
             y datanex/documentos/ (documento if-then float32 de cada modelo, para comprobar).
  comprobar  compara cada CSV de inferencia con el modelo evaluado en local sobre el CSV de variables: comprobacion.csv.
             Lee también CSV formateados por la herramienta de exportación (separadores de miles, valores redondeados):
             entonces compara a la precisión que traiga el CSV.
  ensayar    (opcional) ejecuta un .txt en FakeNex y guarda el CSV como lo daría Athena

datanex/resultados/ está en .gitignore: los CSV de DataNex son datos reales.
"""
import csv
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import binario as B, cliente, config as C, contrato as K, ir, sql as S

VARIABLES_CSV = "00_variables_10_pacientes"


def _escribir(ruta, texto):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _cabecera(titulo, extra=""):
    return f"-- {titulo}\n-- Consulta autocontenida generada por fnxsql: ejecutar en Athena (DataNex) y descargar el resultado como CSV.{extra}\n"


def leer_csv(ruta):
    """Filas de un CSV de Athena como lista de dicts (cabeceras en minúsculas; vacío = None)."""
    csv.field_size_limit(1 << 30)
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return [{k.strip().lower(): (v if v != "" else None) for k, v in fila.items()} for fila in csv.DictReader(f)]


def _doc_modelo(ruta, contrato):
    obj, origen = ir.cargar(ruta)
    variables, errores = K.variables_del_modelo(ir.nombres(obj, origen), contrato)
    if errores:
        raise ValueError(f"{ruta}: {'; '.join(errores)}")
    return ir.nombrar(ir.a_ir(obj, origen, variables, contrato, Path(ruta).stem))


def _num(x):
    """Número de un CSV, también formateado con separadores de miles ("1,521,782"); vacío = None."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == "":
        return None
    t = str(x).strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    return float(t)


def _id(x):
    return None if x is None else re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(x).strip())


def _decimales(x):
    """Decimales de un valor redondeado al exportar ("0.0085" -> 4, "1" -> 0, "6.7e-05" -> 6); None si trae la precisión
    completa (más de 6 cifras significativas, como da Athena)."""
    from decimal import Decimal, InvalidOperation
    try:
        d = Decimal(str(x).strip().replace(",", "")).as_tuple()
    except InvalidOperation:
        return None
    if len(d.digits) > 6:
        return None
    return max(0, -d.exponent)


def _tolerancia(doc):
    """Como `evaluar`: el SQL calcula exactamente el documento cuantizado (1e-6 en los modelos float32 de por sí, que
    suman en real; 1e-9 en los demás)."""
    return C.TOLERANCIA_FLOAT32 if doc["nucleo"].get("precision") == "float32" else C.TOLERANCIA_DOUBLE


# ------------------------------------------------------------------ preparar
def preparar(modelos, salida="datanex", pacientes=10, semilla_pacientes=1, precision="float32", zip_nombre="consultas_datanex.zip"):
    """modelos: lista de (nombre, ruta del modelo). Escribe las consultas .txt, los documentos, plan.json y el zip."""
    carpeta = Path(salida)
    contrato = K.Contrato()
    muestra = K.Cohorte(pacientes, semilla_pacientes)
    plan = {"esquema": C.ESQUEMA, "pacientes": {"n": pacientes, "semilla": semilla_pacientes}, "modelos": [], "variables": []}
    docs = [(nombre, _doc_modelo(ruta, contrato)) for nombre, ruta in modelos]
    for _, doc in docs:  # variables de todos los modelos (para una sola consulta de variables)
        for v in doc["variables"]:
            if v["nombre"] not in plan["variables"]:
                plan["variables"].append(v["nombre"])
    vs, _ = K.variables_del_modelo(plan["variables"], contrato)
    cons = carpeta / "consultas"
    if cons.exists():
        for f in cons.glob("*.txt"):
            f.unlink()
    sql = (_cabecera(f"Variables ({len(vs)}) de los ingresos de {pacientes} pacientes al azar (los mismos que en la inferencia)",
                     "\n-- Una fila por ingreso. Columnas v000, v001...: " + ", ".join(f"v{i:03d} = {n}" for i, n in enumerate(plan["variables"])))
           + K.sql_variables(vs, contrato, None, muestra) + "\nSELECT * FROM vars ORDER BY patient_ref, t0\n")
    _escribir(cons / f"{VARIABLES_CSV}.txt", sql)
    for k, (nombre, doc) in enumerate(docs, start=1):
        base = f"{k:02d}_inferencia_{nombre}"
        texto, info = S.compilar(doc, None, "auto", precision=precision, cohorte=muestra)
        _escribir(cons / f"{base}.txt", _cabecera(f"Inferencia: {nombre} ({doc['modelo'].get('clase', '')}) para los ingresos de "
                                                  f"{pacientes} pacientes al azar", "\n-- Una fila por ingreso: patient_ref, "
                                                  "episode_ref, ingreso, momento, probabilidad.") + texto)
        _escribir(carpeta / "documentos" / f"{nombre}.json", json.dumps(B.cuantizar(doc, precision), ensure_ascii=False))
        plan["modelos"].append({"nombre": nombre, "consulta": base, "documento": f"documentos/{nombre}.json",
                                "tipo_sql": info["tipo_sql"], "precision": precision, "bytes": info["bytes"]})
    _escribir(carpeta / "plan.json", json.dumps(plan, indent=1, ensure_ascii=False))
    _escribir(cons / "LEEME.txt", leeme(plan))
    with zipfile.ZipFile(carpeta / zip_nombre, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(cons.glob("*.txt")):
            z.write(f, f"consultas_datanex/{f.name}")
    return plan


def leeme(plan):
    filas = [f"  {VARIABLES_CSV}.txt\n      Las {len(plan['variables'])} variables de los ingresos de {plan['pacientes']['n']} pacientes al azar."]
    filas += [f"  {m['consulta']}.txt\n      Probabilidad del modelo {m['nombre']} para esos ingresos ({m['bytes'] / 1024:.0f} KB)."
              for m in plan["modelos"]]
    return f"""CONSULTAS PARA DATANEX (AWS Athena) - generadas por fnxsql
==========================================================

Cada fichero .txt es UNA consulta autocontenida: se pega en Athena, se ejecuta y se descarga su resultado como CSV.
No dependen unas de otras. Usan las tablas del esquema {plan['esquema']} (episodes, rc, labs, demographics), las mismas
que FakeNex.

Los pacientes se eligen dentro de cada consulta con un hash de patient_ref entre los que tienen algún ingreso HOSP o
HOSP_IQ: todas las consultas usan los mismos {plan['pacientes']['n']} pacientes.

Consultas
---------
{chr(10).join(filas)}

Resultados
----------
- Inferencia: una fila por ingreso (patient_ref, episode_ref, ingreso, momento, probabilidad). Los modelos son los de
  práctica de fnxsql (entrenados con datos sintéticos de FakeNex): es una prueba de funcionamiento, no de rendimiento.
- Variables: una fila por ingreso con las variables v000, v001... (la cabecera de la consulta dice cuál es cada una).

Para comprobarlo después (en local, con fnxsql): guardar cada CSV con el mismo nombre que su consulta (.txt -> .csv) en
una carpeta y ejecutar
  python -m fnxsql datanex comprobar --csv <carpeta>
que compara cada inferencia con el modelo evaluado en local sobre el CSV de variables (comprobacion.csv).
"""


# ------------------------------------------------------------------ comprobar
def comprobar(carpeta="datanex", csv_dir=None):
    carpeta = Path(carpeta)
    csv_dir = Path(csv_dir or carpeta / "resultados")
    plan = json.loads((carpeta / "plan.json").read_text(encoding="utf-8"))
    filas = []
    ruta_v = csv_dir / f"{VARIABLES_CSV}.csv"
    dv = None
    if ruta_v.exists():
        dv = pd.DataFrame(leer_csv(ruta_v))
        dv = dv.rename(columns={f"v{i:03d}": n.lower() for i, n in enumerate(plan["variables"])})
        filas.append({"consulta": VARIABLES_CSV, "resultado": "PASA" if len(dv) else "FALLA", "ingresos": len(dv),
                      "pacientes": dv.patient_ref.nunique() if len(dv) else 0, "detalle": f"{len(plan['variables'])} variables"})
    else:
        filas.append({"consulta": VARIABLES_CSV, "resultado": "FALTA", "detalle": f"no está {ruta_v}"})
    for m in plan["modelos"]:
        ruta = csv_dir / f"{m['consulta']}.csv"
        if not ruta.exists():
            filas.append({"consulta": m["consulta"], "resultado": "FALTA", "detalle": f"no está {ruta}"})
            continue
        doc = json.loads((carpeta / m["documento"]).read_text(encoding="utf-8"))
        ds = pd.DataFrame(leer_csv(ruta))
        fila = {"consulta": m["consulta"], "ingresos": len(ds), "pacientes": ds.patient_ref.nunique() if len(ds) else 0}
        if dv is None or not len(ds):
            fila.update(resultado="FALLA" if not len(ds) else "SIN COMPROBAR",
                        detalle="sin filas" if not len(ds) else "falta el CSV de variables")
            filas.append(fila)
            continue
        ds = ds.assign(episode_ref=ds.episode_ref.map(_id))
        mm = ds.merge(dv.assign(episode_ref=dv.episode_ref.map(_id)), on="episode_ref", how="outer", indicator=True,
                      suffixes=("", "_v"))
        solo = int((mm["_merge"] != "both").sum())
        mm = mm[mm["_merge"] == "both"]
        p_sql = mm.probabilidad.map(_num).to_numpy(dtype=float)
        p_loc = np.array([ir.evaluar(doc, {v["nombre"]: _num(r[v["nombre"].lower()]) for v in doc["variables"]})
                          for _, r in mm.iterrows()])
        dif = float(np.abs(p_sql - p_loc).max()) if len(mm) else math.nan
        # CSV formateado por la herramienta de exportación (p. ej. 0.0085): se compara a los decimales que trae cada valor
        dec = [_decimales(x) for x in mm.probabilidad]
        redondeado = any(d is not None for d in dec)
        if redondeado:
            ok_filas = [abs(a - b) <= 0.5 * 10 ** -d * (1 + 1e-9) if d is not None else abs(a - b) <= _tolerancia(doc)
                        for a, b, d in zip(p_sql, p_loc, dec)]
            tol, txt = None, f"CSV redondeado (de {min(d for d in dec if d is not None)} a {max(d for d in dec if d is not None)} decimales)"
            ok = len(mm) > 0 and solo == 0 and all(ok_filas)
        else:
            tol = _tolerancia(doc)
            ok, txt = len(mm) > 0 and solo == 0 and dif <= tol, f"tolerancia {tol:.0e}"
        fila.update(resultado="PASA" if ok else "FALLA", dif_max=dif, tolerancia=tol,
                    detalle=f"probabilidad del SQL = modelo en local (dif. máx. {dif:.1e}; {txt})"
                            + (f"; {solo} ingresos no coinciden con la consulta de variables" if solo else ""))
        filas.append(fila)
    out = pd.DataFrame(filas)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_dir / "comprobacion.csv", index=False)
    with pd.option_context("display.max_colwidth", 150, "display.width", 250):
        print(out[[c for c in ("consulta", "resultado", "ingresos", "detalle") if c in out]].to_string(index=False))
    print(f"-> {csv_dir / 'comprobacion.csv'}")
    return out


# ------------------------------------------------------------------ ensayo en FakeNex (mismo dialecto que Athena)
def ensayar(txt, ruta_csv):
    """Ejecuta un .txt en FakeNex y guarda su resultado como CSV al estilo de Athena (todo entre comillas, nulo = vacío)."""
    sql = Path(txt).read_text(encoding="utf-8")
    df, reloj, st = cliente.ejecutar(sql)
    Path(ruta_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(list(df.columns))
        for fila in df.itertuples(index=False):
            w.writerow(["" if v is None or (isinstance(v, float) and math.isnan(v)) else
                        (json.dumps(v) if isinstance(v, (list, dict)) else ("true" if v is True else "false" if v is False else v))
                        for v in fila])
    return len(df), cliente.resumen_stats(st, reloj)
