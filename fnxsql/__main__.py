"""Orquestador: python -m fnxsql <orden> [opciones]   (o `fnxsql` si se instala con pip).

Órdenes:
  evaluar MODELO        todo el circuito con validaciones automáticas e informe (el uso principal: traigo mi modelo)
  nombres MODELO        comprueba que las variables siguen la gramática de DataNex / FakeNex (y existen en FakeNex)
  normalizar MODELO     copia del modelo con las variables renombradas a códigos de DataNex (renombres del contrato)
  convertir MODELO      solo el documento if-then (sin servidor)
  sql MODELO            solo la consulta SQL para unos pacientes (sin ejecutarla)
  practicar             entrena modelos de ejemplo con datos de FakeNex (para practicar)
  demo                  practicar + evaluar todos los modelos de ejemplo: la ejecución "sola" de principio a fin
"""
import argparse
import json
import sys
from pathlib import Path

from . import __version__, config as C


def _pacientes(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def _lotes(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fnxsql", description="Modelos de ML -> reglas if-then -> UNA consulta SQL en FakeNex, "
                                 "con validación automática.")
    ap.add_argument("--version", action="version", version=f"fnxsql {__version__}")
    ap.add_argument("--url", help=f"servidor FakeNex (por defecto {C.URL}; también FAKENEX_URL)")
    ap.add_argument("--usuario", help="X-Trino-User (por defecto FAKENEX_USER o uno aleatorio fnxsql-xxxxxxxx)")
    sub = ap.add_subparsers(dest="orden", required=True)

    def comunes(p, servidor=True):
        p.add_argument("modelo", help="XGBoost .json (save_model) o scikit-learn .joblib / .pkl")
        p.add_argument("--contrato", help="JSON con momento_horas, ventana_horas, si_falta, constantes, renombres")
        if servidor:
            p.add_argument("--pacientes", help="patient_ref separados por comas (por defecto, una muestra de la base)")
            p.add_argument("--muestra", type=int, default=3, help="nº de pacientes al azar si no se dan (por defecto 3)")
            p.add_argument("--sql", default="auto", choices=["auto", "arrays", "case", "terminos"],
                           help="representación SQL (auto = la más robusta: arrays para árboles y redes)")
            p.add_argument("--precision", default=C.PRECISION, choices=["float32", "float64", "int8"],
                           help="parámetros en float32 (por defecto; error <= ~1e-6), float64 (exacto) o int8 (más pequeño; "
                                "error ~1e-3 a 1e-1 en la probabilidad)")
            p.add_argument("--forzar", action="store_true",
                           help="permite superar el límite de términos y el de AWS Athena (hasta el del proxy de FakeNex)")
            p.add_argument("--semilla", type=int, default=C.SEMILLA)

    p = sub.add_parser("evaluar", help="circuito completo con validaciones e informe")
    comunes(p)
    p.add_argument("--lotes", default="", help="tamaños de lote para medir tiempos, p. ej. 1,10,50")
    p.add_argument("--salida", default="salidas", help="carpeta de informes (por defecto ./salidas)")
    p = sub.add_parser("nombres", help="comprueba los nombres de las variables")
    comunes(p, servidor=False)
    p.add_argument("--sin-servidor", action="store_true", help="solo la gramática, sin consultar FakeNex")
    p = sub.add_parser("normalizar", help="renombra las variables a códigos de DataNex")
    comunes(p, servidor=False)
    p.add_argument("--nombres", help="JSON con la lista feature_names (si el modelo no la trae)")
    p.add_argument("--salida", required=True, help="fichero del modelo normalizado")
    p = sub.add_parser("convertir", help="documento if-then (sin servidor)")
    comunes(p, servidor=False)
    p.add_argument("--salida", required=True, help="fichero JSON de salida")
    p = sub.add_parser("sql", help="consulta SQL para unos pacientes (sin ejecutarla)")
    comunes(p)
    p.add_argument("--salida", help="fichero .sql (por defecto, a la salida estándar)")
    p = sub.add_parser("practicar", help="entrena modelos de ejemplo con datos de FakeNex")
    p.add_argument("--salida", default="salidas/practica")
    p.add_argument("--modelos", help="subconjunto separado por comas (logistica, arbol, random_forest, boosting, "
                                     "red_pequena, red_mediana, red_grande, xgboost)")
    p = sub.add_parser("demo", help="practicar + evaluar todos los modelos de ejemplo")
    p.add_argument("--salida", default="salidas")
    p.add_argument("--modelos", help="subconjunto de modelos de práctica")
    p.add_argument("--muestra", type=int, default=3)
    p.add_argument("--lotes", default="", help="tamaños de lote para medir tiempos, p. ej. 1,10,50")
    p.add_argument("--sql", default="auto", choices=["auto", "arrays", "case", "terminos"])
    p.add_argument("--precision", default=C.PRECISION, choices=["float32", "float64", "int8"])
    a = ap.parse_args(argv)
    if a.url:
        C.URL = a.url
    if a.usuario:
        C.USUARIO = a.usuario

    from . import contrato as K, ir, validar
    if a.orden == "evaluar":
        inf = validar.evaluar(a.modelo, a.contrato, _pacientes(a.pacientes), a.muestra, a.sql, _lotes(a.lotes), a.salida,
                              a.forzar, a.semilla, precision=a.precision)
        return 0 if inf.ok else 1
    if a.orden == "nombres":
        obj, tipo = ir.cargar(a.modelo)
        contrato = K.Contrato.leer(a.contrato)
        variables, errores = K.variables_del_modelo(ir.nombres(obj, tipo), contrato)
        for v in variables:
            print(f"  {v.nombre:45s} {v.fuente:10s} {v.codigo or '':16s} {v.valor or '':20s} {v.estadistico or ''}")
        for e in errores:
            print(f"ERROR: {e}")
        if not a.sin_servidor and not errores:
            e2, av = K.comprobar_en_fakenex(variables)
            for x in av:
                print(f"AVISO: {x}")
            for x in e2:
                print(f"ERROR: {x}")
            errores += e2
        print(f"{len(variables)} variables reconocidas, {len(errores)} errores")
        return 1 if errores else 0
    if a.orden == "normalizar":
        from .normalizar import normalizar
        normalizar(a.modelo, a.salida, a.nombres, a.contrato)
        return 0
    if a.orden in ("convertir", "sql"):
        obj, tipo = ir.cargar(a.modelo)
        contrato = K.Contrato.leer(a.contrato)
        variables, errores = K.variables_del_modelo(ir.nombres(obj, tipo), contrato)
        if errores:
            print("\n".join(f"ERROR: {e}" for e in errores))
            return 1
        doc = ir.nombrar(ir.a_ir(obj, tipo, variables, contrato, Path(a.modelo).stem))
        if a.orden == "convertir":
            Path(a.salida).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{ir.resumen(doc)} -> {a.salida}")
            return 0
        from . import sql as S
        pids = _pacientes(a.pacientes) or validar.pacientes_base(a.muestra, a.semilla)
        texto, info = S.compilar(doc, pids, a.sql, a.forzar, a.precision)
        if a.salida:
            Path(a.salida).write_text(texto, encoding="utf-8")
            print(f"{info} -> {a.salida}")
        else:
            print(texto)
        return 0
    if a.orden == "practicar":
        from .practicar import practicar
        practicar(a.salida, a.modelos.split(",") if a.modelos else None)
        return 0
    if a.orden == "demo":
        from .practicar import practicar
        res = practicar(Path(a.salida) / "practica", a.modelos.split(",") if a.modelos else None)
        filas = []
        for nombre, m in res["modelos"].items():
            inf = validar.evaluar(m["fichero"], None, None, a.muestra, a.sql, _lotes(a.lotes), a.salida,
                                  solo_pacientes=res["pacientes_test"], precision=a.precision)
            sqlinfo = inf.d["datos"].get("sql", {})
            filas.append({"modelo": nombre, "resultado": "PASA" if inf.ok else "FALLA", "auc_test": m["auc_test"],
                          "sql": f"{sqlinfo.get('tipo_sql')} {sqlinfo.get('precision', '')}".strip(),
                          "bytes": sqlinfo.get("bytes", 0),
                          "reloj_s": sqlinfo.get("reloj_s"), "planificacion_s": sqlinfo.get("planificacion_s")})
        resumen = Path(a.salida) / "resumen_demo.md"
        lineas = ["# Demo fnxsql", "", f"Límite de AWS Athena: {C.MAX_SQL_BYTES:,} bytes por consulta.", "",
                  "| Modelo | Resultado | AUC test | SQL | Bytes | Reloj (s) | Planificación (s) |",
                  "|---|---|---|---|---|---|---|"]
        lineas += [f"| {f['modelo']} | {f['resultado']} | {f['auc_test']} | {f['sql']} | {f['bytes']:,} | {f['reloj_s']} | "
                   f"{f['planificacion_s']} |" for f in filas]
        resumen.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lineas) + f"\n\n-> {resumen}")
        return 0 if all(f["resultado"] == "PASA" for f in filas) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
