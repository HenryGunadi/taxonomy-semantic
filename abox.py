#!/usr/bin/env python3
"""
A-Box data acquisition + verification (Milestone 1)

A-Box = the raw dataset, here: Indonesian Supreme Court decisions
(Azzindani/ID_Supreme_Court_Parquet) stored in MySQL as two tables:

    documents  (id, klasifikasi, sub_klasifikasi)
    paragraphs (doc_id, position, tag, value)

This script
  1. verifies the tables are loaded correctly (integrity checks), and
  2. optionally exports a snapshot (documents.csv, a sample of full decisions
     as JSONL, and a JSON summary) to use as the "dataset A-Box" deliverable.

Usage
    pip install sqlalchemy pymysql
    python abox_verify.py                       # verify only
    python abox_verify.py --export abox_out     # verify + export snapshot
    python abox_verify.py --land                # also count land-related docs

Connection: edit DEFAULT_DB_URL below, pass --db, or set env ABOX_DB_URL.
Exit code: 0 if no FAIL, 1 if any FAIL.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

from sqlalchemy import bindparam, create_engine, inspect, text

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# XAMPP defaults: user root, empty password, database taxonomy_semantic
DEFAULT_DB_URL = os.getenv(
    "ABOX_DB_URL",
    "mysql+pymysql://root:@localhost:3306/taxonomy_semantic?charset=utf8mb4",
)
EXPECTED_DOCS = 22630  # row count shown on the Hugging Face dataset page
SOURCE = "https://huggingface.co/datasets/Azzindani/ID_Supreme_Court_Parquet"

# Tags seen in the dataset samples. Anything else is reported (WARN), not failed.
KNOWN_TAGS = {
    "kepala_putusan", "identitas", "riwayat_penahanan", "riwayat_perkara",
    "riwayat_tuntutan", "riwayat_dakwaan", "fakta", "fakta_hukum",
    "pertimbangan_hukum", "amar_putusan", "penutup",
}
KNOWN_KLASIFIKASI = {"pidana-umum", "pidana-khusus"}

# Keywords for the optional land-related subset count (--land)
LAND_KEYWORDS = ["tanah", "sertifikat", "penyerobotan", "pertanahan", "hak milik"]


# --------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.items = []

    def add(self, status, check, detail=""):
        self.items.append({"status": status, "check": check, "detail": detail})
        line = f"[{status}] {check}"
        if detail:
            line += f" - {detail}"
        print(line, flush=True)

    def count(self, status):
        return sum(1 for i in self.items if i["status"] == status)


def one(conn, sql, **params):
    return conn.execute(text(sql), params).scalar()


def rows(conn, sql, **params):
    return conn.execute(text(sql), params).fetchall()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def run_checks(engine, expected_docs, land):
    rep = Report()
    summary = {}
    CL = "CHAR_LENGTH" if engine.dialect.name == "mysql" else "LENGTH"

    # --- 1. schema -----------------------------------------------------
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    missing = {"documents", "paragraphs"} - tables
    if missing:
        rep.add("FAIL", "Tables exist", f"missing: {sorted(missing)}; found: {sorted(tables)}")
        return rep, summary
    rep.add("PASS", "Tables exist", "documents, paragraphs")

    doc_cols = {c["name"]: c for c in insp.get_columns("documents")}
    par_cols = {c["name"]: c for c in insp.get_columns("paragraphs")}
    need_d = {"id", "klasifikasi", "sub_klasifikasi"}
    need_p = {"doc_id", "position", "tag", "value"}
    if need_d - set(doc_cols) or need_p - set(par_cols):
        rep.add("FAIL", "Columns exist",
                f"documents missing {sorted(need_d - set(doc_cols))}, "
                f"paragraphs missing {sorted(need_p - set(par_cols))}")
        return rep, summary
    rep.add("PASS", "Columns exist")

    if engine.dialect.name == "mysql":
        vtype = type(par_cols["value"]["type"]).__name__.upper()
        if vtype == "LONGTEXT":
            rep.add("PASS", "paragraphs.value column type", "LONGTEXT")
        elif vtype == "MEDIUMTEXT":
            rep.add("WARN", "paragraphs.value column type", "MEDIUMTEXT (16 MB) - LONGTEXT recommended")
        else:
            rep.add("FAIL", "paragraphs.value column type", f"{vtype} will truncate long decisions; use LONGTEXT")

    with engine.connect() as conn:
        # --- 2. row counts ---------------------------------------------
        n_docs = one(conn, "SELECT COUNT(*) FROM documents")
        n_par = one(conn, "SELECT COUNT(*) FROM paragraphs")
        summary["documents"] = n_docs
        summary["paragraphs"] = n_par
        if expected_docs:
            if n_docs == expected_docs:
                rep.add("PASS", "documents row count", f"{n_docs} (matches expected {expected_docs})")
            else:
                rep.add("FAIL", "documents row count", f"{n_docs}, expected {expected_docs} (partial or duplicated load?)")
        else:
            rep.add("INFO", "documents row count", str(n_docs))
        if n_par == 0:
            rep.add("FAIL", "paragraphs row count", "0 rows")
            return rep, summary
        rep.add("INFO", "paragraphs row count", f"{n_par} (avg {n_par / max(n_docs, 1):.1f} sections per decision)")

        # --- 3. keys / referential integrity ---------------------------
        dup = one(conn, "SELECT COUNT(*) FROM (SELECT id FROM documents GROUP BY id HAVING COUNT(*) > 1) t")
        rep.add("PASS" if dup == 0 else "FAIL", "Duplicate document ids", str(dup))

        bad_id = one(conn, "SELECT COUNT(*) FROM documents WHERE id IS NULL OR LENGTH(id) <> 32")
        rep.add("PASS" if bad_id == 0 else "FAIL", "Document ids are 32 chars", f"{bad_id} bad")

        orphan = one(conn, "SELECT COUNT(*) FROM paragraphs p LEFT JOIN documents d ON d.id = p.doc_id WHERE d.id IS NULL")
        rep.add("PASS" if orphan == 0 else "FAIL", "Orphan paragraphs (no parent document)", str(orphan))

        empty_docs = one(conn, "SELECT COUNT(*) FROM documents d LEFT JOIN paragraphs p ON p.doc_id = d.id WHERE p.doc_id IS NULL")
        rep.add("PASS" if empty_docs == 0 else "WARN", "Documents with no paragraphs", f"{empty_docs} (empty in source if it matches the parquet)")

        gaps = one(conn, """
            SELECT COUNT(*) FROM (
                SELECT doc_id FROM paragraphs GROUP BY doc_id
                HAVING MIN(position) <> 0 OR MAX(position) <> COUNT(*) - 1
            ) t""")
        rep.add("PASS" if gaps == 0 else "FAIL", "Section order intact (positions 0..n-1 per document)", f"{gaps} documents with gaps")

        # --- 4. tags ---------------------------------------------------
        bad_tag = one(conn, "SELECT COUNT(*) FROM paragraphs WHERE tag IS NULL OR TRIM(tag) = ''")
        rep.add("PASS" if bad_tag == 0 else "FAIL", "Empty/NULL tags", str(bad_tag))

        tag_rows = rows(conn, f"""
            SELECT tag, COUNT(*) AS n, COUNT(DISTINCT doc_id) AS docs,
                   MIN({CL}(value)) AS mn, AVG({CL}(value)) AS av, MAX({CL}(value)) AS mx
            FROM paragraphs GROUP BY tag ORDER BY n DESC""")
        found = {r[0] for r in tag_rows}
        unknown = sorted(found - KNOWN_TAGS)
        if unknown:
            rep.add("WARN", "Unknown tags (not in KNOWN_TAGS)", ", ".join(map(str, unknown)))
        else:
            rep.add("PASS", "All tags are known", f"{len(found)} distinct")

        dup_tags = one(conn, """
            SELECT COUNT(*) FROM (
                SELECT doc_id, tag FROM paragraphs GROUP BY doc_id, tag HAVING COUNT(*) > 1
            ) t""")
        rep.add("INFO", "Documents repeating the same tag", f"{dup_tags} (doc, tag) pairs")

        no_amar = one(conn, """
            SELECT COUNT(*) FROM documents d WHERE NOT EXISTS (
                SELECT 1 FROM paragraphs p WHERE p.doc_id = d.id AND p.tag = 'amar_putusan')""")
        rep.add("INFO", "Documents without an 'amar_putusan' (verdict) section", str(no_amar))

        print("\n  Tag distribution:")
        print(f"  {'tag':<22}{'sections':>10}{'docs':>9}{'min_len':>9}{'avg_len':>10}{'max_len':>11}")
        summary["tags"] = {}
        for tag, n, docs, mn, av, mx in tag_rows:
            print(f"  {str(tag):<22}{n:>10}{docs:>9}{int(mn or 0):>9}{int(av or 0):>10}{int(mx or 0):>11}")
            summary["tags"][str(tag)] = {"sections": int(n), "docs": int(docs),
                                         "min_len": int(mn or 0), "avg_len": int(av or 0), "max_len": int(mx or 0)}
        print()

        # --- 5. text content (single pass over paragraphs) -------------
        print("[....] scanning text values (may take a while)", flush=True)
        r = conn.execute(text(f"""
            SELECT
              SUM(CASE WHEN value IS NULL OR {CL}(TRIM(REPLACE(REPLACE(value, CHAR(10), ''), CHAR(13), ''))) = 0
                       THEN 1 ELSE 0 END),
              SUM(CASE WHEN value LIKE :garb THEN 1 ELSE 0 END),
              SUM(CASE WHEN LENGTH(value) = 65535 THEN 1 ELSE 0 END),
              SUM(CASE WHEN {CL}(value) > 65535 THEN 1 ELSE 0 END),
              MIN({CL}(value)), AVG({CL}(value)), MAX({CL}(value))
            FROM paragraphs"""), {"garb": "%dtype=object%"}).fetchone()
        blank, garbage, cut, longer, mn, av, mx = [(x or 0) for x in r]
        blank, garbage, cut, longer = int(blank), int(garbage), int(cut), int(longer)

        rep.add("PASS" if blank == 0 else "WARN", "Blank section values", str(blank))
        rep.add("PASS" if garbage == 0 else "FAIL", "Stringified-numpy leftovers ('dtype=object') in text", str(garbage))
        rep.add("PASS" if cut == 0 else "FAIL", "Values exactly 65535 bytes (TEXT truncation signature)", str(cut))
        rep.add("INFO", "Values longer than 65535 chars", f"{longer} (proves long text is not truncated)" if longer else "0")
        rep.add("INFO", "Section text length (chars)", f"min {int(mn)}, avg {int(av)}, max {int(mx)}")
        summary["text_length"] = {"min": int(mn), "avg": int(av), "max": int(mx), "over_65535": longer}

        # --- 6. classification columns ---------------------------------
        k_rows = rows(conn, "SELECT klasifikasi, sub_klasifikasi, COUNT(*) AS n FROM documents "
                            "GROUP BY klasifikasi, sub_klasifikasi ORDER BY n DESC")
        summary["klasifikasi"] = [{"klasifikasi": a, "sub_klasifikasi": b, "docs": int(n)} for a, b, n in k_rows]
        missing_k = one(conn, "SELECT COUNT(*) FROM documents WHERE klasifikasi IS NULL OR TRIM(klasifikasi) = '' "
                              "OR sub_klasifikasi IS NULL OR TRIM(sub_klasifikasi) = ''")
        rep.add("PASS" if missing_k == 0 else "FAIL", "NULL/empty klasifikasi or sub_klasifikasi", str(missing_k))
        for a, b, n in k_rows:
            print(f"  klasifikasi={a!s:<14} sub_klasifikasi={b!s:<14} docs={n}")
        unk_k = sorted({str(a) for a, _, _ in k_rows} - KNOWN_KLASIFIKASI)
        if unk_k:
            rep.add("INFO", "klasifikasi values outside the two seen in the samples", ", ".join(unk_k))
        else:
            rep.add("INFO", "klasifikasi coverage", "criminal only (pidana-umum / pidana-khusus); no perdata/TUN rows")
        print()

        # --- 7. optional: land-related subset --------------------------
        if land:
            print("[....] counting land-related documents (may take a while)", flush=True)
            clause = " OR ".join(f"value LIKE :k{i}" for i in range(len(LAND_KEYWORDS)))
            params = {f"k{i}": f"%{kw}%" for i, kw in enumerate(LAND_KEYWORDS)}
            n_land = one(conn, f"SELECT COUNT(DISTINCT doc_id) FROM paragraphs WHERE {clause}", **params)
            rep.add("INFO", "Documents mentioning land keywords", f"{n_land} of {n_docs} ({', '.join(LAND_KEYWORDS)})")
            summary["land_docs"] = int(n_land)

    return rep, summary


# --------------------------------------------------------------------------
# Export (the "dataset A-Box" snapshot)
# --------------------------------------------------------------------------
def export_snapshot(engine, out_dir, sample_size, seed, report, summary):
    os.makedirs(out_dir, exist_ok=True)
    with engine.connect() as conn:
        docs = rows(conn, "SELECT id, klasifikasi, sub_klasifikasi FROM documents ORDER BY id")

        # 1) full documents index
        p_docs = os.path.join(out_dir, "abox_documents.csv")
        with open(p_docs, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "klasifikasi", "sub_klasifikasi"])
            w.writerows(docs)

        # 2) reproducible sample of complete decisions, stratified by klasifikasi
        groups = {}
        for did, k, _ in docs:
            groups.setdefault(k, []).append(did)
        per_group = math.ceil(sample_size / max(len(groups), 1))
        rng = random.Random(seed)
        picked = []
        for k in sorted(groups, key=str):
            ids = groups[k]
            picked += rng.sample(ids, min(per_group, len(ids)))
        meta = {d[0]: d for d in docs}

        stmt = text("SELECT doc_id, position, tag, value FROM paragraphs "
                    "WHERE doc_id IN :ids ORDER BY doc_id, position"
                    ).bindparams(bindparam("ids", expanding=True))
        by_doc = {}
        for doc_id, pos, tag, val in conn.execute(stmt, {"ids": picked}):
            by_doc.setdefault(doc_id, []).append({"position": int(pos), "tag": tag, "value": val})

        p_sample = os.path.join(out_dir, "abox_sample.jsonl")
        with open(p_sample, "w", encoding="utf-8") as f:
            for did in picked:
                _, k, sk = meta[did]
                f.write(json.dumps({"id": did, "klasifikasi": k, "sub_klasifikasi": sk,
                                    "paragraphs": by_doc.get(did, [])}, ensure_ascii=False) + "\n")

    # 3) summary + verification report
    p_sum = os.path.join(out_dir, "abox_summary.json")
    with open(p_sum, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": SOURCE,
            "storage": "MySQL tables documents, paragraphs (full A-Box)",
            "sample": {"size": len(picked), "seed": seed, "stratified_by": "klasifikasi"},
            "stats": summary,
            "checks": report.items,
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nExported to ./{out_dir}/")
    print(f"  {p_docs}   ({len(docs)} rows)")
    print(f"  {p_sample}  ({len(picked)} complete decisions)")
    print(f"  {p_sum}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Verify (and optionally export) the court-decision A-Box in MySQL.")
    ap.add_argument("--db", default=DEFAULT_DB_URL, help="SQLAlchemy URL (default: XAMPP taxonomy_semantic)")
    ap.add_argument("--expected-docs", type=int, default=EXPECTED_DOCS,
                    help=f"expected documents row count (default {EXPECTED_DOCS}; 0 to skip)")
    ap.add_argument("--export", metavar="DIR", help="also export a snapshot into DIR")
    ap.add_argument("--sample-size", type=int, default=50, help="decisions in the exported sample (default 50)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    ap.add_argument("--land", action="store_true", help="also count documents matching land keywords")
    args = ap.parse_args()

    try:
        engine = create_engine(args.db)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] Cannot connect to the database: {e}")
        print("       Is MySQL running in XAMPP, and are user/password/database in the URL correct?")
        sys.exit(1)

    print(f"Connected: {engine.url.render_as_string(hide_password=True)}\n")
    report, summary = run_checks(engine, args.expected_docs, args.land)

    print(f"\nResult: {report.count('PASS')} PASS, {report.count('WARN')} WARN, "
          f"{report.count('FAIL')} FAIL, {report.count('INFO')} INFO")

    if args.export and report.count("FAIL") == 0:
        export_snapshot(engine, args.export, args.sample_size, args.seed, report, summary)
    elif args.export:
        print("\nExport skipped: fix the FAIL items first.")

    if report.count("FAIL"):
        print("A-Box verification FAILED.")
        sys.exit(1)
    print("A-Box verification OK.")


if __name__ == "__main__":
    main()