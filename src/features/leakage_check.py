# -*- coding: utf-8 -*-
"""
leakage_check.py
================
Verifica vazamento de dados temporais (data leakage) nas tabelas secundárias
em relação à data de aplicação registrada em application_train.csv.

Convenção Home Credit: todas as colunas DAYS_* são negativos ou zero,
representando dias ANTES da aplicação. Valores > 0 indicam eventos FUTUROS
(posteriores à aplicação) — potencial leakage.

Exceções conhecidas:
  - 365243 em previous_application.DAYS_* → código de dado ausente (tratado como NaN)

Uso:
    python src/features/leakage_check.py
"""

import sys
import warnings
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

MISSING_SENTINEL = 365243  # código de dado ausente em previous_application


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.2f}%" if total > 0 else "N/A"


def check_column(df: pd.DataFrame, col: str, sentinel: int | None = None) -> dict:
    """Retorna contagem de valores > 0 em col (excluindo sentinel se fornecido)."""
    series = df[col].copy()
    if sentinel is not None:
        series = series.replace(sentinel, pd.NA)
    positive = series.dropna()
    positive = positive[positive > 0]
    return {
        "column": col,
        "total_rows": len(df),
        "suspect_rows": len(positive),
        "pct": _pct(len(positive), len(df)),
        "sample_values": sorted(positive.unique().tolist())[:10] if len(positive) > 0 else [],
    }


# ---------------------------------------------------------------------------
# Checks por tabela
# ---------------------------------------------------------------------------

def check_bureau(path: Path) -> list[dict]:
    print("  Carregando bureau.csv...")
    df = pd.read_csv(path, usecols=["SK_ID_CURR", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE",
                                     "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"])
    cols = [c for c in ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE",
                         "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"] if c in df.columns]
    results = []
    for col in cols:
        results.append(check_column(df, col))
    return results


def check_installments(path: Path) -> list[dict]:
    print("  Carregando installments_payments.csv...")
    df = pd.read_csv(path, usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
    results = []
    for col in ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]:
        results.append(check_column(df, col))
    return results


def check_previous_application(path: Path) -> list[dict]:
    print("  Carregando previous_application.csv...")
    cols_to_load = ["SK_ID_CURR", "DAYS_DECISION", "DAYS_FIRST_DRAWING",
                    "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                    "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    df = pd.read_csv(path, usecols=lambda c: c in cols_to_load)
    results = []
    day_cols = [c for c in df.columns if c.startswith("DAYS_")]
    for col in day_cols:
        results.append(check_column(df, col, sentinel=MISSING_SENTINEL))
    return results


def check_pos_cash(path: Path) -> list[dict]:
    print("  Carregando POS_CASH_balance.csv...")
    df = pd.read_csv(path, usecols=["SK_ID_CURR", "MONTHS_BALANCE"])
    # MONTHS_BALANCE: esperado <= 0 (meses antes da aplicação)
    series = df["MONTHS_BALANCE"]
    positive = series[series > 0]
    return [{
        "column": "MONTHS_BALANCE",
        "total_rows": len(df),
        "suspect_rows": len(positive),
        "pct": _pct(len(positive), len(df)),
        "sample_values": sorted(positive.unique().tolist())[:10] if len(positive) > 0 else [],
    }]


def check_credit_card(path: Path) -> list[dict]:
    print("  Carregando credit_card_balance.csv...")
    df = pd.read_csv(path, usecols=["SK_ID_CURR", "MONTHS_BALANCE"])
    series = df["MONTHS_BALANCE"]
    positive = series[series > 0]
    return [{
        "column": "MONTHS_BALANCE",
        "total_rows": len(df),
        "suspect_rows": len(positive),
        "pct": _pct(len(positive), len(df)),
        "sample_values": sorted(positive.unique().tolist())[:10] if len(positive) > 0 else [],
    }]


# ---------------------------------------------------------------------------
# Geração do relatório
# ---------------------------------------------------------------------------

def _recommendation(results: list[dict]) -> str:
    total_suspect = sum(r["suspect_rows"] for r in results)
    if total_suspect == 0:
        return "Nenhum leakage detectado. Sem ação necessária."
    max_pct = max(float(r["pct"].rstrip("%")) for r in results if r["pct"] != "N/A")
    if max_pct < 1.0:
        return (
            "Registros suspeitos < 1% do total. "
            "Recomendar remoção preventiva ou mascarar com NaN antes do treino."
        )
    return (
        f"Registros suspeitos em até {max_pct:.1f}% dos dados. "
        "Investigar se representam datas futuras reais ou erros de codificação. "
        "Recomendar remoção dos registros com valores > 0 antes do treino."
    )


def build_report(all_checks: dict[str, list[dict]]) -> str:
    lines = [
        "# Relatório de Verificação de Data Leakage",
        "",
        "**Convenção:** colunas `DAYS_*` e `MONTHS_BALANCE` devem ser ≤ 0 (passado).",
        "Valores > 0 indicam eventos futuros à data de aplicação — potencial leakage.",
        "",
        "---",
        "",
    ]

    grand_total_suspect = 0

    for table_name, results in all_checks.items():
        total_suspect = sum(r["suspect_rows"] for r in results)
        grand_total_suspect += total_suspect

        lines.append(f"## {table_name}")
        lines.append("")
        lines.append("| Coluna | Total Linhas | Registros Suspeitos | % Afetado | Amostra de Valores |")
        lines.append("|--------|-------------|---------------------|-----------|-------------------|")

        for r in results:
            sample = ", ".join(str(v) for v in r["sample_values"]) if r["sample_values"] else "—"
            lines.append(
                f"| `{r['column']}` | {r['total_rows']:,} | {r['suspect_rows']:,} "
                f"| {r['pct']} | {sample} |"
            )

        lines.append("")
        lines.append(f"**Subtotal suspeitos:** {total_suspect:,} registros")
        lines.append("")
        lines.append(f"**Recomendação:** {_recommendation(results)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Conclusão geral
    lines.append("## Conclusão Geral")
    lines.append("")
    if grand_total_suspect == 0:
        lines.append(
            "Nenhum vazamento temporal detectado em nenhuma das tabelas verificadas. "
            "O overfitting observado nos modelos da Fase 4 é provavelmente causado por "
            "**complexidade do modelo** e pode ser tratado com regularização na Fase 5 "
            "(ajuste de hiperparâmetros, early stopping mais agressivo, dropout)."
        )
    else:
        lines.append(
            f"Total de **{grand_total_suspect:,} registros suspeitos** encontrados. "
            "Aplicar filtragem antes de re-treinar os modelos."
        )
    lines.append("")
    lines.append("---")
    lines.append("*Gerado por `src/features/leakage_check.py`*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Verificação de Data Leakage ===\n")

    tables = {
        "bureau.csv": (RAW / "bureau.csv", check_bureau),
        "installments_payments.csv": (RAW / "installments_payments.csv", check_installments),
        "previous_application.csv": (RAW / "previous_application.csv", check_previous_application),
        "POS_CASH_balance.csv": (RAW / "POS_CASH_balance.csv", check_pos_cash),
        "credit_card_balance.csv": (RAW / "credit_card_balance.csv", check_credit_card),
    }

    all_checks: dict[str, list[dict]] = {}

    for name, (path, fn) in tables.items():
        if not path.exists():
            print(f"  [AVISO] {name} não encontrado em {path} — pulando.")
            continue
        print(f"[{name}]")
        all_checks[name] = fn(path)
        total_suspect = sum(r["suspect_rows"] for r in all_checks[name])
        print(f"  -> {total_suspect:,} registros suspeitos\n")

    report_md = build_report(all_checks)
    report_path = REPORTS / "leakage_check_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"Relatório salvo em: {report_path}")
    print("\n=== Resumo ===")
    for table_name, results in all_checks.items():
        total = sum(r["suspect_rows"] for r in results)
        print(f"  {table_name}: {total:,} suspeitos")


if __name__ == "__main__":
    main()
