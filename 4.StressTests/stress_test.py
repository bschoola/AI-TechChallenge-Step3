"""
Stress Test — Breast Cancer Prediction API

Suporta dois modos:

  MODO LAUDO (padrão) — fila assíncrona via Celery + Redis
    1. Dispara N POSTs concorrentes → /predict/breastCancer/laudo
       Cada requisição retorna imediatamente com task_id
    2. Faz polling concorrente → /predict/breastCancer/laudo/status/{task_id}
       Até status "completed" ou "failed"
    Mede: latência de enqueue (POST) e latência total (enqueue → laudo pronto)

  MODO PREDICT — endpoint síncrono do classificador (sem LLM)
    Dispara N POSTs concorrentes → /predict/breastCancer
    Mede: latência direta de cada requisição

Uso:
    python stress_test.py                          # 10 req simultâneas, laudo
    python stress_test.py --n 20                   # 20 req simultâneas
    python stress_test.py --waves 3                # 3 ondas de N req cada
    python stress_test.py --endpoint predict       # só classificador, sem LLM
    python stress_test.py --poll-interval 1.0      # polling a cada 1s (padrão: 2s)
    python stress_test.py --url http://host:8000   # URL diferente
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

# --------------------------------------------------------------------------- #
# Payloads de exemplo
# --------------------------------------------------------------------------- #

PAYLOADS = [
    {"area_pior": 880.5,  "textura_pior": 25.38, "pontos_concavos_pior": 0.2654, "concavidade_pior": 0.3001},
    {"area_pior": 1.5,    "textura_pior": 14.22, "pontos_concavos_pior": 0.0000, "concavidade_pior": 0.0000},
    {"area_pior": 1200.0, "textura_pior": 30.12, "pontos_concavos_pior": 0.3500, "concavidade_pior": 0.4200},
    {"area_pior": 450.0,  "textura_pior": 18.00, "pontos_concavos_pior": 0.0800, "concavidade_pior": 0.0500},
    {"area_pior": 700.0,  "textura_pior": 22.50, "pontos_concavos_pior": 0.1500, "concavidade_pior": 0.2000},
]


# --------------------------------------------------------------------------- #
# Estruturas de resultado
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    request_id: int
    status: Literal["ok", "error"]
    http_status: int | None  = None
    enqueue_s: float         = 0.0   # tempo do POST até receber task_id
    total_s: float           = 0.0   # tempo do POST até laudo pronto (ou erro)
    poll_cycles: int         = 0     # quantas vezes fez polling
    error_msg: str           = ""
    diagnostico: str         = ""
    confianca: float         = 0.0
    task_id: str             = ""


# --------------------------------------------------------------------------- #
# Worker — modo laudo assíncrono
# --------------------------------------------------------------------------- #

async def call_laudo_async(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict,
    request_id: int,
    poll_interval: float,
) -> Result:
    post_url  = f"{base_url}/predict/breastCancer/laudo"
    poll_base = f"{base_url}/predict/breastCancer/laudo/status"
    t_start   = time.perf_counter()

    # Passo 1: enqueue
    try:
        resp = await client.post(post_url, json=payload, timeout=30.0)
        enqueue_s = time.perf_counter() - t_start

        if resp.status_code != 200:
            return Result(
                request_id=request_id, status="error",
                http_status=resp.status_code,
                enqueue_s=enqueue_s, total_s=enqueue_s,
                error_msg=resp.text[:200],
            )

        data      = resp.json()
        task_id   = data.get("task_id", "")
        diag      = data.get("diagnostico", "?")
        confianca = data.get("confianca", 0.0)

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        return Result(
            request_id=request_id, status="error",
            enqueue_s=elapsed, total_s=elapsed,
            error_msg=f"POST falhou: {exc!s:.150}",
        )

    # Passo 2: polling
    poll_url    = f"{poll_base}/{task_id}"
    poll_cycles = 0

    while True:
        await asyncio.sleep(poll_interval)
        poll_cycles += 1
        try:
            pr = await client.get(poll_url, timeout=10.0)
            if pr.status_code != 200:
                continue
            pdata  = pr.json()
            status = pdata.get("status", "pending")

            if status == "completed":
                total_s = time.perf_counter() - t_start
                return Result(
                    request_id=request_id, status="ok",
                    http_status=200,
                    enqueue_s=enqueue_s, total_s=total_s,
                    poll_cycles=poll_cycles,
                    diagnostico=diag, confianca=confianca,
                    task_id=task_id,
                )
            elif status == "failed":
                total_s = time.perf_counter() - t_start
                return Result(
                    request_id=request_id, status="error",
                    http_status=200,
                    enqueue_s=enqueue_s, total_s=total_s,
                    poll_cycles=poll_cycles,
                    error_msg=pdata.get("error", "task failed"),
                    task_id=task_id,
                )
            # pending ou processing → continua polling

        except Exception:
            pass  # ignora falhas de rede temporárias no poll

        # segurança: não faz polling por mais de 10 min
        if time.perf_counter() - t_start > 600:
            total_s = time.perf_counter() - t_start
            return Result(
                request_id=request_id, status="error",
                enqueue_s=enqueue_s, total_s=total_s,
                poll_cycles=poll_cycles,
                error_msg="Timeout: laudo não ficou pronto em 10 minutos",
                task_id=task_id,
            )


# --------------------------------------------------------------------------- #
# Worker — modo predict síncrono
# --------------------------------------------------------------------------- #

async def call_predict(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict,
    request_id: int,
) -> Result:
    url   = f"{base_url}/predict/breastCancer"
    start = time.perf_counter()
    try:
        resp   = await client.post(url, json=payload, timeout=30.0)
        elapsed = time.perf_counter() - start
        if resp.status_code == 200:
            data = resp.json()
            return Result(
                request_id=request_id, status="ok",
                http_status=200,
                enqueue_s=elapsed, total_s=elapsed,
                diagnostico=data.get("diagnostico", "?"),
                confianca=data.get("confianca", 0.0),
            )
        return Result(
            request_id=request_id, status="error",
            http_status=resp.status_code,
            enqueue_s=elapsed, total_s=elapsed,
            error_msg=resp.text[:200],
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return Result(
            request_id=request_id, status="error",
            enqueue_s=elapsed, total_s=elapsed,
            error_msg=str(exc)[:200],
        )


# --------------------------------------------------------------------------- #
# Execução de uma onda
# --------------------------------------------------------------------------- #

async def run_wave(
    base_url: str,
    mode: Literal["laudo", "predict"],
    n: int,
    wave_num: int,
    poll_interval: float,
) -> list[Result]:
    if mode == "laudo":
        endpoint_label = "POST /predict/breastCancer/laudo  →  polling /status/{task_id}"
    else:
        endpoint_label = "POST /predict/breastCancer"

    print(f"\n{'='*65}")
    print(f"  Onda {wave_num}  |  {n} requisições simultâneas")
    print(f"  Endpoint : {endpoint_label}")
    if mode == "laudo":
        print(f"  Polling  : a cada {poll_interval}s")
    print(f"{'='*65}")

    async with httpx.AsyncClient() as client:
        if mode == "laudo":
            tasks = [
                call_laudo_async(client, base_url, PAYLOADS[i % len(PAYLOADS)], i + 1, poll_interval)
                for i in range(n)
            ]
        else:
            tasks = [
                call_predict(client, base_url, PAYLOADS[i % len(PAYLOADS)], i + 1)
                for i in range(n)
            ]

        wave_start = time.perf_counter()
        results    = await asyncio.gather(*tasks)
        wall_time  = time.perf_counter() - wave_start

    # Imprime resultado por requisição
    header = f"  {'#':>3}  {'Status':<8}  {'Diagnóstico':<10}  {'Confiança':>9}  {'Enqueue':>7}  {'Total':>7}"
    if mode == "laudo":
        header += f"  {'Polls':>5}"
    print(header)
    print(f"  {'-'*60}")

    for r in sorted(results, key=lambda x: x.request_id):
        if r.status == "ok":
            line = (
                f"  #{r.request_id:>3}  {'✓ ok':<8}  {r.diagnostico:<10}  "
                f"{r.confianca:>8.1f}%  {r.enqueue_s:>6.2f}s  {r.total_s:>6.2f}s"
            )
            if mode == "laudo":
                line += f"  {r.poll_cycles:>5}x"
        else:
            code = r.http_status or "ERR"
            line = (
                f"  #{r.request_id:>3}  {'✗ err':<8}  HTTP {code:<6}  "
                f"{'':>9}  {r.enqueue_s:>6.2f}s  {r.total_s:>6.2f}s  "
                f"{r.error_msg[:40]}"
            )
        print(line)

    print(f"\n  Tempo de parede da onda: {wall_time:.2f}s")
    return list(results)


# --------------------------------------------------------------------------- #
# Relatório final
# --------------------------------------------------------------------------- #

def print_report(all_results: list[Result], total_wall: float, mode: str):
    ok      = [r for r in all_results if r.status == "ok"]
    errors  = [r for r in all_results if r.status == "error"]
    enq_lat = [r.enqueue_s for r in ok]
    tot_lat = [r.total_s   for r in ok]

    def stats(values: list[float]) -> str:
        if not values:
            return "  n/a"
        p95_idx = max(0, int(len(values) * 0.95) - 1)
        return (
            f"    Mín    : {min(values):.2f}s\n"
            f"    Máx    : {max(values):.2f}s\n"
            f"    Média  : {statistics.mean(values):.2f}s\n"
            f"    Mediana: {statistics.median(values):.2f}s\n"
            + (f"    StdDev : {statistics.stdev(values):.2f}s\n" if len(values) > 1 else "")
            + f"    P95    : {sorted(values)[p95_idx]:.2f}s"
        )

    print(f"\n{'='*65}")
    print("  RELATÓRIO FINAL")
    print(f"{'='*65}")
    print(f"  Total de requisições  : {len(all_results)}")
    print(f"  Sucesso               : {len(ok)}")
    print(f"  Erros                 : {len(errors)}")
    print(f"  Taxa de sucesso       : {len(ok)/len(all_results)*100:.1f}%")
    print(f"  Tempo total (parede)  : {total_wall:.2f}s")

    if mode == "laudo":
        print(f"\n  Latência de ENQUEUE (POST → task_id)")
        print(stats(enq_lat))
        print(f"\n  Latência TOTAL (POST → laudo pronto)")
        print(stats(tot_lat))

        if ok:
            avg_queue = statistics.mean(r.total_s - r.enqueue_s for r in ok)
            avg_polls = statistics.mean(r.poll_cycles for r in ok)
            print(f"\n  Tempo médio em fila   : {avg_queue:.2f}s")
            print(f"  Média de polls/req    : {avg_polls:.1f}x")
    else:
        print(f"\n  Latência por requisição")
        print(stats(tot_lat))

    if errors:
        print(f"\n  Erros encontrados:")
        for r in errors:
            code = r.http_status or "ERR"
            print(f"    [#{r.request_id}] HTTP {code} — {r.error_msg[:80]}")

    print(f"{'='*65}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main():
    parser = argparse.ArgumentParser(description="Stress test da Prediction API")
    parser.add_argument("--url",           default="http://localhost:8000", help="URL base da API")
    parser.add_argument("--n",             type=int,   default=10,  help="Requisições simultâneas por onda")
    parser.add_argument("--waves",         type=int,   default=1,   help="Número de ondas")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Intervalo de polling em segundos (modo laudo)")
    parser.add_argument(
        "--endpoint",
        choices=["laudo", "predict"],
        default="laudo",
        help="'laudo' = async com polling (padrão)  |  'predict' = síncrono sem LLM",
    )
    args = parser.parse_args()

    print(f"\n  {'─'*55}")
    print(f"  Breast Cancer API — Stress Test")
    print(f"  {'─'*55}")
    print(f"  URL           : {args.url}")
    print(f"  Modo          : {args.endpoint}")
    print(f"  Concorrência  : {args.n} requisições / onda")
    print(f"  Ondas         : {args.waves}")
    if args.endpoint == "laudo":
        print(f"  Poll interval : {args.poll_interval}s")

    all_results: list[Result] = []
    total_start = time.perf_counter()

    for wave in range(1, args.waves + 1):
        results = await run_wave(args.url, args.endpoint, args.n, wave, args.poll_interval)
        all_results.extend(results)
        if wave < args.waves:
            print("  aguardando 2s antes da próxima onda...")
            await asyncio.sleep(2)

    total_wall = time.perf_counter() - total_start
    print_report(all_results, total_wall, args.endpoint)


if __name__ == "__main__":
    asyncio.run(main())
