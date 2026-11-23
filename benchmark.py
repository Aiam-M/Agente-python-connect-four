"""Benchmark de agentes ConnectX (duelo A x B), sem depender do kaggle_environments.

Uso:
  python benchmark.py v2.py v1.py --pairs 20
  python benchmark.py v2.py random --pairs 20

Cada "par" usa uma abertura aleatória (--opening-plies jogadas) jogada duas
vezes, trocando quem começa, para eliminar viés de cor e dar variedade
(os agentes são determinísticos). Partidas rodam em paralelo (--workers).
Reporta vitórias/empates/derrotas de A, tempo por jogada e profundidade média.
"""
import argparse
import importlib.util
import os
import random
import time
from multiprocessing import Pool
from types import SimpleNamespace

ROWS, COLS, K = 6, 7, 4
HERE = os.path.dirname(os.path.abspath(__file__))
_cache = {}


def load_agent(spec):
    """Devolve (função, módulo|None) para um caminho .py ou 'random'."""
    if spec in _cache:
        return _cache[spec]
    if spec == "random":
        def rnd(obs, conf):
            return random.choice([c for c in range(COLS) if obs.board[c] == 0])
        res = (rnd, None)
    else:
        path = spec if os.path.isabs(spec) else os.path.join(HERE, spec)
        s = importlib.util.spec_from_file_location("ag_" + str(len(_cache)), path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        res = (mod.agent, mod)
    _cache[spec] = res
    return res


def wins(board, r, c, p):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr * COLS + cc] == p:
                n += 1
                rr += dr * s
                cc += dc * s
        if n >= K:
            return True
    return False


def drop(board, c, p):
    for r in range(ROWS - 1, -1, -1):
        if board[r * COLS + c] == 0:
            board[r * COLS + c] = p
            return r
    return None


def play_game(args):
    """Joga uma partida. first = 'A' ou 'B' (quem é o jogador 1)."""
    spec_a, spec_b, opening, first, act_timeout = args
    agents = {"A": load_agent(spec_a), "B": load_agent(spec_b)}
    mark_of = {first: 1, ("B" if first == "A" else "A"): 2}
    who = {1: first, 2: ("B" if first == "A" else "A")}
    conf = SimpleNamespace(rows=ROWS, columns=COLS, inarow=K, actTimeout=act_timeout)

    board = [0] * (ROWS * COLS)
    p = 1
    for c in opening:  # abertura forçada, sem vitória possível (poucas jogadas)
        drop(board, c, p)
        p = 3 - p

    times = {"A": [], "B": []}
    depths = {"A": [], "B": []}
    result = None  # 'A', 'B' ou 'D'
    while result is None:
        side = who[p]
        fn, mod = agents[side]
        obs = SimpleNamespace(board=list(board), mark=p)
        t = time.time()
        try:
            c = int(fn(obs, conf))
        except Exception:
            c = -1
        dt = time.time() - t
        times[side].append(dt)
        if mod is not None and hasattr(mod, "STATS"):
            depths[side].append(mod.STATS["depth"])
        if not (0 <= c < COLS) or board[c] != 0 or dt > act_timeout:
            result = "B" if side == "A" else "A"  # jogada inválida/timeout = derrota
            break
        r = drop(board, c, p)
        if wins(board, r, c, p):
            result = side
        elif all(board[i] != 0 for i in range(COLS)):
            result = "D"
        p = 3 - p
    return result, times, depths, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pairs", type=int, default=10,
                    help="nº de aberturas; cada uma gera 2 partidas")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--act-timeout", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    jobs = []
    for _ in range(a.pairs):
        while True:  # abertura legal e sem vitória
            b, p, opening, ok = [0] * (ROWS * COLS), 1, [], True
            for _ in range(a.opening_plies):
                c = rng.choice([c for c in range(COLS) if b[c] == 0])
                r = drop(b, c, p)
                opening.append(c)
                if wins(b, r, c, p):
                    ok = False
                    break
                p = 3 - p
            if ok:
                break
        jobs += [(a.a, a.b, opening, "A", a.act_timeout),
                 (a.a, a.b, opening, "B", a.act_timeout)]

    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(play_game, jobs, chunksize=1)

    cnt = {"A": 0, "B": 0, "D": 0}
    by_first = {"A": [0, 0, 0], "B": [0, 0, 0]}  # A: vit/emp/der quando A é J1 / J2
    tm = {"A": [], "B": []}
    dp = {"A": [], "B": []}
    for result, times, depths, first in res:
        cnt[result] += 1
        i = 0 if result == "A" else (1 if result == "D" else 2)
        by_first[first][i] += 1
        for s in "AB":
            tm[s] += times[s]
            dp[s] += depths[s]

    n = len(res)
    print(f"{a.a} (A)  vs  {a.b} (B)   -   {n} partidas em {time.time()-t0:.0f}s")
    print(f"  A: {cnt['A']} vitórias | {cnt['D']} empates | {cnt['B']} derrotas "
          f"-> score {(cnt['A'] + 0.5*cnt['D'])/n:.1%}")
    print(f"  A como 1º jogador (V/E/D): {tuple(by_first['A'])}   "
          f"A como 2º (V/E/D): {tuple(by_first['B'])}")
    for s, name in (("A", a.a), ("B", a.b)):
        if tm[s]:
            line = (f"  {name}: tempo/jogada médio {sum(tm[s])/len(tm[s]):.2f}s  "
                    f"máx {max(tm[s]):.2f}s")
            if dp[s]:
                line += f"  | profundidade média {sum(dp[s])/len(dp[s]):.1f}"
            print(line)


if __name__ == "__main__":
    main()
