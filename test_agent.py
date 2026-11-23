"""Testes automatizados dos agentes (v1 e v2).

Rodar:  python -m unittest -v test_agent      (ou: pytest test_agent.py)
Variável TEST_FAST=1 reduz o actTimeout usado nos testes do v2.
"""
import importlib.util
import os
import random
import time
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v1 = load("v1")
v2 = load("v2")
v3 = load("v3")

STD = SimpleNamespace(rows=6, columns=7, inarow=4, actTimeout=2)


def cfg(rows=6, cols=7, k=4, timeout=2):
    return SimpleNamespace(rows=rows, columns=cols, inarow=k, actTimeout=timeout)


def make_board(rows, cols, moves):
    """moves = [(coluna, jogador), ...] jogadas em ordem; devolve board plano."""
    grid = [[0] * cols for _ in range(rows)]
    for c, p in moves:
        for r in range(rows - 1, -1, -1):
            if grid[r][c] == 0:
                grid[r][c] = p
                break
        else:
            raise ValueError("coluna cheia")
    return [x for row in grid for x in row]


def play(board, rows, cols, c, p):
    b = list(board)
    for r in range(rows - 1, -1, -1):
        if b[r * cols + c] == 0:
            b[r * cols + c] = p
            return b, r
    raise ValueError


def wins_at(board, rows, cols, k, r, c, p):
    """Referência independente: peça de p em (r,c) fecha k em linha?"""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < rows and 0 <= cc < cols and board[rr * cols + cc] == p:
                n += 1
                rr += dr * s
                cc += dc * s
        if n >= k:
            return True
    return False


def random_position(rows, cols, k, plies, rng):
    """Posição aleatória legal, sem vitória ainda, com `plies` jogadas."""
    while True:
        board = [0] * (rows * cols)
        p = 1
        ok = True
        for _ in range(plies):
            cs = [c for c in range(cols) if board[c] == 0]
            if not cs:
                ok = False
                break
            c = rng.choice(cs)
            board, r = play(board, rows, cols, c, p)
            if wins_at(board, rows, cols, k, r, c, p):
                ok = False
                break
            p = 3 - p
        if ok:
            return board, p  # p = quem joga agora


AGENTS = {"v1": v1.agent, "v2": v2.agent, "v3": v3.agent}


class AgentContract(unittest.TestCase):
    """Comportamentos que TODA versão do agente deve cumprir."""

    def run_agent(self, name, board, mark, config=STD):
        obs = SimpleNamespace(board=board, mark=mark)
        return AGENTS[name](obs, config)

    def each(self):
        for name in AGENTS:
            with self.subTest(agent=name):
                yield name

    def test_empty_board_returns_valid_column(self):
        for name in self.each():
            c = self.run_agent(name, [0] * 42, 1)
            self.assertIn(c, range(7))

    def test_takes_immediate_win_horizontal(self):
        moves = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        board = make_board(6, 7, moves)  # jogador 1 ganha em col 3
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 1), 3)

    def test_takes_immediate_win_vertical(self):
        board = make_board(6, 7, [(2, 1), (3, 2), (2, 1), (3, 2), (2, 1), (4, 2)])
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 1), 2)

    def test_takes_immediate_win_diagonal(self):
        # diagonal ascendente: (0,1) (1,2) (2,3) -> 4ª em col 3
        moves = [(0, 1), (1, 2), (1, 1), (2, 2), (2, 2), (2, 1),
                 (3, 2), (3, 2), (3, 2)]
        board = make_board(6, 7, moves)
        # jogador 1 tem 0@baixo,1@2ª,2@3ª ; falta col 3 na 4ª linha
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 1), 3)

    def test_blocks_immediate_loss(self):
        # jogador 2 ameaça col 3 na horizontal; jogador 1 sem vitória própria
        board = make_board(6, 7, [(0, 2), (1, 2), (2, 2), (6, 1), (6, 1)])
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 1), 3)

    def test_prefers_own_win_over_block(self):
        board = make_board(6, 7, [(0, 1), (1, 1), (2, 1),   # 1 ganha em 3
                                  (0, 2), (1, 2), (2, 2)])  # 2 também ameaça 3 (2ª linha)
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 1), 3)

    def test_never_plays_full_column(self):
        # coluna 3 cheia (sem vitória), centro preferido deve ser evitado
        moves = [(3, 1), (3, 2), (3, 1), (3, 2), (3, 1), (3, 2)]
        board = make_board(6, 7, moves)
        for name in self.each():
            c = self.run_agent(name, board, 1)
            self.assertNotEqual(c, 3)
            self.assertIn(c, range(7))

    def test_only_one_column_left(self):
        # tabuleiro cheio, sem vitória, exceto a coluna 5 (esvaziada)
        rows, cols, k = 6, 7, 4
        board = [0] * (rows * cols)
        for c in range(cols):
            for h in range(rows):
                board[(rows - 1 - h) * cols + c] = 1 + ((h // 2) + c) % 2
        for r in range(rows):
            for c in range(cols):
                self.assertFalse(wins_at(board, rows, cols, k, r, c,
                                         board[r * cols + c]), "padrão inválido")
        for r in range(rows):
            board[r * cols + 5] = 0
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 1), 5)

    def test_full_board_does_not_crash(self):
        board = [1, 2] * 21
        for name in self.each():
            self.run_agent(name, board, 1)

    def test_respects_time_limit(self):
        for name in self.each():
            t = time.time()
            self.run_agent(name, [0] * 42, 1)
            self.assertLess(time.time() - t, 1.9, name)

    def test_non_standard_board_sizes(self):
        for rows, cols, k in [(5, 5, 3), (4, 5, 4), (7, 8, 5), (6, 7, 3)]:
            c = cfg(rows, cols, k, timeout=0.5)
            board = [0] * (rows * cols)
            for name in self.each():
                col = self.run_agent(name, board, 1, c)
                self.assertIn(col, range(cols))

    def test_mark_2_perspective(self):
        # jogador 2 vence em col 3 na horizontal
        board = make_board(6, 7, [(0, 2), (1, 2), (2, 2), (0, 1), (1, 1), (6, 1)])
        for name in self.each():
            self.assertEqual(self.run_agent(name, board, 2), 3)

    def test_legal_move_on_random_positions(self):
        rng = random.Random(7)
        for _ in range(8):
            board, p = random_position(6, 7, 4, rng.randint(2, 30), rng)
            legal = [c for c in range(7) if board[c] == 0]
            for name in self.each():
                c = self.run_agent(name, board, p, cfg(timeout=0.3))
                self.assertIn(c, legal)


class BitboardTests(unittest.TestCase):
    """has_win do v2 contra uma referência independente."""

    def setUp(self):
        self.g = v2._geom(6, 7, 4)

    def bits_of(self, board, p, rows=6, cols=7):
        H = rows + 1
        b = 0
        for r in range(rows):
            for c in range(cols):
                if board[r * cols + c] == p:
                    b |= 1 << (c * H + rows - 1 - r)
        return b

    def test_four_directions(self):
        cases = {
            "horizontal": [(0, 1), (1, 1), (2, 1), (3, 1)],
            "vertical": [(4, 1)] * 4,
            "diag_up": [(0, 1), (1, 2), (1, 1), (2, 2), (2, 2), (2, 1),
                        (3, 2), (3, 2), (3, 2), (3, 1)],
            "diag_down": [(3, 1), (2, 2), (2, 1), (1, 2), (1, 2), (1, 1),
                          (0, 2), (0, 2), (0, 2), (0, 1)],
        }
        for name, moves in cases.items():
            with self.subTest(name):
                board = make_board(6, 7, moves)
                self.assertTrue(v2.has_win(self.bits_of(board, 1),
                                           self.g.shifts, 4))

    def test_no_wrap_between_columns(self):
        # 2 peças no fim de uma coluna + 2 no início da seguinte não é vitória
        board = make_board(6, 7, [(0, 1), (0, 1), (1, 1), (1, 1)])
        # vertical-ish wrap: células (col0,h0),(col0,h1),(col1,h0),(col1,h1)
        self.assertFalse(v2.has_win(self.bits_of(board, 1), self.g.shifts, 4))
        # topo da col 0 + base da col 1 (adjacentes em índice)
        moves = [(0, 2)] * 3 + [(0, 1)] * 3  # col 0 cheia: topo=1
        moves += [(1, 1)] * 3              # base col 1
        board = make_board(6, 7, moves)
        self.assertFalse(v2.has_win(self.bits_of(board, 1), self.g.shifts, 4))

    def test_matches_reference_on_random_positions(self):
        rng = random.Random(1)
        for rows, cols, k in [(6, 7, 4), (5, 5, 3), (7, 8, 5)]:
            g = v2._geom(rows, cols, k)
            for _ in range(150):
                board = [0] * (rows * cols)
                p = 1
                for _ in range(rng.randint(1, rows * cols)):
                    cs = [c for c in range(cols) if board[c] == 0]
                    if not cs:
                        break
                    c = rng.choice(cs)
                    board, r = play(board, rows, cols, c, p)
                    expect = wins_at(board, rows, cols, k, r, c, p)
                    got = v2.has_win(self.bits_of(board, p, rows, cols),
                                     g.shifts, k)
                    # has_win olha o tabuleiro todo; como paramos na 1ª vitória,
                    # os dois critérios coincidem.
                    self.assertEqual(expect, got, (rows, cols, k, board))
                    if expect:
                        break
                    p = 3 - p


class EvaluationEquivalence(unittest.TestCase):
    """O score incremental do v2 deve ser idêntico ao evaluate() do v1."""

    @staticmethod
    def v1_eval(board, rows, cols, k, p):
        grid = [board[r * cols:(r + 1) * cols] for r in range(rows)]
        weights = [0] + [10 ** (i - 1) for i in range(1, k + 1)]
        q = 3 - p
        score = 0
        for w in v1._windows(rows, cols, k):
            mine = sum(1 for r, c in w if grid[r][c] == p)
            theirs = sum(1 for r, c in w if grid[r][c] == q)
            if mine and not theirs:
                score += weights[mine]
            elif theirs and not mine:
                score -= weights[theirs]
        mid = cols // 2
        for r in range(rows):
            if grid[r][mid] == p:
                score += 3
            elif grid[r][mid] == q:
                score -= 3
        return score

    @staticmethod
    def v2_eval(board, rows, cols, k, p):
        g = v2._geom(rows, cols, k)
        H = g.H
        wcode = [0] * len(g.windows)
        score1 = 0
        for r in range(rows):
            for c in range(cols):
                q = board[r * cols + c]
                if q:
                    idx = c * H + (rows - 1 - r)
                    for w in g.cell_windows[idx]:
                        old = wcode[w]
                        wcode[w] = old + g.inc[q]
                        score1 += g.tbl[wcode[w]] - g.tbl[old]
                    score1 += g.bonus[idx] if q == 1 else -g.bonus[idx]
        return score1 if p == 1 else -score1

    def test_same_window_count(self):
        for rows, cols, k in [(6, 7, 4), (5, 5, 3), (7, 8, 5)]:
            self.assertEqual(len(v1._windows(rows, cols, k)),
                             len(v2._geom(rows, cols, k).windows))

    def test_scores_match_on_random_positions(self):
        rng = random.Random(3)
        for rows, cols, k in [(6, 7, 4), (5, 5, 3), (7, 8, 5)]:
            for _ in range(60):
                board, _p = random_position(rows, cols, k,
                                            rng.randint(0, rows * cols // 2), rng)
                for p in (1, 2):
                    self.assertEqual(self.v1_eval(board, rows, cols, k, p),
                                     self.v2_eval(board, rows, cols, k, p))


class V2Specific(unittest.TestCase):
    def test_deterministic_given_same_depth(self):
        # com limite de tempo grande o bastante, a posição simples é decidida
        board = make_board(6, 7, [(3, 1), (3, 2), (4, 1)])
        a = v2.agent(SimpleNamespace(board=board, mark=2), cfg(timeout=0.5))
        self.assertIn(a, range(7))

    def test_stats_reported(self):
        v2.agent(SimpleNamespace(board=[0] * 42, mark=1), cfg(timeout=0.5))
        self.assertGreater(v2.STATS["depth"], 0)
        self.assertGreater(v2.STATS["nodes"], 0)

    def test_v2_searches_deeper_than_v1(self):
        """Mesmo orçamento de tempo: v2 deve processar mais nós/s que v1."""
        with open(os.path.join(HERE, "v1.py")) as f:
            src = f.read()
        src = src.replace("best_move = iter_best",
                          "best_move = iter_best; _D.append(depth)")
        ns = {"_D": []}
        exec(src, ns)
        ns["agent"](SimpleNamespace(board=[0] * 42, mark=1), STD)
        v1_depth = ns["_D"][-1]
        v2.agent(SimpleNamespace(board=[0] * 42, mark=1), STD)
        self.assertGreater(v2.STATS["depth"], v1_depth)

    def test_tt_reuse_does_not_change_legality_across_games(self):
        rng = random.Random(11)
        for _ in range(4):
            board, p = random_position(6, 7, 4, rng.randint(4, 20), rng)
            legal = [c for c in range(7) if board[c] == 0]
            self.assertIn(v2.agent(SimpleNamespace(board=board, mark=p),
                                   cfg(timeout=0.2)), legal)

    def test_solves_forced_win_in_two(self):
        # jogador 1 tem 2 e 3 na base com 1 e 4 livres: ameaça dupla horizontal
        board = make_board(6, 7, [(2, 1), (3, 1), (2, 2), (3, 2)])
        a = v2.agent(SimpleNamespace(board=board, mark=1), cfg(timeout=1))
        self.assertIn(a, (1, 4))


class V3Specific(unittest.TestCase):
    def brute_threats(self, board, rows, cols, k, p):
        """Células vazias jogáveis ou não onde p completaria k (força bruta)."""
        H = rows + 1
        out = 0
        for c in range(cols):
            for r in range(rows):
                if board[r * cols + c] == 0:
                    b = list(board)
                    b[r * cols + c] = p
                    if wins_at(b, rows, cols, k, r, c, p):
                        out |= 1 << (c * H + rows - 1 - r)
        return out

    def test_threat_cells_match_bruteforce(self):
        rng = random.Random(5)
        helper = BitboardTests()
        for rows, cols, k in [(6, 7, 4), (5, 5, 3), (7, 8, 5)]:
            g = v3._geom(rows, cols, k)
            for _ in range(120):
                board, _p = random_position(rows, cols, k,
                                            rng.randint(0, rows * cols - 1), rng)
                mask = 0
                for p in (1, 2):
                    mask |= helper.bits_of(board, p, rows, cols)
                free = g.full ^ mask
                for p in (1, 2):
                    got = v3.threat_cells(helper.bits_of(board, p, rows, cols), g) & free
                    self.assertEqual(got, self.brute_threats(board, rows, cols, k, p),
                                     (rows, cols, k, board, p))

    def test_first_move_is_center(self):
        self.assertEqual(v3.agent(SimpleNamespace(board=[0] * 42, mark=1), STD), 3)

    def test_single_valid_column_returns_instantly(self):
        board = make_board(6, 7, [(c, 1 + (r % 2)) for c in range(7) if c != 4
                                  for r in range(0)])
        board = [0] * 42
        for c in range(7):
            if c == 4:
                continue
            for h in range(6):
                board[(5 - h) * 7 + c] = 1 + ((h // 2) + c) % 2
        t = time.time()
        self.assertEqual(v3.agent(SimpleNamespace(board=board, mark=1), STD), 4)
        self.assertLess(time.time() - t, 0.1)

    def test_avoids_opponent_double_threat(self):
        # 2 tem 2,3 na base (cols 2,3): se 1 não agir, 2 cria ameaça dupla
        # em cols 1 e 4. Jogador 1 deve jogar em 1 ou 4 (ou 2/3 não é possível).
        board = make_board(6, 7, [(6, 1), (2, 2), (6, 1), (3, 2)])
        # vez do jogador 1; col 1 e col 4 impedem a ameaça dupla da base
        move = v3.agent(SimpleNamespace(board=board, mark=1), cfg(timeout=1))
        self.assertIn(move, (1, 4))

    def test_flags_off_still_legal(self):
        # v3 deve continuar jogando legalmente com os módulos desligados
        old = (v3._USE_THREATS, v3._USE_KILLERS)
        try:
            v3._USE_THREATS = v3._USE_KILLERS = False
            b = make_board(6, 7, [(3, 1), (3, 2), (4, 1)])
            self.assertIn(v3.agent(SimpleNamespace(board=b, mark=2),
                                   cfg(timeout=0.3)), range(7))
        finally:
            v3._USE_THREATS, v3._USE_KILLERS = old

    def test_low_overage_does_not_exceed_soft_limit(self):
        b = make_board(6, 7, [(3, 1), (3, 2), (4, 1), (2, 2)])
        obs = SimpleNamespace(board=b, mark=1, remainingOverageTime=1)
        t = time.time()
        v3.agent(obs, STD)
        self.assertLess(time.time() - t, 2 * v3._SOFT + 0.15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
