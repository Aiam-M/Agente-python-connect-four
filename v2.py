"""Agente ConnectX v2.

Mudanças em relação ao v1 (mesma heurística, motor de busca mais rápido):
  * Bitboards (um inteiro por jogador) no lugar de lista de listas:
    teste de vitória e geração de jogadas viram operações de bits.
  * Avaliação incremental: cada janela guarda um código (peças do jogador 1,
    peças do jogador 2) e o score é atualizado só nas janelas que contêm a
    célula jogada, em vez de reescanear todas as janelas a cada folha.
  * Tabela de transposição (valor, profundidade, tipo de limite, melhor
    jogada), mantida entre iterações e entre jogadas da mesma partida.
  * Ordenação: melhor jogada da tabela / iteração anterior primeiro, depois
    centro para as bordas.
  * Iteração parcial aproveitada no timeout.
  * Limite de tempo derivado de configuration.actTimeout.

Layout do bitboard: coluna c ocupa os bits [c*H, c*H + rows), com H = rows + 1;
o bit extra do topo de cada coluna nunca é usado (sentinela contra wrap).
"""
import time

_WIN_SCORE = 1_000_000
_INF = float("inf")
_EXACT, _LOWER, _UPPER = 0, 1, 2
_TT_MAX = 2_000_000

_GEOM_CACHE = {}
_TT = {}
_TT_KEY = None

# Estatísticas da última jogada (úteis para benchmark/depuração).
STATS = {"depth": 0, "nodes": 0}


class _Timeout(Exception):
    pass


class _Geom:
    """Pré-cálculos que dependem só de (rows, cols, k)."""

    def __init__(self, rows, cols, k):
        self.rows, self.cols, self.k = rows, cols, k
        H = rows + 1
        self.H = H
        self.shifts = (1, H, H - 1, H + 1)

        def cell(r, c):  # r = linha do tabuleiro (0 = topo)
            return c * H + (rows - 1 - r)

        # Janelas de k células, na mesma ordem lógica do v1.
        windows = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    er, ec = r + dr * (k - 1), c + dc * (k - 1)
                    if 0 <= er < rows and 0 <= ec < cols:
                        windows.append(
                            tuple(cell(r + dr * i, c + dc * i) for i in range(k)))
        self.windows = windows
        n_cells = cols * H
        self.cell_windows = [[] for _ in range(n_cells)]
        for wi, w in enumerate(windows):
            for idx in w:
                self.cell_windows[idx].append(wi)

        # Tabela de score (perspectiva do jogador 1) por código de janela.
        # código = c1 * (k + 1) + c2.
        weights = [0] + [10 ** (i - 1) for i in range(1, k + 1)]
        base = k + 1
        tbl = [0] * (base * base)
        for c1 in range(base):
            for c2 in range(base):
                if c1 and not c2:
                    tbl[c1 * base + c2] = weights[c1]
                elif c2 and not c1:
                    tbl[c1 * base + c2] = -weights[c2]
        self.tbl = tbl
        self.inc = (0, base, 1)  # incremento do código por jogador (1 ou 2)

        # Bônus da coluna central (mesmo do v1): +3 jogador 1, -3 jogador 2.
        mid = cols // 2
        self.bonus = [0] * n_cells
        for h in range(rows):
            self.bonus[mid * H + h] = 3

        # Ordem das colunas: centro primeiro.
        center = (cols - 1) / 2
        self.order = sorted(range(cols), key=lambda c: abs(c - center))
        self.col_base = [c * H for c in range(cols)]
        self.col_top = [c * H + rows for c in range(cols)]  # 1º bit fora da coluna


def _geom(rows, cols, k):
    key = (rows, cols, k)
    g = _GEOM_CACHE.get(key)
    if g is None:
        g = _GEOM_CACHE[key] = _Geom(rows, cols, k)
    return g


def has_win(b, shifts, k):
    """True se o bitboard b contém k peças em linha."""
    if k == 4:
        for s in shifts:
            m = b & (b >> s)
            if m & (m >> (2 * s)):
                return True
        return False
    for s in shifts:
        m = b
        for i in range(1, k):
            m &= b >> (s * i)
            if not m:
                break
        if m:
            return True
    return False


def agent(observation, configuration):
    global _TT, _TT_KEY
    start = time.time()
    act_timeout = getattr(configuration, "actTimeout", 2) or 2
    time_limit = act_timeout * 0.6  # mesma margem do v1 (1,2 s para 2 s)

    rows = configuration.rows
    cols = configuration.columns
    k = configuration.inarow
    me = observation.mark
    board = observation.board

    g = _geom(rows, cols, k)
    H, shifts, order = g.H, g.shifts, g.order
    col_top, col_base = g.col_top, g.col_base
    cell_windows, tbl, inc, bonus = g.cell_windows, g.tbl, g.inc, g.bonus
    total_cells = rows * cols

    if _TT_KEY != (rows, cols, k) or len(_TT) > _TT_MAX:
        _TT = {}
        _TT_KEY = (rows, cols, k)
    tt = _TT

    # ---- estado inicial a partir da observação ----
    bits = [0, 0, 0]          # bits[1], bits[2]
    height = list(col_base)   # próximo bit livre de cada coluna
    wcode = [0] * len(g.windows)
    score1 = 0                # score da perspectiva do jogador 1
    plies = 0
    for r in range(rows):
        for c in range(cols):
            p = board[r * cols + c]
            if p:
                idx = c * H + (rows - 1 - r)
                bits[p] |= 1 << idx
                plies += 1
                for w in cell_windows[idx]:
                    old = wcode[w]
                    wcode[w] = old + inc[p]
                    score1 += tbl[wcode[w]] - tbl[old]
                score1 += bonus[idx] if p == 1 else -bonus[idx]
    for c in range(cols):
        h = col_base[c]
        while h < col_top[c] and (bits[1] | bits[2]) >> h & 1:
            h += 1
        height[c] = h
    mask = bits[1] | bits[2]

    valid = [c for c in order if height[c] < col_top[c]]
    if not valid:
        return 0
    opp = 3 - me

    # ---- vitória imediata e bloqueio ----
    for c in valid:
        if has_win(bits[me] | (1 << height[c]), shifts, k):
            return c
    for c in valid:
        if has_win(bits[opp] | (1 << height[c]), shifts, k):
            return c

    counter = [0]

    def negamax(p, depth, alpha, beta, cur, oth, mask, score1, plies):
        """cur = bits de p (jogador a mover), oth = bits do adversário."""
        counter[0] += 1
        if counter[0] & 255 == 0 and time.time() - start > time_limit:
            raise _Timeout()
        if plies >= total_cells:
            return 0
        if depth == 0:
            return score1 if p == 1 else -score1

        key = cur + mask
        entry = tt.get(key)
        tt_move = -1
        if entry is not None:
            d, flag, val, tt_move = entry
            if d >= depth:
                if flag == _EXACT:
                    return val
                if flag == _LOWER:
                    if val > alpha:
                        alpha = val
                elif val < beta:
                    beta = val
                if alpha >= beta:
                    return val

        q = 3 - p
        ip = inc[p]
        sgn = 1 if p == 1 else -1
        alpha0 = alpha
        best = -_INF
        best_move = -1

        if tt_move >= 0:
            moves = [tt_move] + [c for c in order if c != tt_move]
        else:
            moves = order

        for c in moves:
            h = height[c]
            if h >= col_top[c]:
                continue
            new_cur = cur | (1 << h)
            if has_win(new_cur, shifts, k):
                val = _WIN_SCORE + depth
            else:
                s = score1
                ws = cell_windows[h]
                for w in ws:
                    old = wcode[w]
                    new = old + ip
                    wcode[w] = new
                    s += tbl[new] - tbl[old]
                s += sgn * bonus[h]
                height[c] = h + 1
                try:
                    val = -negamax(q, depth - 1, -beta, -alpha,
                                   oth, new_cur, mask | (1 << h), s, plies + 1)
                finally:
                    height[c] = h
                    for w in ws:
                        wcode[w] -= ip
            if val > best:
                best = val
                best_move = c
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best <= alpha0:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        tt[key] = (depth, flag, best, best_move)
        return best

    # ---- aprofundamento iterativo na raiz ----
    best_move = valid[0]
    cur_root, oth_root = bits[me], bits[opp]
    ip_me = inc[me]
    sgn_me = 1 if me == 1 else -1
    max_depth = total_cells - plies
    done_depth = 0

    try:
        for depth in range(1, max_depth + 1):
            moves = [best_move] + [c for c in valid if c != best_move]
            iter_best, iter_score = None, -_INF
            alpha = -_INF
            for c in moves:
                h = height[c]
                new_cur = cur_root | (1 << h)
                if has_win(new_cur, shifts, k):
                    val = _WIN_SCORE + depth
                else:
                    s = score1
                    ws = cell_windows[h]
                    for w in ws:
                        old = wcode[w]
                        new = old + ip_me
                        wcode[w] = new
                        s += tbl[new] - tbl[old]
                    s += sgn_me * bonus[h]
                    height[c] = h + 1
                    try:
                        val = -negamax(opp, depth - 1, -_INF, -alpha,
                                       oth_root, new_cur, mask | (1 << h),
                                       s, plies + 1)
                    except _Timeout:
                        # Estado local é descartado; mas se já há resultado
                        # de pelo menos um lance nesta iteração, usa-o.
                        if iter_best is not None:
                            best_move = iter_best
                        raise
                    finally:
                        height[c] = h
                        for w in ws:
                            wcode[w] -= ip_me
                if val > iter_score:
                    iter_score, iter_best = val, c
                if iter_score > alpha:
                    alpha = iter_score
            best_move = iter_best
            done_depth = depth
            if iter_score >= _WIN_SCORE:  # vitória forçada
                break
            if iter_score <= -_WIN_SCORE:  # derrota forçada: mais profundo não ajuda
                break
    except _Timeout:
        pass

    STATS["depth"] = done_depth
    STATS["nodes"] = counter[0]
    return int(best_move)


def act(observation, configuration):
    return agent(observation, configuration)
