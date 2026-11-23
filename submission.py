import time

_WINDOW_CACHE = {}
_WIN_SCORE = 1_000_000


class _Timeout(Exception):
    pass


def _windows(rows, cols, k):
    """Todas as janelas de k células (horizontal, vertical, diagonais)."""
    key = (rows, cols, k)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]
    wins = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + dr * (k - 1), c + dc * (k - 1)
                if 0 <= er < rows and 0 <= ec < cols:
                    wins.append(tuple((r + dr * i, c + dc * i) for i in range(k)))
    _WINDOW_CACHE[key] = wins
    return wins


def _is_win(grid, r, c, p, rows, cols, k):
    """True se a peça de p em (r, c) forma k em linha."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == p:
                n += 1
                if n >= k:
                    return True
                rr += dr * sign
                cc += dc * sign
        if n >= k:
            return True
    return False


def agent(observation, configuration):
    start = time.time()
    time_limit = 1.2  # margem segura abaixo dos 2 s por jogada

    rows = configuration.rows
    cols = configuration.columns
    k = configuration.inarow
    me = observation.mark
    opp = 3 - me
    board = list(observation.board)

    grid = [board[r * cols:(r + 1) * cols] for r in range(rows)]
    fill = [sum(1 for r in range(rows) if grid[r][c] != 0) for c in range(cols)]
    valid = [c for c in range(cols) if fill[c] < rows]
    if not valid:
        return 0

    # Colunas centrais primeiro: melhora a poda alfa-beta.
    center = (cols - 1) / 2
    order = sorted(range(cols), key=lambda c: abs(c - center))
    valid_order = [c for c in order if fill[c] < rows]

    def drop_row(c):
        return rows - 1 - fill[c]

    # 1) Vitória imediata.
    for c in valid_order:
        r = drop_row(c)
        grid[r][c] = me
        won = _is_win(grid, r, c, me, rows, cols, k)
        grid[r][c] = 0
        if won:
            return c

    # 2) Bloqueio de vitória imediata do adversário.
    for c in valid_order:
        r = drop_row(c)
        grid[r][c] = opp
        lost = _is_win(grid, r, c, opp, rows, cols, k)
        grid[r][c] = 0
        if lost:
            return c

    windows = _windows(rows, cols, k)
    weights = [0] + [10 ** (i - 1) for i in range(1, k + 1)]  # 1, 10, 100, ...
    node_counter = [0]

    def evaluate(p):
        q = 3 - p
        score = 0
        for w in windows:
            mine = theirs = 0
            for (r, c) in w:
                v = grid[r][c]
                if v == p:
                    mine += 1
                elif v == q:
                    theirs += 1
            if mine and not theirs:
                score += weights[mine]
            elif theirs and not mine:
                score -= weights[theirs]
        # Pequeno bônus por peças na coluna central.
        mid = cols // 2
        for r in range(rows):
            if grid[r][mid] == p:
                score += 3
            elif grid[r][mid] == q:
                score -= 3
        return score

    def negamax(p, depth, alpha, beta):
        node_counter[0] += 1
        if node_counter[0] % 256 == 0 and time.time() - start > time_limit:
            raise _Timeout()
        if depth == 0:
            return evaluate(p)
        best = -float("inf")
        moved = False
        for c in order:
            if fill[c] >= rows:
                continue
            moved = True
            r = drop_row(c)
            grid[r][c] = p
            fill[c] += 1
            if _is_win(grid, r, c, p, rows, cols, k):
                val = _WIN_SCORE + depth  # vitória mais rápida vale mais
            else:
                val = -negamax(3 - p, depth - 1, -beta, -alpha)
            fill[c] -= 1
            grid[r][c] = 0
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        if not moved:
            return 0  # empate
        return best

    # 3) Aprofundamento iterativo até acabar o tempo.
    best_move = valid_order[0]
    max_depth = rows * cols - sum(fill)
    try:
        for depth in range(1, max_depth + 1):
            iter_best = None
            iter_score = -float("inf")
            alpha = -float("inf")
            for c in valid_order:
                r = drop_row(c)
                grid[r][c] = me
                fill[c] += 1
                if _is_win(grid, r, c, me, rows, cols, k):
                    val = _WIN_SCORE + depth
                else:
                    try:
                        val = -negamax(opp, depth - 1, -float("inf"), -alpha)
                    except _Timeout:
                        fill[c] -= 1
                        grid[r][c] = 0
                        raise
                fill[c] -= 1
                grid[r][c] = 0
                if val > iter_score:
                    iter_score, iter_best = val, c
                if iter_score > alpha:
                    alpha = iter_score
            best_move = iter_best
            if iter_score >= _WIN_SCORE:  # vitória forçada encontrada
                break
    except _Timeout:
        pass

    return int(best_move)


def act(observation, configuration):
    return agent(observation, configuration)