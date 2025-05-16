import heapq
import random
from collections import defaultdict, deque
import numpy as np


# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------
MOVE_DIRS = {
    'S': (0, 0),
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}
DELTA_TO_MOVE = {v: k for k, v in MOVE_DIRS.items()}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -------------------------------------------------------------------
# Hungarian solver  (square-pad, n ≤ 20, O(n³))
# -------------------------------------------------------------------
def hungarian(cost_matrix):
    cost = np.array(cost_matrix, float).copy()
    n = cost.shape[0]
    cost -= cost.min(axis=1)[:, None]
    cost -= cost.min(axis=0)[None, :]
    star = np.zeros_like(cost, bool)
    row_cov = np.zeros(n, bool)
    col_cov = np.zeros(n, bool)

    rows, cols = np.where(cost == 0)
    for r, c in zip(rows, cols):
        if not row_cov[r] and not col_cov[c]:
            star[r, c] = True
            row_cov[r] = col_cov[c] = True
    row_cov[:] = col_cov[:] = False

    def cover_cols(): col_cov[:] = star.any(axis=0)
    cover_cols()
    prime = np.zeros_like(cost, bool)

    while col_cov.sum() < n:
        z_rows, z_cols = np.where((cost == 0) & (~row_cov)[:, None] & (~col_cov)[None, :])
        if z_rows.size == 0:
            m = cost[(~row_cov)[:, None] & (~col_cov)[None, :]].min()
            cost[~row_cov, :] -= m
            cost[:, col_cov] += m
            continue
        r, c = z_rows[0], z_cols[0]
        prime[r, c] = True
        s_col = np.where(star[r])[0]
        if s_col.size == 0:
            path = [(r, c)]
            while True:
                r_star = np.where(star[:, path[-1][1]])[0]
                if r_star.size == 0:
                    break
                r2 = r_star[0]
                c2 = np.where(prime[r2])[0][0]
                path += [(r2, path[-1][1]), (r2, c2)]
            for pr, pc in path:
                star[pr, pc] = not star[pr, pc]
            prime[:] = row_cov[:] = col_cov[:] = False
            cover_cols()
        else:
            row_cov[r] = True
            col_cov[s_col[0]] = False

    assign = [-1] * n
    for r, c in zip(*np.where(star)):
        assign[r] = c
    return assign


# -------------------------------------------------------------------
# Windowed Prioritised Planner
# -------------------------------------------------------------------
class AWPPlanner:
    def __init__(self, grid, horizon, reserve_horizon):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.H = horizon
        self.RH = reserve_horizon

    def in_bounds(self, p): return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols
    def passable(self, p): return self.grid[p] == 0

    def neighbors(self, pos):
        for d in MOVE_DIRS.values():
            nxt = (pos[0] + d[0], pos[1] + d[1])
            if self.in_bounds(nxt) and self.passable(nxt):
                yield nxt

    # -------- joint plan respecting a given robot order --------
    def plan_all(self, starts, goals, order=None):
        n = len(starts)
        order = list(order) if order is not None else list(range(n))
        paths = [[] for _ in range(n)]
        res_v, res_e = defaultdict(set), defaultdict(set)

        for rid in order:
            path = self._a_star(starts[rid], goals[rid], res_v, res_e)
            if len(path) < self.H + 1:
                path += [path[-1]] * (self.H + 1 - len(path))
            paths[rid] = path
            for t in range(min(self.RH, len(path))):
                res_v[t].add(path[t])
                if t > 0:
                    res_e[t - 1].add((path[t - 1], path[t]))
        return paths

    def _a_star(self, start, goal, res_v, res_e):
        if start == goal:
            return [start]
        open_heap = [(manhattan(start, goal), 0, start, 0, None)]
        best_g = {}

        while open_heap:
            f, g, pos, t, parent = heapq.heappop(open_heap)
            if best_g.get((pos, t), 1e9) <= g:
                continue
            best_g[(pos, t)] = g
            if pos == goal or t >= self.H:
                return self._reconstruct((pos, t, parent))
            for nxt in self.neighbors(pos):
                if (t + 1) in res_v and nxt in res_v[t + 1]:
                    continue
                if t in res_e and (pos, nxt) in res_e[t]:
                    continue
                ng = g + 1
                nf = ng + manhattan(nxt, goal)
                heapq.heappush(open_heap, (nf, ng, nxt, t + 1, (pos, t, parent)))
        return [start]

    @staticmethod
    def _reconstruct(node):
        path = []
        while node:
            pos, t, node = node
            path.append(pos)
        return list(reversed(path))


# -------------------------------------------------------------------
# Static map metrics
# -------------------------------------------------------------------
def longest_corridor(grid):
    best = 1
    for axis in [0, 1]:
        for idx in range(grid.shape[axis]):
            cnt = 0
            for jdx in range(grid.shape[1 - axis]):
                r, c = (idx, jdx) if axis == 0 else (jdx, idx)
                if grid[r, c] == 0:
                    cnt += 1
                    best = max(best, cnt)
                else:
                    cnt = 0
    return best


def free_ratio(grid): return (grid == 0).sum() / grid.size


# -------------------------------------------------------------------
# Main Agent class
# -------------------------------------------------------------------
class Agents:
    # trade-off constants
    N_PERMUTATIONS = 100
    RANDOM_SEED = 42
    RW_STEPS = 3        # 10-step random walk when stuck

    def __init__(self):
        self.grid = None
        self.t = 0
        # planner config
        self.H = 2
        self.RH = 30
        self.planner = None
        # robots
        self.n = 0
        self.plan_q = {}
        self.assigned = {}
        self.carrying = {}
        self.stationary = {}
        self.prev_pos = {}
        self.random_walk = {}
        # packages
        self.pinfo = {}
        self.pstatus = {}
        # parking
        self.parking_cells = []
        self._dist_cache = {}
        # misc
        self.plan_needed = True
        random.seed(self.RANDOM_SEED

    # ------------- shortest-path length (BFS + cache) -------------
    def _path_len(self, a, b):
        if a == b:
            return 0
        key = (a, b)
        if key in self._dist_cache:
            return self._dist_cache[key]
        R, C = self.grid.shape
        q = deque([(a, 0)])
        seen = {a}
        while q:
            pos, d = q.popleft()
            for delta in MOVE_DIRS.values():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if not (0 <= nxt[0] < R and 0 <= nxt[1] < C):
                    continue
                if self.grid[nxt] or nxt in seen:
                    continue
                if nxt == b:
                    self._dist_cache[key] = d + 1
                    return d + 1
                seen.add(nxt)
                q.append((nxt, d + 1))
        self._dist_cache[key] = 9999
        return 9999

    # ------------- parking spot discovery -------------
    def _degree(self, cell):
        r, c = cell
        deg = 0
        for d in MOVE_DIRS.values():
            nr, nc = r + d[0], c + d[1]
            if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]:
                if self.grid[nr, nc] == 0:
                    deg += 1
        return deg

    def _find_parking_cells(self):
        free = list(zip(*np.where(self.grid == 0)))
        parking = [p for p in free if self._degree(p) <= 1]
        if len(parking) < self.n:
            parking += [p for p in free if self._degree(p) == 2 and p not in parking]
        return parking or free

    # ------------- package bookkeeping -------------
    def _update_packages(self, pkgs):
        for pkg in pkgs:
            pid, sr, sc, tr, tc, _, dl = pkg
            if pid not in self.pinfo:
                self.pinfo[pid] = {'start': (sr - 1, sc - 1),
                                   'target': (tr - 1, tc - 1),
                                   'deadline': dl}
                self.pstatus[pid] = 'waiting'

    # ------------- Hungarian assignment -------------
    def _build_cost(self, rpos, waiting):
        m, n = len(rpos), len(waiting)
        size = max(m, n)
        C = np.full((size, size), 1e6)
        for i, pos in enumerate(rpos):
            for j, pid in enumerate(waiting):
                info = self.pinfo[pid]
                d1 = self._path_len(pos, info['start'])
                d2 = self._path_len(info['start'], info['target'])
                total = d1 + d2
                slack = info['deadline'] - (self.t + total)
                C[i, j] = total + (0 if slack >= 0 else 10 * -slack)
        return C

    def _assign_packages(self, rpos):
        waiting = [pid for pid, st in self.pstatus.items() if st == 'waiting']
        free_ids = [r for r in range(self.n)
                    if self.assigned[r] is None and self.carrying[r] is None]
        if not waiting or not free_ids:
            return
        C = self._build_cost([rpos[r] for r in free_ids], waiting)
        assign = hungarian(C)
        for i, r in enumerate(free_ids):
            j = assign[i]
            if j < len(waiting) and C[i, j] < 1e6:
                pid = waiting[j]
                self.assigned[r] = pid
                self.pstatus[pid] = 'assigned'
                self.plan_needed = True

    # ------------- determine goals incl. parking -------------
    def _determine_goals(self, rpos):
        goals = [None] * self.n
        used = set()
        occupied = set(rpos)

        # robots with tasks
        for r in range(self.n):
            if self.carrying[r] is not None:
                goals[r] = self.pinfo[self.carrying[r]]['target']
            elif self.assigned[r] is not None:
                goals[r] = self.pinfo[self.assigned[r]]['start']
            if goals[r] is not None:
                used.add(goals[r])

        # idle robots → nearest parking cell
        for r in range(self.n):
            if goals[r] is not None:
                continue
            best_p, best_d = None, 1e9
            for p in self.parking_cells:
                if p in used or p in occupied:
                    continue
                d = self._path_len(rpos[r], p)
                if d < best_d:
                    best_p, best_d = p, d
            goals[r] = best_p if best_p is not None else rpos[r]
            used.add(goals[r])
        return goals

    # ------------- plan scoring -------------
    def _score_paths(self, paths):
        sc = 0
        for rid, path in enumerate(paths):
            sc += len(path)
            pid = (self.carrying[rid] if self.carrying[rid] is not None
                   else self.assigned[rid])
            if pid:
                info = self.pinfo[pid]
                dist_left = self._path_len(path[-1], info['target'])
                finish = self.t + len(path) + dist_left
                late = max(0, finish - info['deadline'])
                sc += 10 * late
        return sc

    # ------------- multi-priority planning -------------
    def _plan_paths(self, rpos):
        goals = self._determine_goals(rpos)
        best_paths = self.planner.plan_all(rpos, goals)
        best_sc = self._score_paths(best_paths)

        ids = list(range(self.n))
        for _ in range(self.N_PERMUTATIONS):
            random.shuffle(ids)
            paths = self.planner.plan_all(rpos, goals, order=ids)
            sc = self._score_paths(paths)
            if sc < best_sc:
                best_paths, best_sc = paths, sc

        for r in range(self.n):
            moves = []
            for k in range(1, len(best_paths[r])):
                d = (best_paths[r][k][0] - best_paths[r][k - 1][0],
                     best_paths[r][k][1] - best_paths[r][k - 1][1])
                moves.append(DELTA_TO_MOVE.get(d, 'S'))
            self.plan_q[r] = deque(moves or ['S'])

    # ================================================================
    # Simulator hooks
    # ================================================================
    def init_agents(self, state):
        self.n = len(state['robots'])
        self.grid = np.array(state['map'], int)

        # adaptive horizons
        corr = longest_corridor(self.grid)
        ratio = free_ratio(self.grid)
        self.H = min(40, corr * 2) if corr >= 10 or ratio < 0.4 else max(5, corr + 2)
        self.RH = self.H * 3
        self.planner = AWPPlanner(self.grid, self.H, self.RH)

        # parking discovery
        self.parking_cells = self._find_parking_cells()

        for r in range(self.n):
            self.plan_q[r] = deque()
            self.assigned[r] = self.carrying[r] = None
            self.stationary[r] = 0
            self.random_walk[r] = 0

    # ---------------------------------------------------------------
    def _random_move(self, pos):
        """Return a random legal move letter from current position (or 'S')."""
        options = []
        for m, d in MOVE_DIRS.items():
            if m == 'S':
                continue
            nxt = (pos[0] + d[0], pos[1] + d[1])
            if (0 <= nxt[0] < self.grid.shape[0] and
                    0 <= nxt[1] < self.grid.shape[1] and
                    self.grid[nxt] == 0):
                options.append(m)
        return random.choice(options) if options else 'S'

    # ---------------------------------------------------------------
    def get_actions(self, state):
        self.t = state['time_step']
        self._update_packages(state['packages'])

        # --- parse robot state ---
        rpos = []
        for r, (row, col, carry) in enumerate(state['robots']):
            pos = (row - 1, col - 1)
            rpos.append(pos)
            self.stationary[r] = (self.stationary[r] + 1
                                   if pos == self.prev_pos.get(r) else 0)
            self.prev_pos[r] = pos

            if carry:
                self.carrying[r] = carry
            elif self.carrying[r]:
                self.pstatus[self.carrying[r]] = 'done'
                self.carrying[r] = None

        # --- new jobs ---
        self._assign_packages(rpos)

        # --- detect stuck & start random walk ---
        for r in range(self.n):
            if self.stationary[r] >= self.H and self.random_walk[r] == 0:
                self.random_walk[r] = self.RW_STEPS
                self.stationary[r] = 0

        # --- planning ---
        need_plan = (self.plan_needed or
                     any(len(self.plan_q[r]) == 0 for r in range(self.n)))
        if need_plan:
            self._plan_paths(rpos)
            self.plan_needed = False

        # --- actions ---
        actions = []
        for r in range(self.n):
            # random walk overrides plan
            if self.random_walk[r] > 0:
                move = self._random_move(rpos[r])
                self.random_walk[r] -= 1
                if self.plan_q[r]:
                    self.plan_q[r].popleft()
            else:
                move = self.plan_q[r].popleft() if self.plan_q[r] else 'S'

            dr, dc = MOVE_DIRS[move]
            nxt = (rpos[r][0] + dr, rpos[r][1] + dc)
            pkg_act = '0'

            # pick
            if self.carrying[r] is None and self.assigned[r] is not None:
                pid = self.assigned[r]
                if nxt == self.pinfo[pid]['start']:
                    pkg_act = '1'
                    self.carrying[r] = pid
                    self.pstatus[pid] = 'carrying'
                    self.assigned[r] = None
                    self.plan_needed = True

            # drop
            if self.carrying[r] is not None:
                pid = self.carrying[r]
                if nxt == self.pinfo[pid]['target']:
                    pkg_act = '2'
                    self.carrying[r] = None
                    self.pstatus[pid] = 'done'
                    self.plan_needed = True

            actions.append((move, pkg_act))

        return actions
