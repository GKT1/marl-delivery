import heapq
import random
from collections import defaultdict, deque
import numpy as np

"""====================================================================
Adaptive multi-robot agent – v7-cbs
----------------------------------
Adds an **on-demand Conflict-Based Search (CBS)** fallback:

⊳  Run WPP with N permutations (default 3 000).
⊳  If its best score  >  LB × 1.5   (LB = sum shortest paths) →
    invoke CBS with a 60-step horizon to find an optimal joint plan.
⊳  CBS prunes when node count > MAX_CBS_NODES (default 20 000)
    to keep runtime finite.

Keeps:
• Parking for idle robots (dead-end & side-cell spots).
• True-distance Hungarian assignment.
• Adaptive horizons & stall detector.

Tune constants at the top for speed / quality.
===================================================================="""


# -------------------------------------------------------------------
# Tunable constants
# -------------------------------------------------------------------
N_PERMUTATIONS      = 100    # WPP priority samples
CBS_HORIZON         = 60      # joint-plan depth for CBS
CBS_THRESHOLD_FACTOR = 1.5    # when to trigger CBS ( > LB×factor )
MAX_CBS_NODES       = 20000   # hard cap on CBS high-level nodes
RANDOM_SEED         = 42


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
# Hungarian solver (square-pad, n ≤ 20)
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
class WPP:
    def __init__(self, grid, H, RH):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.H  = H     # planning horizon
        self.RH = RH    # reservation horizon

    def in_bounds(self, p): return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols
    def passable(self, p): return self.grid[p] == 0

    def neighbors(self, pos):
        for d in MOVE_DIRS.values():
            nxt = (pos[0] + d[0], pos[1] + d[1])
            if self.in_bounds(nxt) and self.passable(nxt):
                yield nxt

    # --- joint plan respecting a given robot order ---
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
        pq = [(manhattan(start, goal), 0, start, 0, None)]
        best = {}
        while pq:
            f, g, pos, t, parent = heapq.heappop(pq)
            if best.get((pos, t), 1e9) <= g:
                continue
            best[(pos, t)] = g
            if pos == goal or t >= self.H:
                return self._reconstruct((pos, t, parent))
            for nxt in self.neighbors(pos):
                if (t + 1) in res_v and nxt in res_v[t + 1]:
                    continue
                if t in res_e and (pos, nxt) in res_e[t]:
                    continue
                ng = g + 1
                nf = ng + manhattan(nxt, goal)
                heapq.heappush(pq, (nf, ng, nxt, t + 1, (pos, t, parent)))
        return [start]

    @staticmethod
    def _reconstruct(node):
        path = []
        while node:
            pos, t, node = node
            path.append(pos)
        return list(reversed(path))


# -------------------------------------------------------------------
# Conflict-Based Search (optimises joint plan for fixed horizon)
# -------------------------------------------------------------------
class CBS:
    class Constraint:
        def __init__(self, agent, cell, time, edge=None):
            self.agent, self.cell, self.time, self.edge = agent, cell, time, edge  # edge = (from,to)

        def conflicts(self, other):
            if self.agent != other.agent:
                return False
            if self.edge and other.edge:
                return self.edge == other.edge and self.time == other.time
            return self.cell == other.cell and self.time == other.time

    def __init__(self, grid, horizon):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.H = horizon
        self.lowlevel_cache = {}

    # ---- low-level planner with constraints ----
    def _plan(self, start, goal, constraints, aid):
        key = (start, goal, tuple((c.cell, c.time, c.edge) for c in constraints if c.agent == aid))
        if key in self.lowlevel_cache:
            return self.lowlevel_cache[key]

        def blocked(pos, t):
            for c in constraints:
                if c.agent == aid:
                    if c.edge:
                        continue
                    if c.time == t and c.cell == pos:
                        return True
            return False

        def blocked_edge(p, q, t):
            for c in constraints:
                if c.agent == aid and c.edge and c.time == t and c.edge == (p, q):
                    return True
            return False

        pq = [(manhattan(start, goal), 0, start, 0, None)]
        seen = {}

        while pq:
            f, g, pos, t, parent = heapq.heappop(pq)
            if seen.get((pos, t), 1e9) <= g:
                continue
            seen[(pos, t)] = g
            if pos == goal or t >= self.H:
                path = []
                node = (pos, t, parent)
                while node:
                    cell, tt, node = node
                    path.append(cell)
                path = list(reversed(path))
                if len(path) < self.H + 1:
                    path += [path[-1]] * (self.H + 1 - len(path))
                self.lowlevel_cache[key] = path
                return path
            for d in MOVE_DIRS.values():
                nxt = (pos[0] + d[0], pos[1] + d[1])
                if not (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols):
                    continue
                if self.grid[nxt] == 1:
                    continue
                if blocked(nxt, t + 1) or blocked_edge(pos, nxt, t):
                    continue
                ng = g + 1
                nf = ng + manhattan(nxt, goal)
                heapq.heappush(pq, (nf, ng, nxt, t + 1, (pos, t, parent)))
        return None  # no path

    # ---- high-level CBS search ----
    def find_solution(self, starts, goals):
        n = len(starts)
        root = {'paths': [], 'constraints': [], 'cost': 0}
        for aid in range(n):
            p = self._plan(starts[aid], goals[aid], [], aid)
            if p is None:
                return None
            root['paths'].append(p)
            root['cost'] += len(p)

        open_set = [(root['cost'], 0, root)]
        node_counter = 1

        def first_conflict(paths):
            T = len(paths[0])
            for t in range(T):
                pos_at_t = {}
                for aid, path in enumerate(paths):
                    cell = path[t] if t < len(path) else path[-1]
                    if cell in pos_at_t:
                        return (aid, pos_at_t[cell], cell, t, None)  # vertex
                    pos_at_t[cell] = aid
                for aid, path in enumerate(paths):
                    if t == 0:
                        continue
                    prev = path[t - 1] if t - 1 < len(path) else path[-1]
                    curr = path[t] if t < len(path) else path[-1]
                    for bid, path2 in enumerate(paths):
                        prev2 = path2[t - 1] if t - 1 < len(path2) else path2[-1]
                        curr2 = path2[t] if t < len(path2) else path2[-1]
                        if aid < bid and prev == curr2 and curr == prev2:
                            return (aid, bid, (prev, curr), t - 1, (prev, curr2))  # edge swap
            return None

        while open_set and node_counter < MAX_CBS_NODES:
            cost, _, node = heapq.heappop(open_set)
            conflict = first_conflict(node['paths'])
            if conflict is None:
                return node['paths']  # found consistent
            a1, a2, cell_or_edge, t, extra = conflict
            for agent in (a1, a2):
                new_constraints = list(node['constraints'])
                if extra is None:  # vertex
                    new_constraints.append(self.Constraint(agent, cell_or_edge, t + 1))
                else:              # edge
                    edge = (cell_or_edge[0], cell_or_edge[1])
                    new_constraints.append(self.Constraint(agent, None, t, edge=edge))
                new_paths = node['paths'].copy()
                new_path = self._plan(starts[agent], goals[agent], new_constraints, agent)
                if new_path is None:
                    continue
                new_paths[agent] = new_path
                new_cost = sum(len(p) for p in new_paths)
                new_node = {'paths': new_paths, 'constraints': new_constraints, 'cost': new_cost}
                heapq.heappush(open_set, (new_cost, node_counter, new_node))
                node_counter += 1
        return None  # fallback failure


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
    def __init__(self):
        self.grid = None
        self.t = 0
        # adaptive planner params
        self.H = 10
        self.RH = 30
        self.wpp = None
        # robots
        self.n = 0
        self.plan_q = {}
        self.assigned = {}
        self.carrying = {}
        self.stationary = {}
        self.prev_pos = {}
        # packages
        self.pinfo = {}
        self.pstatus = {}
        # parking
        self.parking_cells = []
        # caches & flags
        self._dist_cache = {}
        self.plan_needed = True
        random.seed(RANDOM_SEED)

    # ------------- distance on static map (BFS cache) -------------
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

    # ------------- parking discovery -------------
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
        free = [r for r in range(self.n) if self.assigned[r] is None and self.carrying[r] is None]
        if not waiting or not free:
            return
        C = self._build_cost([rpos[r] for r in free], waiting)
        assign = hungarian(C)
        for i, r in enumerate(free):
            j = assign[i]
            if j < len(waiting) and C[i, j] < 1e6:
                pid = waiting[j]
                self.assigned[r] = pid
                self.pstatus[pid] = 'assigned'
                self.plan_needed = True

    # ------------- goal selection incl. parking -------------
    def _determine_goals(self, rpos):
        goals = [None] * self.n
        used = set()
        occupied = set(rpos)
        for r in range(self.n):
            if self.carrying[r] is not None:
                goals[r] = self.pinfo[self.carrying[r]]['target']
            elif self.assigned[r] is not None:
                goals[r] = self.pinfo[self.assigned[r]]['start']
            if goals[r] is not None:
                used.add(goals[r])
        for r in range(self.n):
            if goals[r] is not None:
                continue
            best, bestd = None, 1e9
            for p in self.parking_cells:
                if p in used or p in occupied:
                    continue
                d = self._path_len(rpos[r], p)
                if d < bestd:
                    best, bestd = p, d
            goals[r] = best if best else rpos[r]
            used.add(goals[r])
        return goals

    # ------------- score joint paths -------------
    def _score_paths(self, paths):
        sc = 0
        for rid, path in enumerate(paths):
            sc += len(path)
            pid = self.carrying[rid] if self.carrying[rid] else self.assigned[rid]
            if pid:
                info = self.pinfo[pid]
                finish = self.t + len(path) + self._path_len(path[-1], info['target'])
                late = max(0, finish - info['deadline'])
                sc += late * 10
        return sc

    # ------------- WPP + permutations -------------
    def _wpp_plan(self, rpos, goals):
        best_paths = self.wpp.plan_all(rpos, goals)
        best_sc = self._score_paths(best_paths)
        ids = list(range(self.n))
        for _ in range(N_PERMUTATIONS):
            random.shuffle(ids)
            paths = self.wpp.plan_all(rpos, goals, order=ids)
            sc = self._score_paths(paths)
            if sc < best_sc:
                best_paths, best_sc = paths, sc
        return best_paths, best_sc

    # ------------- CBS fallback -------------
    def _cbs_plan(self, rpos, goals):
        cbs = CBS(self.grid, CBS_HORIZON)
        sol = cbs.find_solution(rpos, goals)
        return sol

    # ------------- convert joint path → deque of moves -------------
    def _paths_to_queues(self, paths):
        for r in range(self.n):
            moves = []
            for k in range(1, len(paths[r])):
                d = (paths[r][k][0] - paths[r][k - 1][0],
                     paths[r][k][1] - paths[r][k - 1][1])
                moves.append(DELTA_TO_MOVE.get(d, 'S'))
            self.plan_q[r] = deque(moves or ['S'])

    # ------------- planning wrapper -------------
    def _plan_paths(self, rpos):
        goals = self._determine_goals(rpos)
        paths, sc = self._wpp_plan(rpos, goals)

        lb = sum(self._path_len(rpos[i], goals[i]) for i in range(self.n))
        if sc > lb * CBS_THRESHOLD_FACTOR:
            cbs_paths = self._cbs_plan(rpos, goals)
            if cbs_paths:
                paths = cbs_paths
        self._paths_to_queues(paths)

    # ------------- simulator hooks -------------
    def init_agents(self, state):
        self.n = len(state['robots'])
        self.grid = np.array(state['map'], int)
        corr = longest_corridor(self.grid)
        ratio = free_ratio(self.grid)
        # longer look-ahead on tight maps
        if corr >= 10 or ratio < 0.4:
            self.H = min(60, corr * 3)
        else:
            self.H = max(8, corr + 3)
        self.RH = self.H * 3
        self.wpp = WPP(self.grid, self.H, self.RH)
        self.parking_cells = self._find_parking_cells()

        for r in range(self.n):
            self.plan_q[r] = deque()
            self.assigned[r] = self.carrying[r] = None
            self.stationary[r] = 0

    def get_actions(self, state):
        self.t = state['time_step']
        self._update_packages(state['packages'])

        # ---- robot state ----
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

        # ---- assignments & stalls ----
        self._assign_packages(rpos)
        if any(self.stationary[r] >= self.H for r in range(self.n)):
            self.plan_needed = True
            for r in range(self.n):
                self.stationary[r] = 0

        # ---- plan when needed ----
        if self.plan_needed or any(len(self.plan_q[r]) == 0 for r in range(self.n)):
            self._plan_paths(rpos)
            self.plan_needed = False

        # ---- emit actions ----
        actions = []
        for r in range(self.n):
            move = self.plan_q[r].popleft() if self.plan_q[r] else 'S'
            dr, dc = MOVE_DIRS[move]
            nxt = (rpos[r][0] + dr, rpos[r][1] + dc)
            pkg_act = '0'

            # pick-up
            if self.carrying[r] is None and self.assigned[r] is not None:
                pid = self.assigned[r]
                if nxt == self.pinfo[pid]['start']:
                    pkg_act = '1'
                    self.carrying[r] = pid
                    self.pstatus[pid] = 'carrying'
                    self.assigned[r] = None
                    self.plan_needed = True
            # drop-off
            if self.carrying[r] is not None:
                pid = self.carrying[r]
                if nxt == self.pinfo[pid]['target']:
                    pkg_act = '2'
                    self.carrying[r] = None
                    self.pstatus[pid] = 'done'
                    self.plan_needed = True

            actions.append((move, pkg_act))
        return actions
