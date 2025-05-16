import heapq, random, numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# =========================================================
#                    Visualizer
# =========================================================
class Visualizer:
    COLORS = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
              'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.frames=[]; self.fig,self.ax = plt.subplots()
        self.ax.set_aspect('equal'); self.ax.axis('off')
    def _draw(self, robots, pkgs):
        self.ax.clear(); self.ax.axis('off')
        oy,ox = np.where(self.grid==1)
        self.ax.scatter(ox,-oy,s=60,c='k',marker='s')
        for pid,sr,sc,_,_,_,_ in pkgs:
            self.ax.scatter(sc-1,-(sr-1),s=40,c='orange',marker='s')
        for rid,(r,c,carry) in enumerate(robots):
            col=self.COLORS[rid%len(self.COLORS)]
            self.ax.scatter(c-1,-(r-1),s=120,c=col,marker='o',edgecolors='k')
            if carry:
                self.ax.scatter(c-1,-(r-1),s=40,c='yellow',marker='s',edgecolors='k')
        self.ax.set_xlim(-.5,self.grid.shape[1]-.5)
        self.ax.set_ylim(-self.grid.shape[0]+.5,.5)
        self.fig.canvas.draw()
        self.frames.append(np.asarray(self.fig.canvas.buffer_rgba()))
    def step(self, robots, pkgs): self._draw(robots, pkgs); plt.pause(0.001)
    def save_gif(self, path='run.gif', fps=4):
        if imageio is None:
            print("Install imageio for GIF export"); return
        imageio.mimsave(path, self.frames, fps=fps)

# =========================================================
#             Constants & small helpers
# =========================================================
MOVE_DIRS={'S':(0,0),'L':(0,-1),'R':(0,1),'U':(-1,0),'D':(1,0)}
DELTA2MOVE={v:k for k,v in MOVE_DIRS.items()}
def manh(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

N_PERM=1000         # permutations each tick
CBS_H=60            # CBS horizon
THRESH=1.6          # score / lower-bound that triggers CBS
MAX_CBS=20000       # CBS node cap
random.seed(42)

# =========================================================
#         Minimal Hungarian (unchanged)
# =========================================================
def hungarian(c):
    c=np.array(c,float); n=c.shape[0]
    c-=c.min(1)[:,None]; c-=c.min(0)
    star=np.zeros_like(c,bool); row=np.zeros(n,bool); col=np.zeros(n,bool)
    r0,c0=np.where(c==0)
    for r,k in zip(r0,c0):
        if not row[r] and not col[k]:
            star[r,k]=row[r]=col[k]=True
    row[:]=col[:]=False
    def cover(): col[:] = star.any(0)
    cover(); prime=np.zeros_like(c,bool)
    while col.sum()<n:
        zR,zC=np.where((c==0)&(~row)[:,None]&(~col)[None,:])
        if zR.size==0:
            m=c[(~row)[:,None]&(~col)[None,:]].min()
            c[~row]-=m; c[:,col]+=m; continue
        r,k=zR[0],zC[0]; prime[r,k]=True
        s=np.where(star[r])[0]
        if s.size==0:
            path=[(r,k)]
            while True:
                rs=np.where(star[:,path[-1][1]])[0]
                if rs.size==0: break
                r2=rs[0]; k2=np.where(prime[r2])[0][0]
                path+=[(r2,path[-1][1]),(r2,k2)]
            for pr,pc in path: star[pr,pc]=~star[pr,pc]
            prime[:]=row[:]=col[:]=False; cover()
        else:
            row[r]=True; col[s[0]]=False
    out=[-1]*n
    for r,k in zip(*np.where(star)): out[r]=k
    return out

# =========================================================
#       Windowed Prioritised Planner  (WPP)
# =========================================================
class WPP:
    def __init__(self, grid, H, RH):
        self.grid, self.H, self.RH = grid, H, RH
        self.rows, self.cols = grid.shape
    def inb(self,p): return 0<=p[0]<self.rows and 0<=p[1]<self.cols
    def free(self,p): return self.grid[p]==0
    def nbrs(self,p):
        for d in MOVE_DIRS.values():
            q=(p[0]+d[0],p[1]+d[1])
            if self.inb(q) and self.free(q): yield q
    def _astar(self,s,g,resV,resE):
        if s==g: return [s]
        pq=[(manh(s,g),0,s,0,None)]; best={}
        while pq:
            f,gc,pos,t,parent=heapq.heappop(pq)
            if best.get((pos,t),1e9)<=gc: continue
            best[(pos,t)]=gc
            if pos==g or t>=self.H:
                path=[]; node=(pos,t,parent)
                while node:
                    cell,tt,node=node; path.append(cell)
                path.reverse()
                if len(path)<self.H+1:
                    path+=[path[-1]]*(self.H+1-len(path))
                return path
            for q in self.nbrs(pos):
                if (t+1) in resV and q in resV[t+1]: continue
                if t in resE and (pos,q) in resE[t]: continue
                ng=gc+1; nf=ng+manh(q,g)
                heapq.heappush(pq,(nf,ng,q,t+1,(pos,t,parent)))
        return [s]
    def plan(self,starts,goals,order=None):
        n=len(starts); order=list(order) if order else list(range(n))
        paths=[[]]*n; resV,resE=defaultdict(set),defaultdict(set)
        for aid in order:
            p=self._astar(starts[aid],goals[aid],resV,resE)
            paths[aid]=p
            for t in range(min(self.RH,len(p))):
                resV[t].add(p[t])
                if t: resE[t-1].add((p[t-1],p[t]))
        return paths

# =========================================================
#              Conflict-Based Search  (CBS)
# =========================================================
class CBS:
    class C:
        def __init__(self, agent, cell, time, edge=None):
            self.a, self.cell, self.t, self.edge = agent, cell, time, edge
    def __init__(self, grid, H):
        self.grid, self.H = grid, H
        self.rows, self.cols = grid.shape
        self.cache={}
    def inb(self,p): return 0<=p[0]<self.rows and 0<=p[1]<self.cols
    def free(self,p): return self.grid[p]==0
    def nbrs(self,p):
        for d in MOVE_DIRS.values():
            q=(p[0]+d[0],p[1]+d[1])
            if self.inb(q) and self.free(q): yield q
    # low-level
    def _plan(self,s,g,cons,aid):
        key=(s,g,tuple((c.cell,c.t,c.edge) for c in cons if c.a==aid))
        if key in self.cache: return self.cache[key]
        def blocked(pos,t):
            return any(c.a==aid and not c.edge and c.t==t and c.cell==pos for c in cons)
        def blockedE(p,q,t):
            return any(c.a==aid and c.edge and c.t==t and c.edge==(p,q) for c in cons)
        pq=[(manh(s,g),0,s,0,None)]; best={}
        while pq:
            f,gc,pos,t,parent=heapq.heappop(pq)
            if best.get((pos,t),1e9)<=gc: continue
            best[(pos,t)]=gc
            if pos==g or t>=self.H:
                path=[]; node=(pos,t,parent)
                while node:
                    cell,tt,node=node; path.append(cell)
                path.reverse()
                if len(path)<self.H+1:
                    path+=[path[-1]]*(self.H+1-len(path))
                self.cache[key]=path; return path
            for q in self.nbrs(pos):
                if blocked(q,t+1) or blockedE(pos,q,t): continue
                ng=gc+1; nf=ng+manh(q,g)
                heapq.heappush(pq,(nf,ng,q,t+1,(pos,t,parent)))
        return None
    # conflict detection
    def _conflict(self,paths):
        T=len(paths[0])
        for t in range(T):
            occ={}
            for a,p in enumerate(paths):
                cell=p[t] if t<len(p) else p[-1]
                if cell in occ: return ('vert',a,occ[cell],cell,t)
                occ[cell]=a
            for a,p in enumerate(paths):
                if t==0: continue
                prev,cur=p[t-1] if t-1<len(p) else p[-1], p[t] if t<len(p) else p[-1]
                for b,p2 in enumerate(paths):
                    prev2,cur2=p2[t-1] if t-1<len(p2) else p2[-1], p2[t] if t<len(p2) else p2[-1]
                    if a<b and prev==cur2 and cur==prev2:
                        return ('edge',a,b,(prev,cur),t-1)
        return None
    # high-level search
    def solve(self,starts,goals,max_nodes=MAX_CBS):
        n=len(starts)
        root={'cons':[], 'paths':[], 'cost':0}
        for aid in range(n):
            p=self._plan(starts[aid],goals[aid],[],aid)
            if p is None: return None
            root['paths'].append(p); root['cost']+=len(p)
        open=[(root['cost'],0,root)]; nid=1
        while open and nid<max_nodes:
            cost,_,node=heapq.heappop(open)
            confl=self._conflict(node['paths'])
            if confl is None: return node['paths']
            typ,a1,a2,data,t=confl
            for ag in (a1,a2):
                cons=list(node['cons'])
                if typ=='vert':
                    cons.append(self.C(ag,data,t+1))
                else:
                    cons.append(self.C(ag,None,t,edge=data))
                paths=node['paths'].copy()
                newp=self._plan(starts[ag],goals[ag],cons,ag)
                if newp is None: continue
                paths[ag]=newp
                newcost=sum(len(p) for p in paths)
                heapq.heappush(open,(newcost,nid,{'cons':cons,'paths':paths,'cost':newcost}))
                nid+=1
        return None

# =========================================================
#                       Agents
# =========================================================
class Agents:
    def __init__(self):
        self.grid=None; self.n=0
        self.H=20; self.RH=60; self.wpp=None; self.viz=None
        self.plan_q={}; self.assigned={}; self.carrying={}
        self.pinfo={}; self.pstatus={}
        self.dist_cache={}
    # BFS distance
    def dist(self,a,b):
        if a==b: return 0
        key=(a,b)
        if key in self.dist_cache: return self.dist_cache[key]
        R,C=self.grid.shape; dq=deque([(a,0)]); seen={a}
        while dq:
            pos,d=dq.popleft()
            for dt in MOVE_DIRS.values():
                q=(pos[0]+dt[0],pos[1]+dt[1])
                if not (0<=q[0]<R and 0<=q[1]<C): continue
                if self.grid[q] or q in seen: continue
                if q==b:
                    self.dist_cache[key]=d+1; return d+1
                seen.add(q); dq.append((q,d+1))
        self.dist_cache[key]=9999; return 9999
    # parking discovery
    def deg(self,c):
        r,c0=c; d=0
        for dt in MOVE_DIRS.values():
            nr,nc=r+dt[0],c0+dt[1]
            if 0<=nr<self.grid.shape[0] and 0<=nc<self.grid.shape[1]:
                if self.grid[nr,nc]==0: d+=1
        return d
    def parks(self):
        free=list(zip(*np.where(self.grid==0)))
        pk=[p for p in free if self.deg(p)<=1]
        if len(pk)<self.n: pk+=[p for p in free if self.deg(p)==2 and p not in pk]
        return pk or free
    # init
    def init_agents(self,state):
        self.grid=np.array(state['map'],int); self.n=len(state['robots'])
        corr=max((self.grid[i]==0).sum() for i in range(self.grid.shape[0]))
        self.H=min(60,corr*3) if corr>=10 else 20; self.RH=self.H*3
        self.wpp=WPP(self.grid,self.H,self.RH); self.parking=self.parks()
        for r in range(self.n):
            self.plan_q[r]=deque(); self.assigned[r]=self.carrying[r]=None
        self.viz=Visualizer(state['map'])
    # pkg bookkeeping
    def upd_pkgs(self,pkgs):
        for pid,sr,sc,tr,tc,_,dl in pkgs:
            if pid not in self.pinfo:
                self.pinfo[pid]={'start':(sr-1,sc-1),'target':(tr-1,tc-1),'deadline':dl}
                self.pstatus[pid]='waiting'
    # assignment
    def assign(self,rpos,t):
        wait=[p for p,st in self.pstatus.items() if st=='waiting']
        free=[r for r in range(self.n) if self.assigned[r] is None and self.carrying[r] is None]
        if not wait or not free: return
        C=np.full((max(len(free),len(wait)),)*2,1e6)
        for i,r in enumerate(free):
            for j,p in enumerate(wait):
                info=self.pinfo[p]
                tot=self.dist(rpos[r],info['start'])+self.dist(info['start'],info['target'])
                slack=info['deadline']-(t+tot)
                C[i,j]=tot+(0 if slack>=0 else 10*-slack)
        sol=hungarian(C)
        for i,r in enumerate(free):
            j=sol[i]
            if j<len(wait) and C[i,j]<1e6:
                pid=wait[j]; self.assigned[r]=pid; self.pstatus[pid]='assigned'
    # goals incl parking
    def goals(self,rpos):
        g=[None]*self.n; used=set(); occ=set(rpos)
        for r in range(self.n):
            if self.carrying[r]: g[r]=self.pinfo[self.carrying[r]]['target']
            elif self.assigned[r]: g[r]=self.pinfo[self.assigned[r]]['start']
            if g[r]: used.add(g[r])
        for r in range(self.n):
            if g[r]: continue
            best=None; bd=1e9
            for p in self.parking:
                if p in used or p in occ: continue
                d=self.dist(rpos[r],p)
                if d<bd: bd=d; best=p
            g[r]=best if best else rpos[r]; used.add(g[r])
        return g
    # score
    def score(self,paths,rpos,g):
        sc=0
        for r,p in enumerate(paths):
            sc+=len(p)
            pid=self.carrying[r] or self.assigned[r]
            if pid:
                info=self.pinfo[pid]
                late=max(0,(len(p)+self.dist(p[-1],info['target']))-info['deadline'])
                sc+=10*late
        return sc
    # force move
    def force(self,pos,goal):
        best=None;bd=1e9
        for mv,(dr,dc) in MOVE_DIRS.items():
            if mv=='S': continue
            q=(pos[0]+dr,pos[1]+dc)
            if 0<=q[0]<self.grid.shape[0] and 0<=q[1]<self.grid.shape[1] and self.grid[q]==0:
                d=self.dist(q,goal)
                if d<bd: bd=d; best=mv
        return best or 'S'
    # full plan
    def plan(self,rpos,t):
        g=self.goals(rpos); self.last_goals=g
        paths=self.wpp.plan(rpos,g); best=self.score(paths,rpos,g)
        order=list(range(self.n))
        for _ in range(N_PERM):
            random.shuffle(order)
            p2=self.wpp.plan(rpos,g,order); s2=self.score(p2,rpos,g)
            if s2<best: paths,best=p2,s2
        lb=sum(self.dist(rpos[i],g[i]) for i in range(self.n))
        if best>lb*THRESH:
            sol=CBS(self.grid,CBS_H).solve(rpos,g)
            if sol: paths=sol
        self.plan_q={}
        for r in range(self.n):
            mv=[]
            for k in range(1,len(paths[r])):
                d=(paths[r][k][0]-paths[r][k-1][0],
                   paths[r][k][1]-paths[r][k-1][1])
                mv.append(DELTA2MOVE.get(d,'S'))
            self.plan_q[r]=deque(mv or ['S'])
    # main
    def get_actions(self,state):
        t=state['time_step']
        self.viz.step(state['robots'],state['packages'])
        self.upd_pkgs(state['packages'])
        rpos=[]
        for r,(row,col,carry) in enumerate(state['robots']):
            pos=(row-1,col-1); rpos.append(pos)
            if carry: self.carrying[r]=carry
            elif self.carrying.get(r):
                self.pstatus[self.carrying[r]]='done'; self.carrying[r]=None
        self.assign(rpos,t)
        self.plan(rpos,t)
        acts=[]
        for r in range(self.n):
            mv=self.plan_q[r].popleft() if self.plan_q[r] else 'S'
            if mv=='S': mv=self.force(rpos[r],self.last_goals[r])
            dr,dc=MOVE_DIRS[mv]; nxt=(rpos[r][0]+dr,rpos[r][1]+dc)
            act='0'
            if self.carrying[r] is None and self.assigned[r] is not None:
                pid=self.assigned[r]
                if nxt==self.pinfo[pid]['start']:
                    act='1'; self.carrying[r]=pid; self.pstatus[pid]='carrying'; self.assigned[r]=None
            if self.carrying[r] is not None:
                pid=self.carrying[r]
                if nxt==self.pinfo[pid]['target']:
                    act='2'; self.carrying[r]=None; self.pstatus[pid]='done'
            acts.append((mv,act))
        return acts
