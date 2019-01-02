'''
Nils Napp
Sliding Probelm for AI-Class
'''
from slideproblem import *
from heapq import *
from operator import *
import operator
import time
## you likely need to inport some more modules to do the serach


class Searches:
    
    def tree_bfs(self, problem):
        #reset the node counter for profiling
        print("tree_BFS")
        Node.nodeCount=0
        n=Node(None,None,0,problem.initialState)
        print(n)
        frontier=[n]
        print("frontierlen: ",len(frontier))
        while len(frontier) > 0:
            n = frontier.pop(0)
            for a in p.applicable(n.state):
                nc=childNode(n,a,p)
                if nc.state == p.goalState:
                    return solution(nc)
                else:
                    frontier.append(nc)
            
    def graph_bfs(self, problem):
        print("graph_bfs")
        Node.nodeCount=0
        n=Node(None,None,0,problem.initialState)
        print(n)
        frontier=[n]
        print(frontier)
        explored=set()
        print("explored: ",explored)
        while len(frontier) > 0:
            n = frontier.pop(0)
            #print("loop n: ",n)
            for a in p.applicable(n.state):
                nc=child_node(n,a,p)
                if nc.state == p.goalState:
                    return solution(nc)
                else:
                    childState=nc.state.toTuple()
                    if not(childState in explored):
                        frontier.append(nc)
                        explored.add(childState)    
        
    
    def recursiveDL_DFS(self, lim,problem):
        n=Node(None,None,0,problem.initialState)
        return self.depthLimitedDFS(n,lim,problem)
    
    def depthLimitedDFS(self, n, lim, problem):
        #print lim
        #print n
    
        #reasons to cut off brnaches    
        if n.state == problem.goalState:
            return solution(n)
        elif lim == 0:
            return None
       
        cutoff=False    
        for a in p.applicable(n.state):
            nc=child_node(n,a,problem)
            result = self.depthLimitedDFS(nc,lim-1,problem)
    
            if not result==None:
                return result
    
        return None        
    
    def id_dfs(self,problem):
    
        Node.nodeCount=0
        
        maxLim=32
        for d in range(1,maxLim):
            result = self.recursiveDL_DFS(d,problem)
            if not result == None:
                return result
        print('Hit max limit of ' + str(maxLim))
        return None
        
        
    def h_1(self,s0: State,sf: State ) -> numbers.Real:
        x=s0.toTuple()
        y=sf.toTuple()
        #print("x: ",x," y: ",y)
        dist=[]
        c=len(y)
        d=len(x)
        k=0
        l=0
        sums=0
        for i in range(c):
            for j in range(c):
                value= y[i][j]
                #print("Value: ",value)
                for k in range(d):
                    for l in range(d):
                        #print("x: ",x[k][l])
                        if(x[k][l]==value):
                            dist.append (abs((k-i)+(l-j)))
                            sums += (abs((k - i)) + abs((l - j)))
                            #print("dist: ",dist,"sums: ",sums)
                            #print("K: ",k,"I: ",i,"l: ",l,"j: ",j)

        print("sums: ",sums)
        return sums
        
    def h_2(self,s0: State,sf: State ) -> numbers.Real:
        x = s0.toTuple()
        y = sf.toTuple()
        return abs(x.pos[0] - y.pos[0]) + abs(x.pos[1] - y.pos[0])
        return "somethign"
        
    def a_star_tree(self,problem : Problem) -> tuple:
        #return "Totally fake return value"
        print("astar tree")
        h1=self.h_1(problem.initialState, problem.goalState)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        start=problem.initialState.toTuple()
        goal=problem.goalState.toTuple()
        gscore = {start: 0}
        fscore = {start: h1}
        print("start: ", start, "goal: ", goal, "gscore: ", gscore, "fscore: ", fscore)
        oheap = []
        heappush(oheap, (fscore[start], start))
        print("oheap: ",oheap)

        while oheap:

            current = heappop(oheap)[1]
            print("current: ",current)
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data
            close_set.add(current)
            for i, j in neighbors:
                neighbor=[]
                neighbor = tuple(map(operator.add,current[0] ,i)), tuple(map(operator.add,current[1], j))
                tentative_g_score = gscore[current] + self.h_1(current, neighbor)
                if 0 <= neighbor[0] < start.shape[0]:
                    if 0 <= neighbor[1] < start.shape[1]:
                        if start[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.h_1(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))

        return False

    def a_star_graph(self,problem : Problem) -> tuple:
        return "Not really the right format for Solution"


import time

p=Problem()
s=State()
n=Node(None,None, 0, s)
n2=Node(n,None, 0, s)

searches = Searches()

p.goalState=State(s)

p.apply('R',s)
p.apply('R',s)
p.apply('D',s)
p.apply('D',s)
p.apply('L',s)

p.initialState=State(s)

print(p.initialState)

si=State(s)
# change the number of random moves appropriately
# If you are curious see if you get a solution >30 moves. The 
apply_rnd_moves(15,si,p)
p.initialState=si

startTime=time.clock()


print('=== Bfs*  ===')
startTime=time.clock()
res=searches.graph_bfs(p)
print(res)
print(time.clock()-startTime)
print(Node.nodeCount)

print('=== id DFS*  ===')
startTime=time.clock()
res=searches.id_dfs(p)
print(res)
print(time.clock()-startTime)
print(Node.nodeCount)

print('\n\n=== A*-Tree ===\n')
startTime=time.clock()
res=searches.a_star_tree(p)
print(time.clock()-startTime)
print(Node.nodeCount)
print("result: ",res)

print('\n\n=== A*-Graph ===\n')
startTime=time.clock()
res=searches.a_star_graph(p)
print(time.clock()-startTime)
print(Node.nodeCount)
print(res)

#
#print('\n\n=== A*-G-SL  ===\n')
#startTime=time.clock()
#res=AstarGraph2(p)
#print(time.clock()-startTime)
#print(node.nodeCount)
#print(res)
#
#print('\n\n=== A*-G-HQ  ===\n')
#startTime=time.clock()
#res=AstarGraph3(p)
#print(time.clock()-startTime)
#print(node.nodeCount)
#print(res)
#
#print('=== Bfs*  ===')
#startTime=time.clock()
#res=bfsGraph(p)
#print(res)
#print(time.clock()-startTime)
#print(node.nodeCount)
#

'''
print('\n\n=== A* - Tree  ===\n')
startTime=time.clock()
res=Astar(p)
print(time.clock()-startTime)
print(node.nodeCount)
print(res)

print('\n\n=== A*-Tree-SL ===\n')
startTime=time.clock()
res=Astar2(p)
print(time.clock()-startTime)
print(node.nodeCount)
print(res)

'''

'''
print('=== iDFS*  ===')
startTime=time.clock()
res=iDL_DFS(p)
print(res)
print(time.clock()-startTime)
print(node.nodeCount)
'''

startTime=time.clock()

