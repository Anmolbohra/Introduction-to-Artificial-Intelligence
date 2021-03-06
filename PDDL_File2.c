coffee-deliver.pddl                                                                                 100666  000000  000000         1270 13171352610  12220                                                                                                       ustar 00root                                                                                                                                                                                                                                                   (define 
   (problem fetch-coffee)
   (:domain coffee-robot)
   (:objects robotRoom livingRoom kitchen bedroom hall - room
	     d1 d2 d3 d4 - door
	     rob - robot
  )	

   (:init 

	(connects d1 robotRoom hall)
	(connects d1 hall robotRoom)
	(isClosed d1)
	(connects d2 hall bedroom)
	(connects d2 bedroom hall)
	(isClosed d2)
	(connects d3 hall livingRoom)
	(connects d3 livingRoom hall)
	(isClosed d3)
	(connects d4 livingRoom kitchen)
	(connects d4 kitchen livingRoom)
	(isClosed d4)


   ;; Rest of inital condition  
   ;;

   ) 
(:goal 
       (and (isClosed d1) (isClosed d2) (isClosed d3) (isClosed d4)
       ;; The goal state goes here
       )

))
                                                                                                                                                                                                                                                                                                                                        coffee-deliver.pddl.soln                                                                            100666  000000  000000            0 13171400042  13071                                                                                                       ustar 00root                                                                                                                                                                                                                                                   coffee-robot-domain.pddl                                                                            100666  000000  000000         3034 13171400042  13151                                                                                                       ustar 00root                                                                                                                                                                                                                                                   
(define (domain coffee-robot)
   (:requirements :typing)
   (:types room door robot location)
   (:predicates
		(adj ?r1 ?2 - room)
		(connects ?d - door ?r1 ?r2 - room)
		(isOpen ?d - door)
		(isClosed ?d - door)
		(coffeeInRoom ?r - room)
		(holdCoffee ?r - robot)
		(emptyHand ?rob - robot)
		(at ?rob - robot ?r - room)
		(where ?rob - robot ?p - location)
		(accessible ?rob - robot ?p1 ?p2 - location)
		)

;; Notice how the opening a door that connects two rooms
;; adds the adjacency predicates in the effect
(:action open
	:parameters (?d - door ?r1 ?r2 - room ?rob - robot)
	:precondition (and (isClosed ?d) (connects ?d ?r1 ?r2) (at ?rob ?r1) (emptyHand ?rob))
	:effect (and (isOpen ?d) (adj ?r1 ?r2) (adj ?r2 ?r1)  (not (isClosed ?d)))
)


(:action close
	:parameters (?d - door ?r1 ?r2 - room ?rob - robot)
	:precondition (and (isOpen ?d) (connects ?d ?r1 ?r2) (at ?rob ?r1) (emptyHand ?rob))
	:effect (and (isClosed ?d) (adj ?r1 ?r2) (adj ?r2 ?r1) (not (isOpen ?d))
)
)

(:action move
	:parameters (?rob - robot ?from ?to - location)
	:precondition (and (where ?rob ?from) (accessible ?rob ?from ?to))
	:effect (and (not (where ?rob ?from)) (at ?robot ?to) )
)

(:action grabCoffe
	:parameters (?rob - robot)
	:precondition (and (at ?rob ?r1) (coffeeInRoom ?r) (emptyHand ?rob))
	:effect (and(holdCoffee ?r) (not(emptyHand ?rob)))
)

(:action dropOffCoffee
	:parameters (?rob - robot )
	:precondition (and (at ?rob ?r1) (holdCoffee ?r))
	:effect (and (not(holdCoffee ?r)) (emptyHand ?rob))
)

)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    slide-domain.pddl                                                                                   100666  000000  000000          611 13166653212  11672                                                                                                       ustar 00root                                                                                                                                                                                                                                                   
(define (domain slide-domain)
   (:predicates (location ?l)
		(tile ?t)
		(adj ?l ?l)
		(empty ?l)
		(at ?t ?l))		
		
(:action move
       :parameters  (?t ?from ?to)
       :precondition (and (tile ?t) (location ?to) (location ?from) (adj ?from ?to) (empty ?to) (at ?t ?from) ) 
       :effect (and  (empty ?from) (at ?t ?to)
		     (not (empty ?to)) (not (at ?t ?from)) ) ))

                                                                                                                       slide-domain-grid.pddl                                                                              100666  000000  000000         2732 13171241432  12634                                                                                                       ustar 00root                                                                                                                                                                                                                                                   
(define (domain slide-domain-grid)
   (:requirements :typing)
   (:types idx tile)
   (:predicates	
		(empty ?r ?c - idx)
		(at ?t - tile ?r ?c - idx)
		(inc ?i  ?j - idx)
		(dec ?i  ?j - idx)
		)		
	
(:action Up
       :parameters  (?t - tile ?rowFrom - idx ?col - idx ?rowTo - idx)
       :precondition (and   (at ?t ?rowFrom ?col) (empty ?rowTo ?col) (dec ?rowFrom ?rowTo)) 
       :effect (and  (empty ?rowFrom ?col) (at ?t ?rowTo ?col)
		     (not (empty ?rowTo ?col)) (not (at ?t ?rowFrom ?col)) ) )

(:action Down
       :parameters  (?t - tile ?rowFrom - idx ?col - idx ?rowTo - idx)
       :precondition (and   (at ?t ?rowFrom ?col) (empty ?rowTo ?col) (inc ?rowFrom ?rowTo))
       :effect (and  (empty ?rowFrom ?col) (at ?t ?rowTo ?col)
		     (not (empty ?rowTo ?col)) (not (at ?t ?rowFrom ?col)) ) )

(:action Right
       :parameters  (?t - tile ?colFrom - idx ?row - idx ?colTo - idx)
       :precondition (and   (at ?t ?colFrom ?row) (empty ?colTo ?row) (inc ?colFrom ?colTo))
       :effect  (and    (empty ?colFrom ?row) (at ?t ?colTo ?row)
                (not    (empty ?colTo ?row)) (not   (at ?t ?colFrom ?row))))

(:action Left
       :parameters  (?t - tile ?colFrom - idx ?row - idx ?colTo - idx)
       :precondition (and   (at ?t ?colFrom ?row) (empty ?colTo ?row) (dec ?colFrom ?colTo))
       :effect  (and    (empty ?colFrom ?row) (at ?t ?colTo ?row)
                (not    (empty ?colTo ?row)) (not   (at ?t ?colFrom ?row))))
)

                                      slideproblem.py                                                                                     100666  000000  000000        13457 13170541742  11566                                                                                                       ustar 00root                                                                                                                                                                                                                                                   # -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:03:51 2017

@author: nnapp
"""

import numbers
import random

class State:
    """ State of sliding number puzzle
        Contains array of values called 'board' to indicate
        tile positions, and the position of tile '0', which
        indicates the empty space on the board.         """
    
    boardSize=3

    def __init__(self,s=None):

        if s == None:
            
            tiles=range(self.boardSize*self.boardSize).__iter__()
            self.board=[[next(tiles) for i in range(self.boardSize)] for j in range(self.boardSize)]

            #keep track of empty position
            self.position=[0,0]
        
        else:
            
            #copy the board

            self.board=[]
            for row in s.board:
                self.board.append(list(row))

            #copy the positions    
            self.position=list(s.position)
            
        
    def __str__(self):
        #won't work for larger boards 
        rstr=''
        for row in self.board:
            rstr += str(row) + '\n'
        return rstr
    
    #overload to allow comparison of lists and states with ==
    def __eq__(self,other):
        if isinstance(other, State):
            return self.board == other.board
        elif isinstance(other,list):
            return self.board == other
        else:
            return NotImplemented
 
    def __lt__(self,other):
        if isinstance(other, State):
            return self.h < other.h
        else:
            return NotImplemented
            
    def __cmp__(self,other):
        if isinstance(other, State):
            return cmp(self.h,other.h)
        else:
            return NotImplemented

    #turn into immutable ojbect for set lookup
    def toTuple(self):
        tpl=()
        for row in self.board:
            tpl += (tuple(row),)
        return tpl
    
    #create board from a list or tuple 
    def setBoard(self,brd):
        self.board=brd
        for row in range(self.boardSize):
            for col in range(self.boardSize):
                if self.board[row][col]==0:
                    self.position=[row,col]
                    return None
        raise StandardError('Set board configuration does not have an empy spot!')

#node class for serach graph        
class Node:
    
    nodeCount=0
    
    def __init__(self,parent=None, action=None, cost=0, state=None):
        
        #keep track of how many nodes were created
        self.__class__.nodeCount += 1    
        self.nodeID=self.nodeCount
        
        self.parent=parent
        self.cost=cost
        self.action=action
        self.state=state
        self.f=0
        
    #test equivalence Should be state

    def __str__(self):
        rstr= 'NodeID: ' + str(self.nodeID) + '\n'
        if self.parent != None:
            rstr+='Parent: ' + str(self.parent.nodeID) + '\n'
        if self.action != None:
            rstr+='Action: ' + self.action  + '\n'
        rstr+='Cost:   ' + str(self.cost) + '\n'
        rstr+='State:\n' + str(self.state)
        return rstr

    def __cmp__(self,other):
        if isinstance(other,Node):
            return cmp(self.f,other.f)
        else:
            return NotImplemented 

            
    def __lt__(self,other):
        if isinstance(other,Node):
            return self.f < other.f
        if issubclass(type(other),numbers.Real):
            return self.f < other
        else:
            return NotImplemented 

            
        
#problem
class Problem:
    """Class that defines a serach problem"""

    def __init__(self):
        self.actions=['U','L','D','R']
        self.initialState=0
        self.goalState=0

    def apply(self,a,s):

        #positions after move, still refers to s.position object
        post=s.position

        #make a copy
        pre=list(post)
        
        #compute post position
        if a == 'U':
            post[0]=max(pre[0]-1,0)
        elif a == 'L':            
            post[1]=max(pre[1]-1,0)
        elif a == 'D':
            post[0]=min(pre[0]+1,s.boardSize-1)
        elif a == 'R':
            post[1]=min(pre[1]+1,s.boardSize-1)
        else:
            print('Undefined action: ' + str(a))
            raise StandardError('Action not defined for this problem!')

        #store the old tile
        tile=s.board[pre[0]][pre[1]]
        
        s.board[pre[0]][pre[1]]=s.board[post[0]][post[1]]
        s.board[post[0]][post[1]]=tile      

 #       print pre, ' ', post,' ',s.board[pre[0]][pre[1]] , '<--', s.board[post[0]][post[1]]      

        return s
        
    def applicable(self,s):
        actionList=[]

        #check if actions are applicable
        #Not in top row
        if s.position[0]>0:
            actionList.append('U')

        #not in left most col
        if s.position[1]>0:
            actionList.append('L')

        #not in bottom row
        if s.position[0]<(s.boardSize-1):
            actionList.append('D')

        #not in right col
        if s.position[1]<(s.boardSize-1):
            actionList.append('R')

        return actionList

    def goalTest(self,s):
        return self.goalState==s    


        
def child_node(n , action, problem):
    return Node(n,action, n.cost + 1, problem.apply(action,State(n.state)))

        
def apply_rnd_moves(numMoves,s,p):
    for i in range(numMoves):
        p.apply(p.actions[random.randint(0,3)],s)
    
def solution(node):
    ''' Returns actionList, cost of the solution generated from the node'''

    actions=[]
    cost=node.cost

    while node.parent != None:
        actions.insert(0,node.action)
        node=node.parent    

    return actions, cost        
                                                                                                                                                                                                                 task3x3.pddl                                                                                        100666  000000  000000         4073 13170533172  10650                                                                                                       ustar 00root                                                                                                                                                                                                                                                   (define (problem 3x3test)
   (:domain slide-domain)
   (:objects loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9
	tile1 tile2 tile3 tile4 tile5 tile6 tile7 tile8)
   (:init (location loc1)
	(location loc2)
	(location loc3)
	(location loc4)
	(location loc5)
	(location loc6)
	(location loc7)
	(location loc8)
	(location loc9)

	(tile tile1)
	(tile tile2)
	(tile tile3)
	(tile tile4)
	(tile tile5)
	(tile tile6)
	(tile tile7)
	(tile tile8)
	

;;  Predicates for following board confiburaiton
;;  adj is not symmetric, so both need to be added 
;;
;;  loc1  | loc2 | loc3
;;  --------------------
;;  loc2  | loc5 | loc6
;;  -------------------
;;  loc7 | loc 8 | loc9


	(adj loc1 loc2)        
	(adj loc2 loc1)        

	(adj loc2 loc3)        
	(adj loc3 loc2)

	(adj loc1 loc4)        
	(adj loc4 loc1)

	(adj loc2 loc5)        
	(adj loc5 loc2)        
        
	(adj loc3 loc6)        
	(adj loc6 loc3)        

	(adj loc5 loc4)        
	(adj loc4 loc5)

    (adj loc5 loc6)
	(adj loc6 loc5)        

	(adj loc4 loc7)        
	(adj loc7 loc4)        

	(adj loc5 loc8)        
	(adj loc8 loc5)        

	(adj loc6 loc9)        
	(adj loc9 loc6)

	(adj loc7 loc8)        
	(adj loc8 loc7) 

	(adj loc8 loc9)        
	(adj loc9 loc8) 

	
	;; board after applying 100 random moves solution length=27 ... a difficult state
	;; [4, 7, 1]
	;; [0, 8, 6]
	;; [2, 5, 3]

;;	(at tile4 loc1)
;;	(at tile7 loc2)
;;	(at tile1 loc3)
;;	(empty loc4)
;;	(at tile8 loc5)
;;	(at tile6 loc6)
;;	(at tile2 loc7)
;;	(at tile5 loc8)
;;	(at tile3 loc9)

	;; board after applying 35 random moves solution length=15 ... a medium difficulty state
	;; [3, 1, 5]
	;; [6, 7, 2]
	;; [4, 0, 8]


	(at tile3 loc1)
	(at tile1 loc2)
	(at tile5 loc3)
	(at tile6 loc4)
	(at tile7 loc5)
	(at tile2 loc6)
	(at tile4 loc7)
	(empty loc8)
	(at tile8 loc9)


)
   (:goal (and 
	(empty loc1)
	(at tile1 loc2)
	(at tile2 loc3)
	(at tile3 loc4)
	(at tile4 loc5)
	(at tile5 loc6)
	(at tile6 loc7)
	(at tile7 loc8)
	(at tile8 loc9)
	
)))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     task3x3.pddl.soln                                                                                   100666  000000  000000          550 13171255614  11601                                                                                                       ustar 00root                                                                                                                                                                                                                                                   (move tile7 loc5 loc8)
(move tile2 loc6 loc5)
(move tile8 loc9 loc6)
(move tile7 loc8 loc9)
(move tile4 loc7 loc8)
(move tile6 loc4 loc7)
(move tile3 loc1 loc4)
(move tile1 loc2 loc1)
(move tile2 loc5 loc2)
(move tile4 loc8 loc5)
(move tile7 loc9 loc8)
(move tile8 loc6 loc9)
(move tile5 loc3 loc6)
(move tile2 loc2 loc3)
(move tile1 loc1 loc2)
                                                                                                                                                        task3x3-grid.pddl                                                                                   100666  000000  000000         1637 13171250102  11564                                                                                                       ustar 00root                                                                                                                                                                                                                                                   (define (problem 3x3test)
   (:domain slide-domain-grid)
   (:objects 1 2 3 - idx
	tile1 tile2 tile3 tile4 tile5 tile6 tile7 tile8 - tile)
   (:init 
        ;; inc and dec
        (inc 1 2) (inc 2 3) (dec 3 2) (dec 2 1)


	;; inc and dec predicates !!
	;; you need to define these operations in logic




	;; (SUBMIT THIS ONE) board after applying 35 random moves solution length=15 ... a medium difficulty state
	;; [3, 1, 5]
	;; [6, 7, 2]
	;; [4, 0, 8]

    (at tile4 3 1) (at tile7 2 2) (at tile1 1 2)
    (empty 3 2) (at tile8 3 3) (at tile6 2 1)
    (at tile2 2 3) (at tile5 1 3) (at tile3 1 1))

	;; board after applying 100 random moves solution as length=27 ... a difficult state
	;; [4, 7, 1]
	;; [0, 8, 6]
	;; [2, 5, 3]


   (:goal (and (empty 1 1) (at tile1 1 2) (at tile2 1 3)
    (at tile3 2 1) (at tile4 2 2) (at tile5 2 3)
    (at tile6 3 1) (at tile7 3 2) (at tile8 3 3) ))
)
                                                                                                 task3x3-grid.pddl.soln                                                                              100666  000000  000000          550 13171400152  12511                                                                                                       ustar 00root                                                                                                                                                                                                                                                   (move tile7 loc5 loc8)
(move tile2 loc6 loc5)
(move tile8 loc9 loc6)
(move tile7 loc8 loc9)
(move tile4 loc7 loc8)
(move tile6 loc4 loc7)
(move tile3 loc1 loc4)
(move tile1 loc2 loc1)
(move tile2 loc5 loc2)
(move tile4 loc8 loc5)
(move tile7 loc9 loc8)
(move tile8 loc6 loc9)
(move tile5 loc3 loc6)
(move tile2 loc2 loc3)
(move tile1 loc1 loc2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        