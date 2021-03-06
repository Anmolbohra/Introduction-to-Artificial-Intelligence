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
                                                                                                                                                                                                                                                                                                                                        coffee-deliver.pddl.soln                                                                            100666  000000  000000            0 13171400042  13071                                                                                                       ustar 00root                                                                                                                                                                                                                                                   coffee-robot-domain.pddl                                                                            100666  000000  000000         3653 13171376650  13201                                                                                                       ustar 00root                                                                                                                                                                                                                                                   
(define (domain coffee-robot)
   (:requirements :typing)
   (:types room door robot location battery-level)
   (:predicates
		(adj ?r1 ?r2 - room)
		(connects ?d - door ?r1 ?r2 - room)
		(isOpen ?d - door)
		(isClosed ?d - door)
		(isKitchen ?k - room)
		(isBedroom ?b - room)
		(coffeeInRoom ?r - room)
		(holdCoffee ?r - robot)
		(emptyHand ?rob - robot)
		(at ?rob - robot ?r - room)
		( ?rob - robot ?p - location)
		(battery ?rob - robot ?battery-level)
		(accessible ?rob - robot ?p1 ?p2 - location)
		(next ?f1 ?f2 - battery-level)
        (allDoorsOpen ?ado - door)
        (inRobotRoom ?rr - room)
		)

;; Notice how the opening a door that connects two rooms
;; adds the adjacency predicats in the effect
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
	:parameters (?rob - robot ?from ?to - location ?fbefore ?fafter - battery-level)
	:precondition (and (at ?rob ?from) (accessible ?rob ?from ?to) (battery ?rob ?fbefore) (next ?fbefore ?fafter))
	:effect (and (not (at ?rob ?from)) (at ?robot ?to) (not(battery ?rob ?fbefore)) (battery ?rob ?fafter))
)

(:action grabCoffe
	:parameters ( ?rob - robot)
	:precondition (and (emptyHand ?rob) (isKitchen ?k) (coffeeInRoom ?r) (allDoorsOpen ?ado))
	:effect (and (not (emptyHand ?rob)) (holdCoffee ?r))
)

(:action dropOffCoffee
	:parameters ( ?rob - robot )
	:precondition (and (isBedroom ?b) (holdCoffee ?r))
	:effect (and (not(holdCoffee ?r)) (emptyHand ?rob) (not(allDoorsOpen ?ado)) (inRobotRoom ?rr))
)

)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     