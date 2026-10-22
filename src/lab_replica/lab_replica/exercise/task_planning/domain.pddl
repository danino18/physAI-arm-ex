; The order-assembly domain: the robot picks items up and puts them on the
; table or into the basket.
;
; The names below are fixed: the robot runs exactly these three actions, each
; with one item as its parameter (and place-in-slot, below), and the goal
; console accepts goals over exactly these four predicates. Write the
; preconditions and the effects.
;
;   (on-table ?i)          ?i stands on the table
;   (in-basket ?i)         ?i is in the basket
;   (robot-gripping ?i)    the gripper holds ?i
;   (robot-gripper-empty)  the gripper holds nothing
;
; The pyramid (exercise/pyramid.py: three cubes in a row, two on them, one on
; the top; the console's `pyramid:`) adds the type slot, one more action and
; these predicates. The three actions above and their predicates stay as they are.
;
;   (cube ?i)              ?i is a cube of the pyramid's size
;   (slot-free ?s)         nothing is in the pyramid's slot ?s
;   (in-slot ?i ?s)        ?i is in the slot ?s
;   (base ?s)              ?s is a slot on the table
;   (on-slots ?s ?a ?b)    a cube in ?s rests on the cubes in ?a and ?b
;
; The requirements are those of the goals (and, or, not, imply, forall,
; exists over items), so that UPF accepts any goal typed in the console.

(define (domain order-assembly-domain)
    (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions
                   :existential-preconditions :universal-preconditions)
    (:types item slot)
    (:predicates
        (on-table ?i - item)
        (in-basket ?i - item)
        (robot-gripping ?i - item)
        (robot-gripper-empty)
        (cube ?i - item)
        (slot-free ?s - slot)
        (in-slot ?i - item ?s - slot)
        (base ?s - slot)
        (on-slots ?s - slot ?a - slot ?b - slot)
    )

    (:action pick-up
        :parameters (?i - item)
        :precondition (and
            (on-table ?i)
            (robot-gripper-empty)
        )
        :effect (and
            (robot-gripping ?i)
            (not (robot-gripper-empty))
            (not (on-table ?i))
        )
    )

    (:action put-on-table
        :parameters (?i - item)
        :precondition (and
            (robot-gripping ?i)
        )
        :effect (and
            (on-table ?i)
            (robot-gripper-empty)
            (not (robot-gripping ?i))
        )
    )

    (:action put-in-basket
        :parameters (?i - item)
        :precondition (and
            (robot-gripping ?i)
        )
        :effect (and
            (in-basket ?i)
            (robot-gripper-empty)
            (not (robot-gripping ?i))
        )
    )

    ; A cube into a slot of the pyramid: a base slot, or one whose two
    ; supporting slots are taken already.
    (:action place-in-slot
        :parameters (?i - item ?s - slot)
        :precondition (and
            (robot-gripping ?i)
            (cube ?i)
            (slot-free ?s)
            (or
                (base ?s)
                (exists (?a - slot ?b - slot)
                    (and (on-slots ?s ?a ?b) (not (slot-free ?a)) (not (slot-free ?b)))
                )
            )
        )
        :effect (and
            (in-slot ?i ?s)
            (not (slot-free ?s))
            (robot-gripper-empty)
            (not (robot-gripping ?i))
        )
    )
)
