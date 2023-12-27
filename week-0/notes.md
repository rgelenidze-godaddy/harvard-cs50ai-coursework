## [Lecture 0](https://cs50.harvard.edu/ai/2023/notes/0/#lecture-0)

## [Artificial Intelligence](https://cs50.harvard.edu/ai/2023/notes/0/#artificial-intelligence)

Artificial Intelligence (AI) covers a range of techniques that appear as sentient behavior by the computer. For example,
AI is used to recognize faces in photographs on your social media, beat the World’s Champion in chess, and process your
speech when you speak to Siri or Alexa on your phone.

In this course, we will explore some of the ideas that make AI possible:

0. **Search**

Finding a solution to a problem, like a navigator app that finds the best route from your origin to the destination, or
like playing a game and figuring out the next move.

1. **Knowledge**

Representing information and drawing inferences from it.

2. **Uncertainty**

Dealing with uncertain events using probability.

3. **Optimization**

Finding not only a correct way to solve a problem, but a better—or the best—way to solve it.

4. **Learning**

Improving performance based on access to data and experience. For example, your email is able to distinguish spam from
non-spam mail based on past experience.

5. **Neural Networks**

A program structure inspired by the human brain that is able to perform tasks effectively.

6. **Language**

Processing natural language, which is produced and understood by humans.

## [Search](https://cs50.harvard.edu/ai/2023/notes/0/#search)

Search problems involve an agent that is given an initial state and a goal state, and it returns a solution of how to
get from the former to the latter. A navigator app uses a typical search process, where the agent (the thinking part of
the program) receives as input your current location and your desired destination, and, based on a search algorithm,
returns a suggested path. However, there are many other forms of search problems, like puzzles or mazes.

![15 puzzle](https://cs50.harvard.edu/ai/2023/notes/0/15puzzle.png)

Finding a solution to a 15 puzzle would require the use of a search algorithm.

- **Agent**

  An entity that perceives its environment and acts upon that environment. In a navigator app, for example, the agent
  would be a representation of a car that needs to decide on which actions to take to arrive at the destination.

- **State**

  A configuration of an agent in its environment. For example, in
  a [15 puzzle](https://en.wikipedia.org/wiki/15_puzzle), a state is any one way that all the numbers are arranged on
  the board.

    - **Initial State**

      The state from which the search algorithm starts. In a navigator app, that would be the current location.

- **Actions**

  Choices that can be made in a state. More precisely, actions can be defined as a function. Upon receiving state `s` as
  input, `Actions(s)` returns as output the set of actions that can be executed in state `s`. For example, in a _15
  puzzle_, the actions of a given state are the ways you can slide squares in the current configuration (4 if the empty
  square is in the middle, 3 if next to a side, 2 if in the corner).

- **Transition Model**

  A description of what state results from performing any applicable action in any state. More precisely, the transition
  model can be defined as a function. Upon receiving state `s` and action `a` as input, `Results(s, a)` returns the
  state resulting from performing action `a` in state `s`. For example, given a certain configuration of a _15 puzzle_ (
  state `s`), moving a square in any direction (action `a`) will bring to a new configuration of the puzzle (the new
  state).

- **State Space**

  The set of all states reachable from the initial state by any sequence of actions. For example, in a 15 puzzle, the
  state space consists of all the 16!/2 configurations on the board that can be reached from any initial state. The
  state space can be visualized as a directed graph with states, represented as nodes, and actions, represented as
  arrows between nodes.

![State Space](https://cs50.harvard.edu/ai/2023/notes/0/statespace.png)

- **Goal Test**

  The condition that determines whether a given state is a goal state. For example, in a navigator app, the goal test
  would be whether the current location of the agent (the representation of the car) is at the destination. If it is —
  problem solved. If it’s not — we continue searching.

- **Path Cost**

  A numerical cost associated with a given path. For example, a navigator app does not simply bring you to your goal; it
  does so while minimizing the path cost, finding the fastest way possible for you to get to your goal state.

## [Solving Search Problems](https://cs50.harvard.edu/ai/2023/notes/0/#solving-search-problems)

- **Solution**

  A sequence of actions that leads from the initial state to the goal state.

    - **Optimal Solution**

      A solution that has the lowest path cost among all solutions.

In a search process, data is often stored in a **_node_**, a data structure that contains the following data:

- A _state_
- Its _parent node_, through which the current node was generated
- The _action_ that was applied to the state of the parent to get to the current node
- The _path cost_ from the initial state to this node

_Nodes_ contain information that makes them very useful for the purposes of search algorithms. They contain a _state_,
which can be checked using the _goal test_ to see if it is the final state. If it is, the node’s _path cost_ can be
compared to other nodes’ _path costs_, which allows choosing the _optimal solution_. Once the node is chosen, by virtue
of storing the _parent node_ and the _action_ that led from the _parent_ to the current node, it is possible to trace
back every step of the way from the _initial state_ to this node, and this sequence of actions is the _solution_.

However, _nodes_ are simply a data structure — they don’t search, they hold information. To actually search, we use the
**frontier**, the mechanism that “manages” the _nodes_. The _frontier_ starts by containing an initial state and an
empty set of explored items, and then repeats the following actions until a solution is reached:

Repeat:

1. If the frontier is empty,

    - _Stop._ There is no solution to the problem.
2. Remove a node from the frontier. This is the node that will be considered.

3. If the node contains the goal state,

    - Return the solution. _Stop_.

   Else,

   ```
   * Expand the node (find all the new nodes that could be reached from this node), and add resulting nodes to the frontier.
   * Add the current node to the explored set.
   ```

#### [Depth-First Search](https://cs50.harvard.edu/ai/2023/notes/0/#depth-first-search)

In the previous description of the _frontier_, one thing went unmentioned. At stage 1 in the pseudocode above, which
node should be removed? This choice has implications on the quality of the solution and how fast it is achieved. There
are multiple ways to go about the question of which nodes should be considered first, two of which can be represented by
the data structures of **stack** (in _depth-first_ search) and **queue** (in _breadth-first search_;
and [here is a cute cartoon demonstration](https://www.youtube.com/watch?v=2wM6_PuBIxY) of the difference between the
two).

We start with the _depth-first_ search (_DFS_) approach.

A _depth-first_ search algorithm exhausts each one direction before trying another direction. In these cases, the
frontier is managed as a _stack_ data structure. The catchphrase you need to remember here is “_last-in first-out_.”
After nodes are being added to the frontier, the first node to remove and consider is the last one to be added. This
results in a search algorithm that goes as deep as possible in the first direction that gets in its way while leaving
all other directions for later.

(An example from outside lecture: Take a situation where you are looking for your keys. In a _depth-first_ search
approach, if you choose to start with searching in your pants, you’d first go through every single pocket, emptying each
pocket and going through the contents carefully. You will stop searching in your pants and start searching elsewhere
only once you will have completely exhausted the search in every single pocket of your pants.)

- Pros:
    - At best, this algorithm is the fastest. If it “lucks out” and always chooses the right path to the solution (by
      chance), then _depth-first_ search takes the least possible time to get to a solution.
- Cons:
    - It is possible that the found solution is not optimal.
    - At worst, this algorithm will explore every possible path before finding the solution, thus taking the longest
      possible time before reaching the solution.

Code example:

```python
# Define the function that removes a node from the frontier and returns it.
def remove(self):
      # Terminate the search if the frontier is empty, because this means that there is no solution.
    if self.empty():
        raise Exception("empty frontier")
    else:
          # Save the last item in the list (which is the newest node added)
        node = self.frontier[-1]
        # Save all the items on the list besides the last node (i.e. removing the last node)
        self.frontier = self.frontier[:-1]
        return node
```

#### [Breadth-First Search](https://cs50.harvard.edu/ai/2023/notes/0/#breadth-first-search)

The opposite of _depth-first_ search would be _breadth-first_ search (_BFS_).

A _breadth-first_ search algorithm will follow multiple directions at the same time, taking one step in each possible
direction before taking the second step in each direction. In this case, the frontier is managed as a _queue_ data
structure. The catchphrase you need to remember here is “_first-in first-out_.” In this case, all the new nodes add up
in line, and nodes are being considered based on which one was added first (first come first served!). This results in a
search algorithm that takes one step in each possible direction before taking a second step in any one direction.

(An example from outside lecture: suppose you are in a situation where you are looking for your keys. In this case, if
you start with your pants, you will look in your right pocket. After this, instead of looking at your left pocket, you
will take a look in one drawer. Then on the table. And so on, in every location you can think of. Only after you will
have exhausted all the locations will you go back to your pants and search in the next pocket.)

- Pros:
    - This algorithm is guaranteed to find the optimal solution.
- Cons:
    - This algorithm is almost guaranteed to take longer than the minimal time to run.
    - At worst, this algorithm takes the longest possible time to run.

Code example:

```python
# Define the function that removes a node from the frontier and returns it.
def remove(self):
    # Terminate the search if the frontier is empty, because this means that there is no solution.
    if self.empty():
        raise Exception("empty frontier")
    else:
        # Save the oldest item on the list (which was the first one to be added)
        node = self.frontier[0]
        # Save all the items on the list besides the first one (i.e. removing the first node)
        self.frontier = self.frontier[1:]
        return node
```

#### [Greedy Best-First Search](https://cs50.harvard.edu/ai/2023/notes/0/#greedy-best-first-search)

Breadth-first and depth-first are both **uninformed** search algorithms. That is, these algorithms do not utilize any
knowledge about the problem that they did not acquire through their own exploration. However, most often is the case
that some knowledge about the problem is, in fact, available. For example, when a human maze-solver enters a junction,
the human can see which way goes in the general direction of the solution and which way does not. AI can do the same. A
type of algorithm that considers additional knowledge to try to improve its performance is called an **informed** search
algorithm.

**Greedy best-first** search expands the node that is the closest to the goal, as determined by a **heuristic function**
_h(n)_. As its name suggests, the function estimates how close to the goal the next node is, but it can be mistaken. The
efficiency of the _greedy best-first_ algorithm depends on how good the heuristic function is. For example, in a maze,
an algorithm can use a heuristic function that relies on the **Manhattan distance** between the possible nodes and the
end of the maze. The _Manhattan distance_ ignores walls and counts how many steps up, down, or to the sides it would
take to get from one location to the goal location. This is an easy estimation that can be derived based on the (x, y)
coordinates of the current location and the goal location.

![Manhattan Distance](https://cs50.harvard.edu/ai/2023/notes/0/manhattandistance.png)

Manhattan Distance

However, it is important to emphasize that, as with any heuristic, it can go wrong and lead the algorithm down a slower
path than it would have gone otherwise. It is possible that an _uninformed_ search algorithm will provide a better
solution faster, but it is less likely to do so than an _informed_ algorithm.

#### [A\* Search](https://cs50.harvard.edu/ai/2023/notes/0/#a-search)

A development of the _greedy best-first_ algorithm, _A\* search_ considers not only _h(n)_, the estimated cost from the
current location to the goal, but also _g(n)_, the cost that was accrued until the current location. By combining both
these values, the algorithm has a more accurate way of determining the cost of the solution and optimizing its choices
on the go. The algorithm keeps track of (_cost of path until now_ + _estimated cost to the goal_), and once it exceeds
the estimated cost of some previous option, the algorithm will ditch the current path and go back to the previous
option, thus preventing itself from going down a long, inefficient path that _h(n)_ erroneously marked as best.

Yet again, since this algorithm, too, relies on a heuristic, it is as good as the heuristic that it employs. It is
possible that in some situations it will be less efficient than _greedy best-first_ search or even the _uninformed_
algorithms. For _A\* search_ to be optimal, the heuristic function, _h(n)_, should be:

1. _Admissible_, or never _overestimating_ the true cost, and
2. _Consistent_, which means that the estimated path cost to the goal of a new node in addition to the cost of
   transitioning to it from the previous node is greater or equal to the estimated path cost to the goal of the previous
   node. To put it in an equation form, _h(n)_ is consistent if for every node _n_ and successor node _n’_ with step
   cost _c_, _h(n) ≤ h(n’) + c_.

### [Adversarial Search](https://cs50.harvard.edu/ai/2023/notes/0/#adversarial-search)

Whereas, previously, we have discussed algorithms that need to find an answer to a question, in **adversarial search**
the algorithm faces an opponent that tries to achieve the opposite goal. Often, AI that uses adversarial search is
encountered in games, such as tic tac toe.

#### [Minimax](https://cs50.harvard.edu/ai/2023/notes/0/#minimax)

A type of algorithm in adversarial search, **Minimax** represents winning conditions as (-1) for one side and (+1) for
the other side. Further actions will be driven by these conditions, with the minimizing side trying to get the lowest
score, and the maximizer trying to get the highest score.

**Representing a Tic-Tac-Toe AI**:

- _S₀_: Initial state (in our case, an empty 3X3 board)
- _Players(s)_: a function that, given a state _s_, returns which player’s turn it is (X or O).
- _Actions(s)_: a function that, given a state _s_, return all the legal moves in this state (what spots are free on the
  board).
- _Result(s, a)_: a function that, given a state _s_ and action _a_, returns a new state. This is the board that
  resulted from performing the action _a_ on state _s_ (making a move in the game).
- _Terminal(s)_: a function that, given a state _s_, checks whether this is the last step in the game, i.e. if someone
  won or there is a tie. Returns _True_ if the game has ended, _False_ otherwise.
- _Utility(s)_: a function that, given a terminal state _s_, returns the utility value of the state: -1, 0, or 1.

**How the algorithm works**:

Recursively, the algorithm simulates all possible games that can take place beginning at the current state and until a
terminal state is reached. Each terminal state is valued as either (-1), 0, or (+1).

![Minimax in Tic Tac Toe](https://cs50.harvard.edu/ai/2023/notes/0/minimax_tictactoe.png)

Minimax Algorithm in Tic Tac Toe

Knowing based on the state whose turn it is, the algorithm can know whether the current player, when playing optimally,
will pick the action that leads to a state with a lower or a higher value. This way, alternating between minimizing and
maximizing, the algorithm creates values for the state that would result from each possible action. To give a more
concrete example, we can imagine that the maximizing player asks at every turn: “if I take this action, a new state will
result. If the minimizing player plays optimally, what action can that player take to bring to the lowest value?”
However, to answer this question, the maximizing player has to ask: “To know what the minimizing player will do, I need
to simulate the same process in the minimizer’s mind: the minimizing player will try to ask: ‘if I take this action,
what action can the maximizing player take to bring to the highest value?’” This is a recursive process, and it could be
hard to wrap your head around it; looking at the pseudo code below can help. Eventually, through this recursive
reasoning process, the maximizing player generates values for each state that could result from all the possible actions
at the current state. After having these values, the maximizing player chooses the highest one.

![Minimax Algorithm](https://cs50.harvard.edu/ai/2023/notes/0/minimax_theoretical.png)

The Maximizer Considers the Possible Values of Future States.

To put it in pseudocode, the Minimax algorithm works the following way:

- Given a state _s_

    - The maximizing player picks action _a_ in _Actions(s)_ that produces the highest value of _Min-Value(Result(s,
      a))_.
    - The minimizing player picks action _a_ in _Actions(s)_ that produces the lowest value of _Max-Value(Result(s,
      a))_.
- Function _Max-Value(state)_

    - _v = -∞_

    - if _Terminal(state)_:

      return _Utility(state)_

    - for _action_ in _Actions(state)_:

      _v = Max(v, Min-Value(Result(state, action)))_

      return _v_

- Function _Min-Value(state)_:

    - _v = ∞_

    - if _Terminal(state)_:

      return _Utility(state)_

    - for _action_ in _Actions(state)_:

      _v = Min(v, Max-Value(Result(state, action)))_

      return _v_

#### [Alpha-Beta Pruning](https://cs50.harvard.edu/ai/2023/notes/0/#alpha-beta-pruning)

A way to optimize _Minimax_, **Alpha-Beta Pruning** skips some of the recursive computations that are decidedly
unfavorable. After establishing the value of one action, if there is initial evidence that the following action can
bring the opponent to get to a better score than the already established action, there is no need to further investigate
this action because it will decidedly be less favorable than the previously established one.

This is most easily shown with an example: a maximizing player knows that, at the next step, the minimizing player will
try to achieve the lowest score. Suppose the maximizing player has three possible actions, and the first one is valued
at 4. Then the player starts generating the value for the next action. To do this, the player generates the values of
the minimizer’s actions if the current player makes this action, knowing that the minimizer will choose the lowest one.
However, before finishing the computation for all the possible actions of the minimizer, the player sees that one of the
options has a value of three. This means that there is no reason to keep on exploring the other possible actions for the
minimizing player. The value of the not-yet-valued action doesn’t matter, be it 10 or (-10). If the value is 10, the
minimizer will choose the lowest option, 3, which is already worse than the preestablished 4. If the not-yet-valued
action would turn out to be (-10), the minimizer will this option, (-10), which is even more unfavorable to the
maximizer. Therefore, computing additional pomoz-extension:
//38938761-f08e-4a49-bf2c-01fffa3b2e13/popup/popup.html#ssible actions for the minimizer at this point is irrelevant to
the maximizer, because the maximizing player already has an unequivocally better choice whose value is 4.

![Alpha Beta Pruning](https://cs50.harvard.edu/ai/2023/notes/0/alphabeta.png)

#### [Depth-Limited Minimax](https://cs50.harvard.edu/ai/2023/notes/0/#depth-limited-minimax)

There is a total of 255,168 possible Tic Tac Toe games, and 10²⁹⁰⁰⁰ possible games in Chess. The minimax algorithm, as
presented so far, requires generating all hypothetical games from a certain point to the terminal condition. While
computing all the Tic-Tac-Toe games doesn’t pose a challenge for a modern computer, doing so with chess is currently
impossible.

**Depth-limited Minimax** considers only a pre-defined number of moves before it stops, without ever getting to a
terminal state. However, this doesn’t allow for getting a precise value for each action, since the end of the
hypothetical games has not been reached. To deal with this problem, _Depth-limited Minimax_ relies on an **evaluation
function** that estimates the expected utility of the game from a given state, or, in other words, assigns values to
states. For example, in a chess game, a utility function would take as input a current configuration of the board, try
to assess its expected utility (based on what pieces each player has and their locations on the board), and then return
a positive or a negative value that represents how favorable the board is for one player versus the other. These values
can be used to decide on the right action, and the better the evaluation function, the better the Minimax algorithm that
relies on it.