import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()

        return self.backtrack(dict())


    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Filter all value using length predicate
        self.domains = {
            var: set(filter(lambda word: len(word) == var.length, self.domains[var]))
            for var in self.domains.keys()
        }


    def revise(self, x: Variable, y: Variable):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised: bool = False
        overlap: tuple[int, int] = self.crossword.overlaps[x, y]

        if overlap:
            for wordX in self.domains[x].copy():
                # If none of Y domains are compatible, remove the word from X
                if all(map(lambda wordY: wordX[overlap[0]] != wordY[overlap[1]], self.domains[y])):
                    self.domains[x].remove(wordX)
                    revised = True

        return revised


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue: list[tuple[Variable, Variable]] = arcs \
            if arcs is not None \
            else [(x, y) for x in self.crossword.variables for y in self.crossword.neighbors(x)]

        while queue:
            x, y = queue.pop()

            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.insert(0, (z, x))
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Assert that all variables are present as keys,
        # where each key will have their assigned domain string
        return self.crossword.variables == set(assignment.keys())


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check that variable assignments are distinct
        are_distinct = len(set(assignment.values())) == len(assignment)
        if not are_distinct:
            return False

        # Check that the assigned value is consistent with
        # variables unary constraint (word length)
        for var, val in assignment.items():
            if not var.length == len(val):
                return False

        # Check that there is no conflict with neighbors
        for var in assignment:
            for comp_var in self.crossword.neighbors(var):
                if comp_var not in assignment:
                    continue

                i, j = self.crossword.overlaps[var, comp_var]
                if assignment[var][i] != assignment[comp_var][j]:
                    return False

        return True


    def order_domain_values(self, var: Variable, assignment: dict[Variable, str]) -> list[str]:
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # get a list of unassigned neighbors
        unassigned_neighbors: list[Variable] = list(filter(
            lambda v: v not in assignment,
            self.crossword.neighbors(var)
        ))

        # maintain a rating for sorting later
        rating: dict[str, int] = {
            domain: 0
            for domain in self.domains[var]
        }

        # calculate number of rule outs for each domain scenario
        for domain in self.domains[var]:
            for neighbor in unassigned_neighbors:
                # Find where variables overlap, and check that point
                i, j = self.crossword.overlaps[var, neighbor]
                for neighbor_domain in self.domains[neighbor]:
                    if domain[i] != neighbor_domain[j]:
                        rating[domain] += 1

        return sorted(self.domains[var], key=lambda val: rating[val])


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables: list[Variable] = [
            var for var in self.crossword.variables
            if var not in assignment
        ]

        return min(
            unassigned_variables,
            key=lambda var: (
                len(self.domains[var]),  # remaining domains
                -len(self.crossword.neighbors(var))  # degree of variable
            )
        )

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            # Check adding that key: value, breaks consistency
            if self.consistent(assignment | {var: value}):
                assignment[var] = value

                domains_backup = {
                    v: self.domains[v].copy()
                    for v in self.crossword.variables
                }

                # use AC3 inference optimization
                inferences = self.ac3(
                    [(neighbor, var) for neighbor in self.crossword.neighbors(var)]
                )

                if inferences:
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result

                # Fallback scenario
                del assignment[var]
                self.domains = domains_backup

        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
