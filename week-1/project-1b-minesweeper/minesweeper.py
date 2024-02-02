import itertools
import random


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells: {(int, int)}, count: int):
        self.cells: set = set(cells)
        self.count: int = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self) -> {(int, int)}:
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return self.cells
        return set()

    def known_safes(self) -> {(int, int)}:
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell: (int, int)) -> None:
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell: (int, int)) -> None:
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height: int = height
        self.width: int = width

        # Keep track of which cells have been clicked on
        self.moves_made: set = set()

        # Keep track of cells known to be safe or mines
        self.mines: set = set()
        self.safes: set = set()

        # List of sentences about the game known to be true
        self.knowledge: [Sentence] = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def get_neighbors(self, cell: (int, int)) -> {(int, int)}:
        """
        Returns all valid neighbors of a cell
        """
        neighbors = set()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) == (0, 0):
                    continue

                neighbor = (cell[0] + i, cell[1] + j)

                i_in_bounds = 0 <= neighbor[0] < self.height
                j_in_bounds = 0 <= neighbor[1] < self.width

                if i_in_bounds and j_in_bounds:
                    neighbors.add(neighbor)

        return neighbors

    def remove_empty_sentences(self):
        new_knowledge: [Sentence] = []
        for sentence in self.knowledge:
            if len(sentence.cells) != 0:
                new_knowledge.append(sentence)

        self.knowledge = new_knowledge

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1. Record a cell as a move made
        self.moves_made.add(cell)

        # 2. Mark the cell as safe
        self.mark_safe(cell)

        # 3. Conclude a new sentence
        neighbors: {(int, int)} = self.get_neighbors(cell)

        undetermined_neighbors: {(int, int)} = neighbors - self.mines - self.safes
        undetermined_mines_count: int = count - len(neighbors.intersection(self.mines))

        new_sentence = Sentence(undetermined_neighbors, undetermined_mines_count)

        # 4. Mark additional cells as safe or mines if it can be concluded based on current sentence
        if new_sentence.count == 0:
            for safe in undetermined_neighbors:
                self.mark_safe(safe)
        elif new_sentence.count == len(undetermined_neighbors):
            for mine in undetermined_neighbors:
                self.mark_mine(mine)
        else:
            self.knowledge.append(new_sentence)

        self.remove_empty_sentences()

        # 5. Infer new knowledge sentences as much as possible
        while True:
            old_knowledge = self.knowledge.copy()
            self.infer_new_knowledge()
            if old_knowledge == self.knowledge:
                break

        self.remove_empty_sentences()

        # 6. Mark additional cells as safe or mines if it can be concluded based on newly infered knowledge
        for sentence in self.knowledge:
            for safe in tuple(sentence.known_safes()):
                self.mark_safe(safe)
            for mine in tuple(sentence.known_mines()):
                self.mark_mine(mine)

        self.remove_empty_sentences()

    def infer_new_knowledge(self) -> [Sentence]:
        """
        Infer new knowledge sentences
        """
        new_knowledge: [Sentence] = []
        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                if sentence1 != sentence2 and sentence1.cells.issubset(sentence2.cells):
                    new_cells: set = sentence2.cells - sentence1.cells
                    new_count: int = sentence2.count - sentence1.count
                    new_sentence = Sentence(new_cells, new_count)

                    if new_sentence not in self.knowledge:
                        new_knowledge.append(new_sentence)

        # add new knowledge to the knowledge base
        for sentence in new_knowledge:
            if sentence not in self.knowledge:
                self.knowledge.append(sentence)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        candidates: set = self.safes - self.moves_made

        if len(candidates) == 0:
            return None

        return random.choice(tuple(candidates))

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        all_possibles: set = set(itertools.product(range(self.height), range(self.width)))
        candidates: tuple = tuple(all_possibles - self.moves_made - self.mines)

        if len(candidates) == 0:
            return None

        return random.choice(candidates)
