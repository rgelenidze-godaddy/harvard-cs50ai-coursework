import sys
from maze import Maze

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python maze.py maze.txt")

    m = Maze(sys.argv[1])
    print("Maze:")
    m.print()
    print("Solving...")
    m.solve()
    print("States Explored:", m.num_explored)
    print("Solution:")
    m.print()
    m.output_image("maze.png", show_explored=True)

if __name__ == "__main__":
    main()