# Simple Maze Pathfinding for Kids!
# Let's help a robot find ALL possible paths through a maze

print("🤖 Welcome to the Maze Robot Adventure! 🤖")
print("=" * 50)

# Our maze - think of it like a grid of rooms
# 0 = empty room (robot can go here)
# 1 = wall (robot cannot go here)
maze = [
    [0, 0, 1, 0],
    [0, 1, 0, 0], 
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]

# Show the maze to kids
print("Here's our maze:")
print("0 = empty room, 1 = wall")
for row in maze:
    print(row)

# Where does our robot start and where should it go?
start_row = 0
start_col = 0
end_row = 3
end_col = 3

print(f"\n🤖 Robot starts at: row {start_row}, column {start_col}")
print(f"🎯 Robot wants to reach: row {end_row}, column {end_col}")

# Let's find ALL possible paths!
all_paths = []  # This will store all the paths we find

def find_all_paths(current_row, current_col, path_so_far):
    """
    This function helps our robot explore the maze
    It remembers where it's been and tries all possible moves
    """

    # Add current position to our path
    current_path = path_so_far + [('COORDINATE WHERE WE ARE RIGHT NOW', 'COORDINATE WHERE WE ARE RIGHT NOW')]

    # Did we reach the goal? Yahoo! 🎉
    if current_row == 'SOME END COORD' and current_col == 'SOME END COORDINATE':
        all_paths.append(current_path)
        return

    # Try moving in all 4 directions: up, down, left, right
    moves = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1)    # right
    ]

    for move_row, move_col in moves:
        new_row = current_row + #DORECTION WE WANT TO MOVE IN - ROWS
