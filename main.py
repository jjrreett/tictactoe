from ast import Tuple
from typing import Any, Callable


def minimax(moves: list, depth: int, alpha: float, beta: float):
    if depth == 0 or is_terminal(moves):
        return evaluate(moves)

    maximizingPlayer = len(moves) % 2 == 0

    if maximizingPlayer:
        maxEval = float("-inf")
        for move in get_possible_moves(moves):
            eval = minimax(moves + [move], depth - 1, alpha, beta)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float("inf")
        for move in get_possible_moves(moves):
            eval = minimax(moves + [move], depth - 1, alpha, beta)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def moves_to_state(moves: list) -> tuple[int, int]:
    xs = 0b000_000_000
    os = 0b000_000_000
    state = [xs, os]
    for i, move in enumerate(moves):
        state[i % 2] |= 1 << move
    return tuple(state)


def is_terminal(moves: list) -> bool:
    winning_patterns = [
        0b111_000_000,
        0b000_111_000,
        0b000_000_111,
        0b100_100_100,
        0b010_010_010,
        0b001_001_001,
        0b100_010_001,
        0b001_010_100,
    ]

    xs, os = moves_to_state(moves)

    for pattern in winning_patterns:
        if xs & pattern == pattern:
            return True
        if os & pattern == pattern:
            return True

    if xs | os == 0b111_111_111:
        return True


def compute_bit_shifts(pattern_dims: tuple, target_dims: tuple) -> list:
    """
    Compute the bit shifts for a given pattern and target dimensions.
    Args:
        pattern_dims (tuple): The dimensions of the pattern (height, width).
        target_dims (tuple): The dimensions of the target (height, width).
    Returns:
        list: A list of bit shifts.
    Example:
        pattern_dims = (2, 2)
        target_dims = (4, 4)
        compute_bit_shifts(pattern_dims, target_dims)
        # Output: [0, 1, 4, 5, 8, 9, 12, 13]
    """

    pattern_height, pattern_width = pattern_dims
    target_height, target_width = target_dims

    bit_shifts = []

    for i in range(target_height - pattern_height + 1):
        for j in range(target_width - pattern_width + 1):
            shift = i * target_width + j
            bit_shifts.append(shift)

    return bit_shifts


def evaluate(moves: list) -> float:
    xs, os = moves_to_state(moves)

    # Winning patterns
    winning_patterns = [
        0b111_000_000,
        0b000_111_000,
        0b000_000_111,
        0b100_100_100,
        0b010_010_010,
        0b001_001_001,
        0b100_010_001,
        0b001_010_100,
    ]

    # Check for terminal state
    for pattern in winning_patterns:
        if xs & pattern == pattern:
            return 1.0  # X wins
        if os & pattern == pattern:
            return -1.0  # O wins

    # Draw
    if xs | os == 0b111_111_111:
        return 0.0

    # Evaluate intermediate board
    x_pairs = 0
    o_pairs = 0

    pairs = (0b000_000_011, (1, 2)), (0b000_010_001, (2, 2)), (0b000_001_001, (2, 1))

    for pattern, dims in pairs:
        for shift in compute_bit_shifts(dims, (3, 3)):
            p = pattern << shift
            if xs & p == p:
                x_pairs += 1
            if os & p == p:
                o_pairs += 1

    evaluation = x_pairs * 0.1 - o_pairs * 0.1
    return evaluation


def render(moves: list) -> str:
    xs, os = moves_to_state(moves)
    nums = "０１２３４５６７８９"
    board = ""

    # ANSI escape codes for colors
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    for i in range(8, -1, -1):
        if xs & (1 << i):
            board += f"{RED}Ｘ{RESET}"
        elif os & (1 << i):
            board += f"{BLUE}Ｏ{RESET}"
        else:
            board += nums[i]

        if i % 3 == 0:
            board += "\n"

    return board



def get_possible_moves(moves: list) -> list:
    xs, os = moves_to_state(moves)
    occupied = xs | os  # Cells occupied by either player
    possible_moves = []

    for i in range(9):
        if not (occupied & (1 << i)):
            possible_moves.append(i)

    return possible_moves


def test(fun: Callable, input: Tuple, expected: Any):
    output = fun(*input)
    assert (
        fun(*input) == expected
    ), f"Test failed! {fun.__name__}{input} -> {output} != {expected}"


# Example usage
if __name__ == "__main__":

    def test(func, args, expected):
        result = func(*args)
        assert result == expected, f"Expected {expected}, but got {result}"

    test(moves_to_state, ([0, 1, 4, 3, 8],), (0b100_010_001, 0b000_001_010))
    test(moves_to_state, ([0, 1, 4, 3, 8, 2, 6, 5, 7],), (0b111_010_001, 0b000_101_110))

    test(get_possible_moves, ([],), [0, 1, 2, 3, 4, 5, 6, 7, 8])
    test(get_possible_moves, ([0, 1, 4, 3, 8],), [2, 5, 6, 7])
    test(get_possible_moves, ([0, 1, 4, 3, 8, 2, 6, 5, 7],), [])

    test(compute_bit_shifts, ((2, 2), (3, 3)), [0, 1, 3, 4])

    test(evaluate, ([],), 0.0)
    test(evaluate, ([0, 1, 4, 3, 8],), 1.0)  # X wins
    test(evaluate, ([0, 1, 4, 3, 2, 5],), 0.0)  # Draw
    test(evaluate, ([0, 1, 4, 3, 8, 2, 6, 5, 7],), 1.0)  # X Wins
    test(evaluate, ([0, 1, 3],), 0.1)  # One pair for X
    test(evaluate, ([0, 1, 3, 4],), 0.0)  # One pair for O

    # print("All tests passed")

    depth = 9
    moves = []
    while not is_terminal(moves):
        print(render(moves))
        player_move = int(input("> "))
        moves.append(player_move)
        min_score = beta = float("inf")
        for move in get_possible_moves(moves):
            score = minimax(moves + [move], depth, float("-inf"), beta=float("inf"))
            min_score = min(min_score, score)
            if score == min_score:
                best_move = move
        moves.append(best_move)

    print(render(moves))
    result = evaluate(moves)
    if result == 1.0:
        print("X wins!")
    elif result == -1.0:
        print("O wins!")
    else:
        print("Draw!")

