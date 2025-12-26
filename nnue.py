from enum import Enum
from typing import List, Iterable



class Piece(Enum):
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5

class Color(Enum):
    WHITE = 0
    BLACK = 1

def square(r: int, f: int) -> int:
    return (r << 3) + f


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def feature_index_for_perspective(perspective: Color, piece_color: Color, piece_type: Piece, sq: int) -> int:
    friendly = (piece_color == perspective)
    piece_idx = piece_type.value + (0 if friendly else 6)  # 0..11
    return 64 * piece_idx + sq  # 0..767



class LinearLayer:
    """
    weights[input_index][output_index], biases[output_index]
    Simple, explicit loops (easy to port).
    """
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # initialize to zeros (fill later with trained weights)
        self.weights: List[List[float]] = [[0.0] * output_size for _ in range(input_size)]
        self.biases: List[float] = [0.0] * output_size

    def forward(self, inputs: List[float]) -> List[float]:
        assert len(inputs) == self.input_size
        out = [self.biases[o] for o in range(self.output_size)]
        for i, x in enumerate(inputs):
            if x == 0.0:
                continue
            wcol = self.weights[i]
            # add contribution x * wcol to out
            for o in range(self.output_size):
                out[o] += wcol[o] * x
        return out


class Accumulator:
    """
    Two accumulators stored as values[2][output_dim].
    values[0] = perspective WHITE, values[1] = perspective BLACK
    active_sets keep which features are currently present for each perspective.
    """
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        self.values: List[List[float]] = [[0.0] * output_dim for _ in range(2)]
        # self.active: List[Set[int]] = [set(), set()]

    def __getitem__(self, color: Color) -> List[float]:
        return self.values[color.value]

    def refresh(self, layer: LinearLayer, features: Iterable[int], perspective: Color):
        """
        Recompute values[perspective] = b + sum_{f in features} H[:, f]
        features: iterable of feature indices (0..input_size-1)
        """
        p = perspective.value
        # start from bias
        for o in range(layer.output_size):
            self.values[p][o] = layer.biases[o]
        # add columns
        for f in features:
            col = layer.weights[f]
            for o in range(layer.output_size):
                self.values[p][o] += col[o]
        # store active set
        # self.active[p] = set(features)

    def update(self, layer: LinearLayer, removed: Iterable[int], added: Iterable[int], perspective: Color):
        """
        Incremental update: subtract columns for removed, add for added.
        Skips features not currently active / already active to be safe.
        """
        p = perspective.value
        removed_iter = removed or []
        added_iter = added or []

        for f in removed_iter:
            # if f not in self.active[p]:
            #     continue
            col = layer.weights[f]
            for o in range(layer.output_size):
                self.values[p][o] -= col[o]
            # self.active[p].remove(f)

        for f in added_iter:
            # if f in self.active[p]:
            #     continue
            col = layer.weights[f]
            for o in range(layer.output_size):
                self.values[p][o] += col[o]
            # self.active[p].add(f)


class NNUE:
    """
    Shared hidden layer H,b (LinearLayer), two accumulators, two output vectors and bias.
    Evaluate as: y = dot(O1, h_side) + dot(O2, h_other) + c
    Concatenation order: side-to-move first.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 32):
        self.hidden = LinearLayer(input_dim, hidden_dim)  # H, b
        self.acc = Accumulator(hidden_dim)
        self.O1: List[float] = [0.0] * hidden_dim
        self.O2: List[float] = [0.0] * hidden_dim
        self.c: float = 0.0

    def set_output_weights(self, O1: List[float], O2: List[float], c: float):
        assert len(O1) == len(self.O1) and len(O2) == len(self.O2)
        self.O1 = list(O1)
        self.O2 = list(O2)
        self.c = float(c)

    def refresh_both(self, features_side: Iterable[int], features_other: Iterable[int], side: Color):
        """
        Recompute accumulators for both perspectives. 'side' tells which perspective is side-to-move.
        features_side: features encoded from side-to-move perspective
        features_other: features encoded from opponent perspective
        """
        # perspective semantics: we always refresh WHITE slots with features computed for WHITE perspective
        # and BLACK slot with features computed for BLACK perspective.
        self.acc.refresh(self.hidden, features_side, side)
        other = Color.BLACK if side == Color.WHITE else Color.WHITE
        self.acc.refresh(self.hidden, features_other, other)

    def update_side(self, removed: Iterable[int], added: Iterable[int], side: Color):
        self.acc.update(self.hidden, removed, added, side)

    def evaluate(self, side: Color) -> float:
        """
        Evaluate by concatenating (side-to-move accumulator first).
        h_side = acc[side], h_other = acc[other]
        y = dot(O1, h_side) + dot(O2, h_other) + c
        """
        h_side = self.acc[side]
        other = Color.BLACK if side == Color.WHITE else Color.WHITE
        h_other = self.acc[other]

        return dot(self.O1, h_side) + dot(self.O2, h_other) + self.c


if __name__ == "__main__":
    # build a tiny deterministic example
    IN_DIM = 768
    H_DIM = 8

    nnue = NNUE(IN_DIM, H_DIM)

    # fill H and b with a few handcrafted nonzeros (rest are zero)
    # example: feature 10 contributes +1.0 to hidden 0, feature 20 contributes -2.0 to hidden 1
    nnue.hidden.weights[10][0] = 1.0
    nnue.hidden.weights[11][0] = 1.5

    nnue.hidden.biases[0] = 0.1
    nnue.hidden.biases[1] = -0.2

    # set simple output weights
    nnue.set_output_weights([1.0]*H_DIM, [-1.0]*H_DIM, c=0.0)
    
    assert feature_index_for_perspective(Color.WHITE, Color.WHITE, Piece.PAWN, 10) == feature_index_for_perspective(Color.BLACK, Color.BLACK, Piece.PAWN, 10)

    # imagine a board with two pieces (square indices 10 and 20) and side = WHITE
    # piece at square 10: friendly pawn (from WHITE perspective)
    feat_w_10 = feature_index_for_perspective(Color.WHITE, Color.WHITE, Piece.PAWN, 10)
    feat_w_11 = feature_index_for_perspective(Color.WHITE, Color.WHITE, Piece.PAWN, 11)
    # piece at square 20: enemy knight (from WHITE perspective)
    feat_w_20 = feature_index_for_perspective(Color.WHITE, Color.BLACK, Piece.KNIGHT, 20)

    # For opponent perspective (BLACK to move), same board features must be computed from black's perspective:
    feat_b_10 = feature_index_for_perspective(Color.BLACK, Color.WHITE, Piece.PAWN, 10)
    feat_b_11 = feature_index_for_perspective(Color.BLACK, Color.WHITE, Piece.PAWN, 11)
    feat_b_20 = feature_index_for_perspective(Color.BLACK, Color.BLACK, Piece.KNIGHT, 20)

    # refresh accumulators
    nnue.refresh_both([feat_w_10, feat_w_20], [feat_b_10, feat_b_20], side=Color.WHITE)

    # evaluate (WHITE to move)
    print("eval W:", nnue.evaluate(Color.WHITE))
    print("eval B:", nnue.evaluate(Color.BLACK))

    # do an incremental update: move piece from square 10 -> 11 for WHITE perspective
    removed = [feat_w_10]
    # added = [ feature_index_for_perspective(Color.WHITE, Color.WHITE, Piece.PAWN, 11) ]
    added = [feat_w_11]
    nnue.update_side(removed, added, Color.WHITE)

    # update opponent perspective similarly (if needed)
    removed_b = [feat_b_10]
    # added_b = [ feature_index_for_perspective(Color.BLACK, Color.WHITE, Piece.PAWN, 11) ]
    added_b = [feat_b_11]
    nnue.update_side(removed_b, added_b, Color.BLACK)

    print("eval after move:", nnue.evaluate(Color.WHITE))
    print("eval after move:", nnue.evaluate(Color.BLACK))
