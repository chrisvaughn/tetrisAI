from tetris import Tetrominoes

position = 6, 10
zero_based = 3, 7


def test_i():
    piece = Tetrominoes[0]
    piece.set_position(position[0], position[1])
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])

    l, r = piece.possible_translations()
    assert (l, r) == (3, 3)

    piece.rot_cw()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])
    l, r = piece.possible_translations()
    assert (l, r) == (5, 4)

    piece.rot_ccw()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])
    l, r = piece.possible_translations()
    assert (l, r) == (3, 3)

    piece.move_down()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1] + 1)
    l, r = piece.possible_translations()
    assert (l, r) == (3, 3)

    piece.move_left()
    piece.move_left()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0] - 2, zero_based[1] + 1)
    l, r = piece.possible_translations()
    assert (l, r) == (1, 5)

    piece.set_detected_position(4, 5, 0)
    x, y = piece.zero_based_corner_xy
    assert x == 2
    assert y == 5

    piece.set_detected_position(5, 10, 1)
    x, y = piece.zero_based_corner_xy
    assert x == 5
    assert y == 8

    piece.move_left(2)
    x, y = piece.zero_based_corner_xy
    assert x == 3
    assert y == 8

    piece.move_right(2)
    x, y = piece.zero_based_corner_xy
    assert x == 5
    assert y == 8

    piece.move_down(2)
    x, y = piece.zero_based_corner_xy
    assert x == 5
    assert y == 10


def test_l():
    piece = Tetrominoes[1]
    piece.set_position(position[0], position[1])
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])
    l, r = piece.possible_translations()
    assert (l, r) == (4, 3)

    piece.rot_cw()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])
    l, r = piece.possible_translations()
    assert (l, r) == (4, 4)

    piece.rot_ccw()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])
    l, r = piece.possible_translations()
    assert (l, r) == (4, 3)

    piece.move_down()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1] + 1)
    l, r = piece.possible_translations()
    assert (l, r) == (4, 3)

    piece.move_left()
    piece.move_left()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0] - 2, zero_based[1] + 1)
    l, r = piece.possible_translations()
    assert (l, r) == (2, 5)

    piece.set_detected_position(4, 5, 0)
    x, y = piece.zero_based_corner_xy
    assert x == 2
    assert y == 4

    piece.set_detected_position(5, 10, 1)
    x, y = piece.zero_based_corner_xy
    assert x == 4
    assert y == 8

    piece.move_left(2)
    x, y = piece.zero_based_corner_xy
    assert x == 2
    assert y == 8

    piece.move_right(2)
    x, y = piece.zero_based_corner_xy
    assert x == 4
    assert y == 8

    piece.move_down(2)
    x, y = piece.zero_based_corner_xy
    assert x == 4
    assert y == 10


def test_j():
    piece = Tetrominoes[2]
    piece.set_position(position[0], position[1])
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])

    piece.rot_cw()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])

    piece.rot_ccw()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1])

    piece.move_down()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0], zero_based[1] + 1)

    piece.move_left()
    piece.move_left()
    x, y = piece.zero_based_corner_xy
    assert (x, y) == (zero_based[0] - 2, zero_based[1] + 1)

    piece.set_detected_position(4, 5, 0)
    x, y = piece.zero_based_corner_xy
    assert x == 3
    assert y == 4

    piece.set_detected_position(5, 10, 1)
    x, y = piece.zero_based_corner_xy
    assert x == 4
    assert y == 9

    piece.move_left(2)
    x, y = piece.zero_based_corner_xy
    assert x == 2
    assert y == 9

    piece.move_right(2)
    x, y = piece.zero_based_corner_xy
    assert x == 4
    assert y == 9

    piece.move_down(2)
    x, y = piece.zero_based_corner_xy
    assert x == 4
    assert y == 11
