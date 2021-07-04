from tetris import Board, GameState, Tetrominoes


def test_I():
    board = Board()
    cp = Tetrominoes[0]
    cp.set_position(5, 1)
    gs = GameState(board, cp, None)
    assert gs.count_holes() == 0
    assert gs.cumulative_height() == 0
    assert gs.roughness() == 0
    assert gs.relative_height() == 0

    # can move left 2
    for i in range(2):
        success = gs.move_left()
        assert success

    success = gs.move_left()
    assert not success

    # can move right 6 times
    for i in range(6):
        success = gs.move_right()
        assert success

    success = gs.move_right()
    assert not success

    # can move down 19 times
    for i in range(19):
        success = gs.move_down()
        assert success

    # reset in the middle
    cp.set_position(5, 10)

    assert gs.rot_cw()
    assert gs.rot_ccw()
