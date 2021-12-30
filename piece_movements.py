#!/usr/bin/env python
import cv2

from tetris import Board, GameState, Tetrominoes


def main():
    for t in Tetrominoes:
        print(f"Piece: {t.name}")
        t.set_position(6, 10)
        board = Board()
        gs = GameState()
        gs.update(board, t)
        gs.display()
        cv2.waitKey(0)

        print("Moving Left")
        while gs.move_left():
            gs.display()
            cv2.waitKey(0)

        print("Moving Right")
        while gs.move_right():
            gs.display()
            cv2.waitKey(0)

        gs.current_piece.set_position(6, 10)
        gs.display()
        cv2.waitKey(0)

        print("Rotating CW")
        for i in range(len(t.shapes) + 1):
            gs.rot_cw()
            gs.display()
            cv2.waitKey(0)

        print("Rotating CCW")
        for i in range(len(t.shapes) + 1):
            gs.rot_ccw()
            gs.display()
            cv2.waitKey(0)

        print("Moving Down")
        while gs.move_down_possible():
            gs.move_down()
            gs.display()
            cv2.waitKey(0)

        print("Next?")
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
