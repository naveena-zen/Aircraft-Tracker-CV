"""
tracker.py
----------
Lucas-Kanade Optical Flow Tracker using cv2.calcOpticalFlowPyrLK.

This module exposes:
  - `track_points`     : compute LK optical flow between two grey frames.
  - `draw_tracked_points` : draw circles on the frame at each tracked point.
"""

import cv2
import numpy as np


# Lucas-Kanade parameters (pyramidal implementation)
LK_PARAMS = dict(
    winSize=(70, 70),          # search window at each pyramid level
    maxLevel=2,                # number of pyramid levels
    criteria=(                 # stop iterating when either condition is met:
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,    # max iteration count
        0.03   # epsilon
    )
)


def track_points(old_gray: np.ndarray,
                 new_gray: np.ndarray,
                 points: np.ndarray) -> tuple:
    """
    Track `points` from `old_gray` to `new_gray` using Lucas-Kanade.

    Parameters
    ----------
    old_gray : Previous greyscale frame (H x W, uint8).
    new_gray : Current greyscale frame (H x W, uint8).
    points   : Points to track, shape (N, 1, 2), dtype float32.

    Returns
    -------
    good_new : Successfully tracked points in the new frame (M, 2) float32.
    good_old : Corresponding points in the old frame (M, 2) float32.
               M ≤ N (only points where status == 1 are returned).
    """
    # Calculate sparse optical flow
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        new_gray,
        points,
        None,
        **LK_PARAMS
    )

    if next_pts is None or status is None:
        # No points could be tracked
        return np.array([]), np.array([])

    # Keep only successfully tracked points (status == 1)
    good_new = next_pts[status == 1]
    good_old = points[status == 1]

    return good_new, good_old


def draw_tracked_points(frame: np.ndarray,
                        points: np.ndarray,
                        color: tuple = (0, 255, 0),
                        radius: int = 5,
                        thickness: int = 2) -> np.ndarray:
    """
    Draw circles / rings around each tracked point on `frame`.

    Parameters
    ----------
    frame     : BGR frame to draw on (modified in-place copy is returned).
    points    : Tracked points, shape (N, 2) float32.
    color     : Circle colour in BGR.
    radius    : Circle radius in pixels.
    thickness : Border thickness (-1 fills the circle).

    Returns
    -------
    frame : Annotated frame.
    """
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), radius, color, thickness)

    return frame