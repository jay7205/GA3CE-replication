import numpy as np


# GA3CE: smpl+head_30 -> 15 selected keypoints
POSE_INDICES = np.array(
    [0, 4, 5, 6, 7, 8, 12, 18, 19, 20, 21, 24, 26, 28, 29],
    dtype=np.int64,
)

HEAD_INDICES = np.array(
    [24, 25, 26, 27, 28, 29],
    dtype=np.int64,
)


def compute_t_pose(pose_3d: np.ndarray) -> np.ndarray:
    """
    Compute t_pose as the mean 3D position of head keypoints 24-29.

    Parameters
    ----------
    pose_3d : np.ndarray
        30 MeTRAbs keypoints with shape (30, 3).

    Returns
    -------
    np.ndarray
        t_pose with shape (3,).
    """
    pose_3d = np.asarray(pose_3d, dtype=np.float64)

    if pose_3d.shape != (30, 3):
        raise ValueError(
            f"Expected pose_3d shape (30, 3), got {pose_3d.shape}"
        )

    return pose_3d[HEAD_INDICES].mean(axis=0)


def compute_scale(pose_3d: np.ndarray) -> float:
    """
    Compute GA3CE pose scale.

    s = 1 / ||P_24 - P_28||

    Parameters
    ----------
    pose_3d : np.ndarray
        30 MeTRAbs keypoints with shape (30, 3).

    Returns
    -------
    float
        Reciprocal head width.
    """
    pose_3d = np.asarray(pose_3d, dtype=np.float64)

    if pose_3d.shape != (30, 3):
        raise ValueError(
            f"Expected pose_3d shape (30, 3), got {pose_3d.shape}"
        )

    head_width = np.linalg.norm(
        pose_3d[24] - pose_3d[28]
    )

    if head_width <= 0.0:
        raise ValueError("Head width must be greater than zero.")

    return 1.0 / head_width


def normalize_pose(
    pose_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize a 30-keypoint MeTRAbs pose according to GA3CE.

    Returns
    -------
    pose_normalized : np.ndarray
        15 selected and normalized keypoints, shape (15, 3).

    t_pose : np.ndarray
        Head reference position, shape (3,).

    scale : float
        Reciprocal head width.
    """
    pose_3d = np.asarray(pose_3d, dtype=np.float64)

    t_pose = compute_t_pose(pose_3d)
    scale = compute_scale(pose_3d)

    pose_centered = pose_3d - t_pose
    pose_normalized_full = scale * pose_centered

    pose_normalized = pose_normalized_full[POSE_INDICES]

    return pose_normalized, t_pose, scale


def compute_cyclotorsion_rotation(
    view_direction: np.ndarray,
) -> np.ndarray:
    """
    Compute the GA3CE cyclotorsion rotation.

    The rotation aligns the view direction with the positive
    z-axis while minimizing roll.

    Parameters
    ----------
    view_direction : np.ndarray
        3D view direction [vx, vy, vz].

    Returns
    -------
    np.ndarray
        Rotation matrix R with shape (3, 3).
    """
    v = np.asarray(view_direction, dtype=np.float64)

    if v.shape != (3,):
        raise ValueError(
            f"Expected view_direction shape (3,), got {v.shape}"
        )

    norm = np.linalg.norm(v)

    if norm <= 0.0:
        raise ValueError(
            "view_direction must have non-zero magnitude."
        )

    v = v / norm

    vx, vy, vz = v

    # GA3CE cyclotorsion equations.
    phi = np.arctan2(-vx, vz)

    theta = np.arctan2(
        vy,
        vz * np.cos(phi) - vx * np.sin(phi),
    )

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    c_theta = np.cos(theta)
    s_theta = np.sin(theta)

    # First rotate around Y by phi.
    R_y = np.array(
        [
            [c_phi, 0.0, s_phi],
            [0.0, 1.0, 0.0],
            [-s_phi, 0.0, c_phi],
        ],
        dtype=np.float64,
    )

    # Then rotate around X by theta.
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_theta, -s_theta],
            [0.0, s_theta, c_theta],
        ],
        dtype=np.float64,
    )

    return R_x @ R_y