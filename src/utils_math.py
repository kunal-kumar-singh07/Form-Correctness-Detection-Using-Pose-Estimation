import numpy as np

# math helpers for angles and smoothing
def angle_3pts(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # vectors BA and BC
    ba = a - b
    bc = c - b

    # avoid division by zero
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom

    # clamp and convert to degrees
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(ang)


# simple exponential moving average smoother
class EMASmoother:
    def __init__(self, alpha=0.3):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.value = None

    def update(self, x):
        # return last value if input is None
        if x is None:
            return self.value
        if self.value is None:
            self.value = float(x)
        else:
            self.value = float(self.alpha * x + (1.0 - self.alpha) * self.value)
        return self.value
