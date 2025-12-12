
import numpy as np

def angle_3pts(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # vectors BA and BC
    ba = a - b
    bc = c - b

    # tiny epsilon to avoid division by zero
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom

    # keep value in valid range for arccos
    cosang = np.clip(cosang, -1.0, 1.0)

    # convert to degrees
    ang = np.degrees(np.arccos(cosang))
    return float(ang)


class EMASmoother:


    def __init__(self, alpha=0.3):
        # alpha should be between 0 and 1
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.value = None

    def update(self, x):
        # if x is None, just return last stored value
        if x is None:
            return self.value
        if self.value is None:
            # initialize on first sample
            self.value = float(x)
        else:
            # standard EMA update
            self.value = float(self.alpha * x + (1.0 - self.alpha) * self.value)
        return self.value
