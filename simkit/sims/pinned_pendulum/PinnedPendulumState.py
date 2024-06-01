import numpy as np
from ..State import State
from .PinnedPendulumFEMState import PinnedPendulumFEMState
from .PinnedPendulumMFEMState import PinnedPendulumMFEMState

class PinnedPendulumState(State):
    def __init__(self, state : np.ndarray | PinnedPendulumFEMState | PinnedPendulumMFEMState = PinnedPendulumFEMState()):

        if state is None:
            state = PinnedPendulumFEMState()

        if isinstance(state, np.ndarray):
            if len(state) == 4:
                state = PinnedPendulumMFEMState(p=state)
            elif len(state) == 2:
                state = PinnedPendulumFEMState(x=state)
            else:
                ValueError("state must be a 2 or 4 element array, or a PinnedPendulumFEMState or PinnedPendulumMFEMState object.")

        self.sub_state = state
        return

    def position(self):
        if isinstance(self.sub_state, PinnedPendulumFEMState):
            return self.sub_state.x
        elif isinstance(self.sub_state, PinnedPendulumMFEMState):
            return self.sub_state.p[:2]

    def length(self):
        if isinstance(self.sub_state, PinnedPendulumFEMState):
            return np.linalg.norm(self.sub_state.x)
        elif isinstance(self.sub_state, PinnedPendulumMFEMState):
            return self.sub_state.p[2]
