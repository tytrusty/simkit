from .SimState import SimState


class Sim():
    def __init__(self):
        pass


    def step(self, WorldState : SimState = SimState()):
        return WorldState

    def stepp(self):
        return

    def rollout(self, num_steps = 1):
        states = []
        state = self.step()
        states.append(state)
        return states