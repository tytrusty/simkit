from ..sims.State import State


class Sim():
    def __init__(self):
        pass


    def step(self, world_state : State = State()):
        return world_state

    def stepp(self):
        return

    def rollout(self, num_steps = 1):
        states = []
        state = self.step()
        states.append(state)
        return states