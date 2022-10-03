from typing import Optional, Union
import random
import contextlib

class RandomState:
    def __init__(self, seed: Optional[int]):
        self.reset(seed)
    
    def reset(self, seed: Optional[int]):
        self.seed = seed
        if self.seed is not None:
            self.prev = random.getstate()
            random.seed(seed)
            self.seed = random.getstate()
    
    def freeze(self):
        if self.seed is not None:
            self.seed = random.getstate()
            random.setstate(self.prev)
    
    def unfreeze(self):
        if self.seed is not None:
            self.prev = random.getstate()
            random.setstate(self.seed)

@contextlib.contextmanager
def seed_context(seed: Union[Optional[int], RandomState]):
    random_state = seed
    if not isinstance(seed, RandomState):
        random_state = RandomState(seed)
    random_state.unfreeze()
    yield
    random_state.freeze()

def seed_generator(seed: Optional[int]):
    random_state = RandomState(seed)
    while True:
        random_state.unfreeze()
        seed = random.getrandbits(64)
        random_state.freeze()
        yield seed
