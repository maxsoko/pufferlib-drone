import pufferlib.models


class Policy(pufferlib.models.Default):
    def __init__(self, env, hidden_size=256):
        super().__init__(env, hidden_size)


# Optional recurrent wrapper exposed via `rnn_name = Recurrent`.
Recurrent = pufferlib.models.LSTMWrapper
