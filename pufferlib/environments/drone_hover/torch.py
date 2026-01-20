import torch
import pufferlib.models


class SquashedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, eps=1e-6):
        super().__init__(loc, scale)
        self.eps = eps

    def _logit(self, action):
        action = torch.clamp(action, self.eps, 1.0 - self.eps)
        return torch.log(action) - torch.log1p(-action)

    def sample(self, sample_shape=torch.Size()):
        z = super().rsample(sample_shape)
        return torch.sigmoid(z)

    def log_prob(self, action):
        z = self._logit(action)
        log_prob_z = super().log_prob(z)
        log_det = -torch.log(action) - torch.log1p(-action)
        return (log_prob_z + log_det).sum(dim=-1)

    def entropy(self):
        # Approximate entropy using base Normal entropy
        return super().entropy().sum(dim=-1)

    def sample_with_logprob(self, action=None):
        batch = self.loc.shape[0]
        if action is None:
            action = self.sample().view(batch, -1)
        else:
            action = action.view(batch, -1)
        log_prob = self.log_prob(action)
        entropy = self.entropy()
        return action, log_prob, entropy


class Policy(pufferlib.models.Default):
    def __init__(self, env, hidden_size=256):
        super().__init__(env, hidden_size)


class SquashedPolicy(pufferlib.models.Default):
    def __init__(self, env, hidden_size=256):
        super().__init__(env, hidden_size)
        if self.is_continuous:
            try:
                base_env = getattr(env, "env", env)
                if getattr(base_env, "pd_assist", False):
                    hover_ratio = 0.5
                elif getattr(base_env, "action_centered", False):
                    hover_ratio = 0.5
                else:
                    hover_ratio = (base_env.mass * base_env.gravity) / (4.0 * base_env.max_thrust_per_motor)
                hover_ratio = float(min(0.999, max(0.001, hover_ratio)))
                bias = torch.logit(torch.tensor(hover_ratio))
                with torch.no_grad():
                    if hasattr(self, "decoder_mean") and self.decoder_mean.bias is not None:
                        self.decoder_mean.bias.fill_(bias.item())
                    if hasattr(self, "decoder_logstd"):
                        self.decoder_logstd.fill_(-1.0)
            except Exception:
                pass

    def decode_actions(self, hidden):
        if not self.is_continuous:
            return super().decode_actions(hidden)

        mean = self.decoder_mean(hidden)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        logstd = self.decoder_logstd.expand_as(mean)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=2.0, neginf=-5.0)
        logstd = torch.clamp(logstd, -5.0, 2.0)
        std = torch.exp(logstd)
        logits = SquashedNormal(mean, std)
        values = self.value(hidden)
        return logits, values
