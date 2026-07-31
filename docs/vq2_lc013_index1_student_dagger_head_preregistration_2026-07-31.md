# VQ2 LC013 index-1 student DAgger head — 2026-07-31

Fit exactly one Puffer residual head, public index 1, on the 215,813 admitted
LC012 labels collected from LC010S-owned plant states. Preserve the complete
LC010S recurrent encoder, MinGRU, action trunk, public progress embedding, and
all residual rows other than index 1 bit-exact.

Use a fixed 7/8 train and 1/8 held-agent split, seed `431130`, ten epochs,
learning rate `2e-4`, zero weight decay, gradient cap `1`, and 32,768-row
chunks. Retain the minimum held-action-MSE epoch. Require at least `1.02x`
held phase-1 action-MSE improvement, finite residual norm at most 512, exact
non-residual parameters, and exact preservation of every non-target head.

LC013 is one recurrent legal-observation Puffer actor. It sends no FlightSim
packets and can authorize only one fresh bounded teacher-free prefix screen;
it grants no shadow, live, or Submission authority.
