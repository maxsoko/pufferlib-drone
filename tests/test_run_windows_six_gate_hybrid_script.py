from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_six_gate_hybrid.ps1"


def test_shadow_is_default_and_has_zero_control_lifecycle() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '[string]$Mode = "Shadow"' in source
    assert '"--policy-shadow-only"' in source
    assert '"--no-arm-on-start"' in source
    assert '"--policy-gate-phase-onehot-adapter-observation"' in source
    assert '"--policy-hybrid-prefix-confidence-observation"' in source
    assert '"--policy-race-phase-denominator", "6"' in source
    shadow_block = source.split('if ($Mode -eq "Shadow") {', 1)[1].split(
        "$TargetGateCount =", 1
    )[0]
    assert '"--send-sim-reset"' not in shadow_block
    assert '"--official-reset-on-start"' not in shadow_block
    assert "run_windows_full_policy.ps1" not in shadow_block


def test_hybrid_artifacts_are_hash_pinned_and_flight_modes_explicit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '[ValidateSet("Shadow", "BoundedGate3", "BoundedGate4", "FullLap")]' in source
    assert '[string]$Tag = "n184_six_gate_hybrid"' in source
    assert source.count("Assert-SHA256 $") == 13
    assert "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760" in source
    assert "9feb33df3ec6a023c86819bb4911cb051b1a096a7d4b95fd18fa74f5989efbca" in source
    assert "e862206308f52eff67da16e90803d1884ee0a5c9092ad18edc9829da4c8aac30" in source
    assert "4592bda5801748301f00de3f0723b2dc9ca58a7f50f1c29188871e1cebceb354" in source
    assert "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514" in source
    assert "9f63f47ee3132c76dfb0a987fb97e9d697a50ce990cdb39a6df3ff69a9bbb883" in source
    assert "fbbf16d8fc61610f180e914658dac3fd504ef856e4a456b2ba64e9dc394097bd" in source
    assert "ce0bdde0ec4d2cb3b46df0ee35d1ede3f805391899fa24ba19b2df756d42b267" in source
    assert "70292925b0748e959eb654febf994afd623143b319bb241ec2e93312ef9e24fb" in source
    assert "f301b682edc10efffc3c131910ea31a479cd323beb888c1d3ddccfd1f13bd73a" in source
    assert "3750e1620f4a7d244b93c370f31ec135a89572e64a60888d7ea032c244949fae" in source
    assert "9da6cde4ed3c5a25f2c16872427ab6aab15ed57e379399ab725fee28b34869f2" in source
    assert "a68c07a2a55f61fc1b9ad6522a27e3f6b0ba206dab6dd554392f8c5da417a6a7" in source
    assert "46a075eafd49a1e44a14e32af458a26e8d411e4cb507b510c6aaaa7ee6b97b4a" in source
    assert "e6f2d717138456abba5b9a95905f91144541a8a6bd939a3155fc95881dfdb3b3" in source
    assert '$env:PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX = "3"' in source
    assert '$env:PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX = "3"' in source
    assert "d9e97ff6ac38c7a476eebc5ddfa5fd6f47d705c11f613922c17e9690f9031f7d" in source
    assert "HybridPrefixCheckpoint = $PrefixCheckpoint" in source
    assert '$TargetGateCount = if ($Mode -eq "BoundedGate3") {' in source
    assert '$StopAfterOfficialGateIndex = if ($Mode -eq "BoundedGate3") {' in source
    assert "HybridPolicyCallable = $PolicyCallableSpec" in source
    assert "PolicyLayoutPrecisionBytes = 4" in source
    assert "SixGateComposite = $true" in source
    assert "CommandHz = $CommandHz" in source
    assert "PolicyStateHz = $PolicyStateHz" in source
    assert '"--policy-state-hz", "$PolicyStateHz"' in source
    assert "[switch]$Gate3SevereAcquisition" in source
    assert "[switch]$Gate3CompositeAcquisitionPD" in source
    assert "policy_callable_gate3_composite_acquisition_pd.py:infer" in source
    assert "[switch]$Gate2VerticalMarginN189" in source
    assert "policy_callable_gate2_vertical_margin_n189.py:infer" in source
    assert "[switch]$Gate2SevereVerticalN190" in source
    assert "policy_callable_gate2_severe_vertical_n190.py:infer" in source
    assert "[switch]$Gate3TerminalLevelN191" in source
    assert "policy_callable_gate3_terminal_level_n191.py:infer" in source
    assert "[switch]$Gate2RapidVerticalDeficitN192" in source
    assert "policy_callable_gate2_rapid_vertical_deficit_n192.py:infer" in source
    assert "[switch]$Gate3FrozenObservationRecoveryN193" in source
    assert "policy_callable_gate3_frozen_observation_recovery_n193.py:infer" in source
    assert "[switch]$Gate2FrozenObservationDropoutN194" in source
    assert "policy_callable_gate2_frozen_observation_dropout_n194.py:infer" in source
    assert "[switch]$Gate3LaterTerminalLevelN195" in source
    assert "policy_callable_gate3_later_terminal_level_n195.py:infer" in source
    assert "3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df" in source
    assert "[switch]$Gate3ZeroConfidenceRecoveryN196" in source
    assert "policy_callable_gate3_zero_confidence_recovery_n196.py:infer" in source
    assert "ca2ceb12ecbc62a721bf3f34b7cd6b2675592626604899153c3ab7b92754b210" in source
    assert "[switch]$Gate3FinalTerminalLevelN197" in source
    assert "policy_callable_gate3_final_terminal_level_n197.py:infer" in source
    assert "cfd9308910dbe63a857a67a3b96d53204a2ad929c9ad9b1a93787ba9392e0724" in source
    assert "[switch]$Gate3LowConfidenceLatchedRecoveryN198" in source
    assert "policy_callable_gate3_low_confidence_latched_recovery_n198.py:infer" in source
    assert "9deaecd3957d02b11dabdc7b1fb36ce649f1e5575fd11fc1e190332508742d30" in source
    assert "[switch]$Gate3RestoredTerminalLevelN199" in source
    assert "policy_callable_gate3_restored_terminal_level_n199.py:infer" in source
    assert "acf17a600e25e2c4c43e928052789c03dfa934e865e7bbd7e7d39ddd9a056885" in source
    assert "[switch]$Gate1HighDownThrustCeilingN200" in source
    assert "policy_callable_gate1_high_down_thrust_ceiling_n200.py:infer" in source
    assert "1c555692497ebb2e0ec35c309158ee55813d5caad299709483439bf636c4c1e9" in source
    assert "[switch]$Gate3TightTerminalLevelN201" in source
    assert "policy_callable_gate3_tight_terminal_level_n201.py:infer" in source
    assert "892f38e34da75f29d7b20a42638ff26a19773518ecab751e4f408842733538dd" in source
    assert "[switch]$Gate3CounterHysteresis08N202" in source
    assert "[switch]$Gate3CounterHysteresis12N202" in source
    assert "[switch]$Gate3CounterHysteresis16N202" in source
    assert "policy_callable_gate3_counter_hysteresis_n202.py:infer" in source
    assert "b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe" in source
    assert '$env:PUFFER_GATE3_COUNTER_ENTRY_M = "-0.8"' in source
    assert '$env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.2"' in source
    assert '$env:PUFFER_GATE3_COUNTER_ENTRY_M = "-1.6"' in source
    assert "[switch]$Gate4YawSignN203" in source
    assert "policy_callable_gate4_yaw_sign_n203.py:infer" in source
    assert "a06c41623d7dd154d63450ff028f3de274d107e59a8c2d7407d88b39c2628db9" in source
    assert "[switch]$Gate4CoherentTargetHoldN203" in source
    assert "policy_callable_gate4_coherent_target_hold_n203.py:infer" in source
    assert "23100ecad829add77d9de88f4778d8ec9d920d3a2d03ce93c460b47ff484d511" in source
    assert "[switch]$Gate4LearnedTailN203" in source
    assert "policy_callable_gate4_learned_tail_n203.py:infer" in source
    assert "4a976df88dc95af46d2bd022fd8497608ff9cbe3a47c2265c4215e8243e775af" in source
    assert "[switch]$Gate4TerminalCrossingN206" in source
    assert "policy_callable_gate4_terminal_crossing_n206.py:infer" in source
    assert "b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63" in source
    assert "[switch]$Gate4FilteredTerminalCrossingN206" in source
    assert "[switch]$Gate4PhasePredictiveTerminalN206" in source
    assert "[switch]$Gate4VisualMpcN232" in source
    assert "policy_callable_gate4_mpc_n232.py:infer" in source
    assert "0c291b21ca49cd71281744a1cf1da770f42a36b81dec916691d965e91f5228cf" in source
    assert "n232_official_visual_dynamics_fit_20260719.json" in source
    assert "9aaefaae111734314353757adbd49da913f13c9bb558627e382ba3c2eb8cc275" in source
    assert 'throw "Gate4VisualMpcN232 requires PolicyStateHz=0 (one state step per live update)"' in source
    assert "$env:PUFFER_POLICY_GATE4_DYNAMICS_PATH" in source
    assert "[switch]$Gate4VisualMpcFixedN233" in source
    assert "policy_callable_gate4_mpc_n233.py:infer" in source
    assert "4a3f0c7a0c8b73053f6aa7da8995b4fffe1e2c8ed2f430c79135476ca8f1c4ba" in source
    assert 'throw "Gate4VisualMpcFixedN233 requires PolicyStateHz=60"' in source
    assert "[switch]$Gate4ReacquiringMpcN234" in source
    assert "policy_callable_gate4_reacquiring_mpc_n234.py:infer" in source
    assert "84c08dac68be66039567173a7319831232601c22790792588a044401d7671530" in source
    assert 'throw "Gate4ReacquiringMpcN234 requires PolicyStateHz=60"' in source
    assert "[switch]$Gate4SearchMpcN235" in source
    assert "policy_callable_gate4_search_mpc_n235.py:infer" in source
    assert "dd8cd8d89f070c9ca51820c18a9565bb61eb478b1dcd9cce464d4fd4674c569a" in source
    assert 'throw "Gate4SearchMpcN235 requires PolicyStateHz=60"' in source
    assert "[switch]$Gate4StagedMpcN236" in source
    assert "policy_callable_gate4_staged_mpc_n236.py:infer" in source
    assert "fe7984c17b203ade4b3967ff0e6166be17c4d9a89a827231e2e34be6e28862f4" in source
    assert 'throw "Gate4StagedMpcN236 requires PolicyStateHz=60"' in source
    assert "[switch]$Gate4FreshProjectedInterceptN237" in source
    assert "policy_callable_gate4_fresh_projected_intercept_n237.py:infer" in source
    assert "6fc71223f11246bc90d0c70df41cf67e5bc3e721ae83e6b66c05fcfe1720cb5d" in source
    assert 'throw "Gate4FreshProjectedInterceptN237 requires PolicyStateHz=60"' in source
    assert "[switch]$Gate4IdentityLockedInterceptN238" in source
    assert "policy_callable_gate4_identity_locked_intercept_n238.py:infer" in source
    assert "2e2f05b25907a14ba062f2a140c1068fbc625693302165b97cf811e00d43b8fc" in source
    assert 'throw "Gate4IdentityLockedInterceptN238 requires PolicyStateHz=60"' in source
    assert "[switch]$Gate4IdentityBodyMpcN239" in source
    assert "[switch]$Gate4RollProbeN241" in source
    assert "policy_callable_gate4_identity_body_mpc_n239.py:infer" in source
    assert "716352e7b2ec603e0316e59f0d97cc6fe96ca5bdde4f5f7923b0be2fb6b75f4c" in source
    assert "n239_gate4_identity_body_dynamics_fit_20260719.json" in source
    assert "a40e9279bc2d34f69a2296350971ee27eb1393f64e5527b37f2bdc1a545677df" in source
    assert "$env:PUFFER_POLICY_GATE4_IDENTITY_DYNAMICS_PATH" in source
    assert 'throw "Gate4IdentityBodyMpcN239 requires PolicyStateHz=60"' in source
    assert 'throw "Gate4RollProbeN241 requires PolicyStateHz=60"' in source
    assert "policy_callable_gate4_roll_probe_n241.py:infer" in source
    assert "if ($Gate4FilteredTerminalCrossingN206 -or $Gate4PhasePredictiveTerminalN206)" in source
    assert '$env:PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX = "3"' in source
    assert "Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX" in source
    assert "Remove-Item Env:PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX" in source
    assert "d36e52e4a19c2b24c8f0075d76ea3b61e559fa10b5af506c7fadcb15625df5db" in source
    assert 'throw "Select only one policy variant"' in source
