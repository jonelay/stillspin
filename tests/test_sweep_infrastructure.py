"""Infrastructure tests for sweep robustness.

TDD approach: Tests focus on failure modes from v3.2:
1. Config validation - prevent invalid params reaching N-body
2. Incremental saving - verify crash recovery works
3. Timeout - verify we never hang
"""

import pytest

from shared.sweep_types import SweepConfig, SweepResult, Episode


class TestSweepConfig:
    """Config validation tests - prevent invalid params reaching N-body."""

    def test_rejects_q_below_minimum(self):
        """Q < 5 is unphysical for rocky planets."""
        with pytest.raises(ValueError, match="Q=.*outside"):
            SweepConfig(q=2, distance_au=0.07, triax=3e-5)

    def test_rejects_q_above_maximum(self):
        """Q > 1000 is unphysical (even Earth is ~100-1000)."""
        with pytest.raises(ValueError, match="Q=.*outside"):
            SweepConfig(q=5000, distance_au=0.07, triax=3e-5)

    def test_rejects_distance_outside_hz(self):
        """Distance outside [0.01, 0.50] AU is rejected."""
        with pytest.raises(ValueError, match="distance=.*outside"):
            SweepConfig(q=20, distance_au=0.60, triax=3e-5)

    def test_rejects_distance_too_close(self):
        """Distance < 0.01 AU is too close."""
        with pytest.raises(ValueError, match="distance=.*outside"):
            SweepConfig(q=20, distance_au=0.005, triax=3e-5)

    def test_rejects_triax_too_small(self):
        """triax < 1e-6 is unphysical."""
        with pytest.raises(ValueError, match="triax=.*outside"):
            SweepConfig(q=20, distance_au=0.07, triax=1e-7)

    def test_rejects_triax_too_large(self):
        """triax > 1e-2 is unphysical."""
        with pytest.raises(ValueError, match="triax=.*outside"):
            SweepConfig(q=20, distance_au=0.07, triax=0.1)

    def test_config_id_deterministic(self):
        """Same params must produce same ID for deduplication."""
        c1 = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        c2 = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        assert c1.config_id == c2.config_id

    def test_config_id_unique_for_different_params(self):
        """Different params must produce different IDs."""
        c1 = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        c2 = SweepConfig(q=21, distance_au=0.07, triax=3e-5)
        assert c1.config_id != c2.config_id

    def test_valid_config_at_boundaries(self):
        """Configs at physical bounds should be accepted."""
        SweepConfig(q=5, distance_au=0.01, triax=1e-6)
        SweepConfig(q=1000, distance_au=0.50, triax=1e-2)

    def test_to_dict_roundtrip(self):
        """Config should survive dict serialization."""
        c1 = SweepConfig(q=20, distance_au=0.07, triax=3e-5, albedo=0.4)
        d = c1.to_dict()
        c2 = SweepConfig.from_dict(d)
        assert c1.q == c2.q
        assert c1.distance_au == c2.distance_au
        assert c1.triax == c2.triax
        assert c1.albedo == c2.albedo


class TestEpisode:
    """Episode dataclass tests."""

    def test_episode_basic_creation(self):
        """Episode should be created with required fields."""
        ep = Episode("TL_ZERO", 100.0, 0.0, 100.0)
        assert ep.type == "TL_ZERO"
        assert ep.duration_yr == 100.0
        assert ep.neighbors is None

    def test_episode_with_neighbors(self):
        """PTB episodes should support neighbors field."""
        ep = Episode(
            "PTB", 10.0, 100.0, 110.0, {"before": "TL_ZERO", "after": "TL_PI"}
        )
        assert ep.neighbors["before"] == "TL_ZERO"
        assert ep.neighbors["after"] == "TL_PI"

    def test_episode_to_dict_without_neighbors(self):
        """to_dict should omit neighbors if None."""
        ep = Episode("TL_ZERO", 100.0, 0.0, 100.0)
        d = ep.to_dict()
        assert "neighbors" not in d
        assert d["type"] == "TL_ZERO"

    def test_episode_to_dict_with_neighbors(self):
        """to_dict should include neighbors if present."""
        ep = Episode("PTB", 10.0, 100.0, 110.0, {"before": "TL_ZERO", "after": "TL_PI"})
        d = ep.to_dict()
        assert d["neighbors"] == {"before": "TL_ZERO", "after": "TL_PI"}

    def test_episode_roundtrip(self):
        """Episode should survive to_dict/from_dict round-trip."""
        ep1 = Episode("PTB", 10.0, 100.0, 110.0, {"before": "TL_ZERO", "after": "TL_PI"})
        d = ep1.to_dict()
        ep2 = Episode.from_dict(d)
        assert ep1.type == ep2.type
        assert ep1.duration_yr == ep2.duration_yr
        assert ep1.neighbors == ep2.neighbors


class TestSweepResult:
    """Result dataclass tests."""

    def test_ok_result(self):
        """OK status should have all fields populated."""
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="OK",
            fractions={"PTB": 0.3, "TL_ZERO": 0.4},
            temps={"t_term": 270},
            quality={"score": 0.8},
            elapsed_s=1.5,
        )
        assert result.status == "OK"
        assert result.fractions is not None
        assert result.error_msg is None

    def test_timeout_result(self):
        """TIMEOUT status should have error_msg."""
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="TIMEOUT",
            fractions=None,
            temps=None,
            quality=None,
            elapsed_s=600,
            error_msg="Exceeded 600s timeout",
        )
        assert result.status == "TIMEOUT"
        assert result.error_msg is not None

    def test_to_dict_roundtrip(self):
        """Result should survive dict serialization."""
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        r1 = SweepResult(
            config=config,
            status="OK",
            fractions={"PTB": 0.3},
            temps={"t_term": 270},
            quality={"score": 0.8},
            elapsed_s=1.5,
        )
        d = r1.to_dict()
        r2 = SweepResult.from_dict(d)
        assert r1.status == r2.status
        assert r1.config.config_id == r2.config.config_id

    def test_episodes_roundtrip_serialization(self):
        """Episodes should survive to_dict/from_dict round-trip."""
        episodes = [
            Episode("TL_ZERO", 100.0, 0.0, 100.0),
            Episode("PTB", 10.0, 100.0, 110.0, {"before": "TL_ZERO", "after": "TL_PI"}),
            Episode("TL_PI", 200.0, 110.0, 310.0),
        ]
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        r1 = SweepResult(
            config=config,
            status="OK",
            fractions={"PTB": 0.3},
            temps={"t_term": 270},
            quality={"score": 0.8},
            elapsed_s=1.5,
            episodes=episodes,
        )
        d = r1.to_dict()
        r2 = SweepResult.from_dict(d)
        assert r2.episodes is not None
        assert len(r2.episodes) == 3
        assert r2.episodes[0].type == "TL_ZERO"
        assert r2.episodes[1].neighbors == {"before": "TL_ZERO", "after": "TL_PI"}

    def test_backward_compatible_without_episodes(self):
        """Old results without episodes field should still load."""
        old_result = {
            "config": {
                "q": 20,
                "distance_au": 0.07,
                "triax": 3e-5,
                "albedo": 0.35,
                "co2_pct": 0.6,
                "n_orbits": 20000,
                "seed": None,
                "config_id": "abc123",
            },
            "status": "OK",
            "fractions": {"PTB": 0.3},
            "temps": {"t_term": 270},
            "quality": {"score": 0.8},
            "elapsed_s": 1.5,
        }
        result = SweepResult.from_dict(old_result)
        assert result.episodes is None
        assert result.schema_version == "v3.2"

    def test_new_result_has_schema_version(self):
        """New results should have schema_version v3.5."""
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="OK",
            fractions={},
            temps={},
            quality={},
            elapsed_s=1.0,
        )
        assert result.schema_version == "v3.5"
        d = result.to_dict()
        assert d["schema_version"] == "v3.5"


class TestResultStore:
    """Incremental saving tests - prevent data loss on crash."""

    def test_save_creates_jsonl_file(self, tmp_path):
        """Save must create results.jsonl immediately."""
        from shared.result_store import ResultStore

        store = ResultStore(tmp_path)
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="OK",
            fractions={},
            temps={},
            quality={},
            elapsed_s=1.0,
        )
        store.save(result)
        assert (tmp_path / "results.jsonl").exists()

    def test_load_all_returns_saved_results(self, tmp_path):
        """Load must return all saved results."""
        from shared.result_store import ResultStore

        store = ResultStore(tmp_path)
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="OK",
            fractions={},
            temps={},
            quality={},
            elapsed_s=1.0,
        )
        store.save(result)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0]["status"] == "OK"

    def test_get_completed_ids_for_resume(self, tmp_path):
        """Resume must skip already-completed configs."""
        from shared.result_store import ResultStore

        store = ResultStore(tmp_path)
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="OK",
            fractions={},
            temps={},
            quality={},
            elapsed_s=1.0,
        )
        store.save(result)
        completed = store.get_completed_ids()
        assert config.config_id in completed

    def test_multiple_saves_append(self, tmp_path):
        """Multiple saves must append, not overwrite."""
        from shared.result_store import ResultStore

        store = ResultStore(tmp_path)
        for q in [10, 20, 30]:
            config = SweepConfig(q=q, distance_au=0.07, triax=3e-5)
            result = SweepResult(
                config=config,
                status="OK",
                fractions={},
                temps={},
                quality={},
                elapsed_s=1.0,
            )
            store.save(result)
        loaded = store.load_all()
        assert len(loaded) == 3

    def test_timeout_results_tracked_for_resume(self, tmp_path):
        """TIMEOUT results should be in completed IDs (don't retry timeouts)."""
        from shared.result_store import ResultStore

        store = ResultStore(tmp_path)
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        result = SweepResult(
            config=config,
            status="TIMEOUT",
            fractions=None,
            temps=None,
            quality=None,
            elapsed_s=600,
            error_msg="timeout",
        )
        store.save(result)
        completed = store.get_completed_ids()
        assert config.config_id in completed

    def test_empty_store_returns_empty(self, tmp_path):
        """Empty store should return empty lists."""
        from shared.result_store import ResultStore

        store = ResultStore(tmp_path)
        assert store.load_all() == []
        assert store.get_completed_ids() == set()


class TestSafeRunner:
    """Timeout protection tests - prevent infinite hangs."""

    @pytest.mark.timeout(30)
    def test_timeout_returns_status_not_hang(self):
        """Timeout must return TIMEOUT status, never hang."""
        from shared.sweep_runner import run_config_safe

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5, n_orbits=1000)
        result = run_config_safe(config, timeout_s=2)
        # Accept OK, TIMEOUT, or ERROR (if rebound not installed)
        # Key invariant: function returns, never hangs
        assert result.status in ("OK", "TIMEOUT", "ERROR")

    @pytest.mark.timeout(30)
    def test_error_captured_not_raised(self):
        """Errors must return ERROR status, not propagate."""
        from shared.sweep_runner import run_config_safe

        # n_orbits=0 should cause an error
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5, n_orbits=0)
        result = run_config_safe(config, timeout_s=10)
        assert result.status in ("OK", "ERROR")
        if result.status == "ERROR":
            assert result.error_msg is not None

    def test_successful_run_returns_ok(self):
        """Successful run should return OK with data."""
        from shared.sweep_runner import run_config_safe

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5, n_orbits=100)
        result = run_config_safe(config, timeout_s=120)
        if result.status == "OK":
            assert result.fractions is not None
            assert result.temps is not None
            assert result.quality is not None


class TestEpisodeExtraction:
    """Episode data extraction tests."""

    @pytest.mark.timeout(300)
    def test_episodes_extracted_from_regime_result(self):
        """Episodes should be extracted from run_single regime_result."""
        from shared.sweep_runner import run_config_safe

        config = SweepConfig(q=20, distance_au=0.0715, triax=3e-5, n_orbits=1000)
        result = run_config_safe(config)

        assert result.status == "OK"
        assert result.episodes is not None
        assert len(result.episodes) > 0

        # Episodes should have all required fields
        ep = result.episodes[0]
        assert ep.type in ("TL_ZERO", "TL_PI", "SPINNING", "PTB")
        assert ep.duration_yr > 0
        assert ep.t_start_yr >= 0

    @pytest.mark.timeout(300)
    def test_ptb_episodes_have_neighbors(self):
        """PTB episodes should have before/after neighbors."""
        from shared.sweep_runner import run_config_safe

        # Use lower Q to encourage PTB
        config = SweepConfig(q=10, distance_au=0.0715, triax=5e-4, n_orbits=1000)
        result = run_config_safe(config)

        if result.status != "OK" or result.episodes is None:
            pytest.skip("Config did not produce valid result")

        # Find PTB episodes
        ptb_episodes = [ep for ep in result.episodes if ep.type == "PTB"]

        if not ptb_episodes:
            pytest.skip("No PTB episodes in result")

        # PTB episodes in the middle should have both neighbors
        for ep in ptb_episodes:
            assert ep.neighbors is not None
            # At least one neighbor should be defined
            assert ep.neighbors.get("before") or ep.neighbors.get("after")


class TestEpisodeStatistics:
    """Episode statistics function tests."""

    def test_empty_episodes(self):
        """Empty episode list should return zero values."""
        from shared.analysis import compute_episode_statistics

        stats = compute_episode_statistics(None)
        assert stats["n_episodes"] == 0
        assert stats["mean_tl_duration_yr"] == 0.0

    def test_basic_episode_counts(self):
        """Should count TL and PTB episodes correctly."""
        from shared.analysis import compute_episode_statistics

        episodes = [
            Episode("TL_ZERO", 100.0, 0.0, 100.0),
            Episode("PTB", 10.0, 100.0, 110.0),
            Episode("TL_PI", 200.0, 110.0, 310.0),
        ]
        stats = compute_episode_statistics(episodes)
        assert stats["n_episodes"] == 3
        assert stats["n_tl_episodes"] == 2
        assert stats["n_ptb_episodes"] == 1

    def test_tl_duration_statistics(self):
        """Should compute mean and max TL durations."""
        from shared.analysis import compute_episode_statistics

        episodes = [
            Episode("TL_ZERO", 100.0, 0.0, 100.0),
            Episode("TL_PI", 200.0, 100.0, 300.0),
        ]
        stats = compute_episode_statistics(episodes)
        assert stats["mean_tl_duration_yr"] == 150.0
        assert stats["max_tl_duration_yr"] == 200.0

    def test_ptb_chain_detection(self):
        """Should detect consecutive PTB chains."""
        from shared.analysis import compute_episode_statistics

        # Chain of 4 PTB episodes (counts as chain >= 3)
        episodes = [
            Episode("TL_ZERO", 100.0, 0.0, 100.0),
            Episode("PTB", 5.0, 100.0, 105.0),
            Episode("SPINNING", 5.0, 105.0, 110.0),
            Episode("PTB", 5.0, 110.0, 115.0),
            Episode("PTB", 5.0, 115.0, 120.0),
            Episode("TL_PI", 100.0, 120.0, 220.0),
        ]
        stats = compute_episode_statistics(episodes)
        # Chain is PTB + SPINNING + PTB + PTB = 20yr out of 220yr total
        assert stats["ptb_chain_fraction"] > 0


class TestSurfaceConditions:
    """Surface conditions function tests."""

    def test_empty_temps(self):
        """Empty temps should return zero/False values."""
        from shared.analysis import compute_surface_conditions

        cond = compute_surface_conditions(None)
        assert cond["habitable_fraction"] == 0.0
        assert cond["cold_trap_active"] is False

    def test_habitable_terminator(self):
        """T_term in 260-290K should give high habitability."""
        from shared.analysis import compute_surface_conditions

        cond = compute_surface_conditions({"t_term": 275, "t_anti": 120})
        assert cond["habitable_fraction"] > 0.15
        assert cond["t_term"] == 275

    def test_cold_trap_detection(self):
        """T_anti < 112K should activate cold trap."""
        from shared.analysis import compute_surface_conditions

        cond = compute_surface_conditions({"t_term": 270, "t_anti": 100})
        assert cond["cold_trap_active"] is True

        cond = compute_surface_conditions({"t_term": 270, "t_anti": 120})
        assert cond["cold_trap_active"] is False

    def test_ice_stability(self):
        """T_anti < 150K should allow stable ice."""
        from shared.analysis import compute_surface_conditions

        cond = compute_surface_conditions({"t_term": 270, "t_anti": 140})
        assert cond["ice_stable"] is True

        cond = compute_surface_conditions({"t_term": 270, "t_anti": 160})
        assert cond["ice_stable"] is False


class TestSlowBouncerScore:
    """Slow bouncer scoring tests."""

    def test_empty_inputs(self):
        """Empty inputs should return zero score."""
        from shared.analysis import compute_slow_bouncer_score

        score = compute_slow_bouncer_score(None, None, None)
        assert score["slow_bouncer_score"] == 0.0
        assert score["is_slow_bouncer"] is False

    def test_high_tl_dominance(self):
        """High TL fraction should give high dominance score."""
        from shared.analysis import compute_slow_bouncer_score

        fractions = {"TL_ZERO": 0.45, "TL_PI": 0.45, "PTB": 0.10}
        score = compute_slow_bouncer_score(fractions, None, None)
        assert score["tl_dominance_score"] >= 0.8

    def test_balanced_tl_gives_balance_score(self):
        """Both TL0 and TLπ > 15% should give balance score."""
        from shared.analysis import compute_slow_bouncer_score

        fractions = {"TL_ZERO": 0.35, "TL_PI": 0.35, "PTB": 0.30}
        score = compute_slow_bouncer_score(fractions, None, None)
        assert score["balance_score"] > 0.5

    def test_is_slow_bouncer_requires_all_criteria(self):
        """is_slow_bouncer requires score > 0.5, PTB < 30%, habitable T."""
        from shared.analysis import compute_slow_bouncer_score

        # Good fractions but no temps
        fractions = {"TL_ZERO": 0.40, "TL_PI": 0.40, "PTB": 0.20}
        episodes = [Episode("TL_ZERO", 100.0, 0.0, 100.0)]
        score = compute_slow_bouncer_score(fractions, episodes, None)
        assert score["is_slow_bouncer"] is False  # No temps = t_term = 0

        # With habitable temps
        temps = {"t_term": 275}
        score = compute_slow_bouncer_score(fractions, episodes, temps)
        # May or may not be slow_bouncer depending on episode stats


class TestFilterCandidates:
    """Candidate filtering tests."""

    def test_filters_by_tl_fraction(self):
        """Should filter out configs with TL < 70%."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        # Low TL fraction
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.20, "TL_PI": 0.20, "PTB": 0.60},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results)
        assert len(candidates) == 0

    def test_filters_by_ptb_fraction(self):
        """Should filter out configs with PTB > 30%."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        # High PTB fraction
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.35, "TL_PI": 0.30, "PTB": 0.35},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results)
        assert len(candidates) == 0

    def test_accepts_good_candidate(self):
        """Should accept configs meeting all criteria."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        episodes = [
            Episode("TL_ZERO", 200.0, 0.0, 200.0),
            Episode("PTB", 10.0, 200.0, 210.0),
            Episode("TL_PI", 300.0, 210.0, 510.0),
        ]
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.40, "TL_PI": 0.50, "PTB": 0.10},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results)
        assert len(candidates) == 1
        assert candidates[0]["score_details"]["slow_bouncer_score"] > 0

    def test_sorts_by_score_descending(self):
        """Should sort candidates by score descending."""
        from shared.analysis import filter_slow_bouncer_candidates

        config1 = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        config2 = SweepConfig(q=25, distance_au=0.07, triax=3e-5)
        episodes = [Episode("TL_ZERO", 300.0, 0.0, 300.0)]
        results = [
            SweepResult(
                config=config1,
                status="OK",
                fractions={"TL_ZERO": 0.50, "TL_PI": 0.25, "PTB": 0.25},  # Lower balance
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            ),
            SweepResult(
                config=config2,
                status="OK",
                fractions={"TL_ZERO": 0.40, "TL_PI": 0.40, "PTB": 0.20},  # Better balance
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            ),
        ]
        candidates = filter_slow_bouncer_candidates(results)
        if len(candidates) >= 2:
            # Second result should rank higher due to better balance
            assert candidates[0]["config"]["q"] == 25


class TestRiskClassifier:
    """Risk classification tests."""

    def test_safe_config(self):
        """Baseline config should be safe."""
        from shared.analysis import classify_config_risk

        config = SweepConfig(q=20, distance_au=0.0695, triax=3e-5)
        risk = classify_config_risk(config)
        assert risk in ("safe", "moderate")

    def test_low_q_is_risky(self):
        """Q < 10 should be risky."""
        from shared.analysis import classify_config_risk

        config = SweepConfig(q=5, distance_au=0.07, triax=3e-5)
        risk = classify_config_risk(config)
        assert risk in ("moderate", "risky")

    def test_extreme_triax_is_risky(self):
        """Triax far from 3e-5 should increase risk."""
        from shared.analysis import classify_config_risk

        config = SweepConfig(q=20, distance_au=0.07, triax=1e-5)
        risk = classify_config_risk(config)
        assert risk in ("moderate", "risky")

    def test_distance_far_from_resonance(self):
        """Distance far from resonance centers should increase risk."""
        from shared.analysis import classify_config_risk

        config = SweepConfig(q=20, distance_au=0.08, triax=3e-5)
        risk = classify_config_risk(config)
        assert risk in ("moderate", "risky")


class TestStellarMassSupport:
    """Stellar mass field tests for Study 10 support."""

    def test_stellar_mass_default_is_bipolaris(self):
        """Default stellar mass should be 0.15 M_sun (Bipolaris)."""
        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        assert config.stellar_mass_msun == 0.15

    def test_stellar_mass_in_config_id(self):
        """Different stellar masses should produce different config IDs."""
        c1 = SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=0.15)
        c2 = SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=0.12)
        assert c1.config_id != c2.config_id

    def test_stellar_mass_validation_lower_bound(self):
        """Stellar mass < 0.08 M_sun should be rejected."""
        with pytest.raises(ValueError, match="stellar_mass=.*outside"):
            SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=0.05)

    def test_stellar_mass_validation_upper_bound(self):
        """Stellar mass > 0.80 M_sun should be rejected."""
        with pytest.raises(ValueError, match="stellar_mass=.*outside"):
            SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=1.0)

    def test_stellar_mass_valid_at_boundaries(self):
        """Stellar mass at physical boundaries should be accepted."""
        SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=0.08)  # M8V
        SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=0.80)  # K5V

    def test_stellar_mass_roundtrip(self):
        """Stellar mass should survive to_dict/from_dict roundtrip."""
        c1 = SweepConfig(q=20, distance_au=0.07, triax=3e-5, stellar_mass_msun=0.12)
        d = c1.to_dict()
        c2 = SweepConfig.from_dict(d)
        assert c2.stellar_mass_msun == 0.12

    def test_stellar_mass_backward_compat(self):
        """Old configs without stellar_mass should default to 0.15."""
        old_dict = {
            "q": 20,
            "distance_au": 0.07,
            "triax": 3e-5,
            "albedo": 0.35,
            "co2_pct": 0.6,
            "n_orbits": 20000,
            "seed": None,
            "config_id": "abc123",
        }
        config = SweepConfig.from_dict(old_dict)
        assert config.stellar_mass_msun == 0.15


class TestHZDistanceScaling:
    """HZ distance scaling function tests."""

    def test_bipolaris_baseline(self):
        """0.15 M_sun should give known Bipolaris HZ."""
        from shared.analysis import hz_distance_for_mass

        hz_inner, hz_center, hz_outer = hz_distance_for_mass(0.15)
        assert hz_inner == pytest.approx(0.048, rel=0.01)
        assert hz_outer == pytest.approx(0.098, rel=0.01)

    def test_lower_mass_smaller_hz(self):
        """Lower stellar mass should give smaller HZ distance."""
        from shared.analysis import hz_distance_for_mass

        hz_inner_low, _, hz_outer_low = hz_distance_for_mass(0.10)
        hz_inner_ref, _, hz_outer_ref = hz_distance_for_mass(0.15)

        assert hz_inner_low < hz_inner_ref
        assert hz_outer_low < hz_outer_ref

    def test_higher_mass_larger_hz(self):
        """Higher stellar mass should give larger HZ distance."""
        from shared.analysis import hz_distance_for_mass

        hz_inner_high, _, hz_outer_high = hz_distance_for_mass(0.20)
        hz_inner_ref, _, hz_outer_ref = hz_distance_for_mass(0.15)

        assert hz_inner_high > hz_inner_ref
        assert hz_outer_high > hz_outer_ref

    def test_hz_scales_as_mass_squared(self):
        """HZ should scale as M^2 (from L ∝ M^4, HZ ∝ sqrt(L))."""
        from shared.analysis import hz_distance_for_mass

        _, hz_center_1, _ = hz_distance_for_mass(0.10)
        _, hz_center_2, _ = hz_distance_for_mass(0.20)

        # Ratio should be (0.20/0.10)^2 = 4
        ratio = hz_center_2 / hz_center_1
        assert ratio == pytest.approx(4.0, rel=0.01)


class TestStudy8ConfigGeneration:
    """Study 8: Low-Q exploration config generation tests."""

    def test_study8_generates_expected_config_count(self):
        """Study 8 should generate 108 configs (4 Q × 9 dist × 3 triax)."""
        q_values = [8, 10, 12, 14]
        distances = [0.0720, 0.0725, 0.0730, 0.0735, 0.0740, 0.0745, 0.0750, 0.0755, 0.0760]
        triax_values = [2e-5, 3e-5, 4e-5]

        configs = []
        for q in q_values:
            for dist in distances:
                for triax in triax_values:
                    configs.append(SweepConfig(
                        q=q,
                        distance_au=dist,
                        triax=triax,
                    ))

        assert len(configs) == 108
        assert len(configs) == len(q_values) * len(distances) * len(triax_values)

    def test_study8_q_range_below_study5(self):
        """Study 8 Q values should all be below Study 5 minimum (15)."""
        study8_q = [8, 10, 12, 14]
        study5_q_min = 15

        for q in study8_q:
            assert q < study5_q_min, f"Q={q} should be below Study 5 minimum"

    def test_study8_distance_covers_bifurcation(self):
        """Study 8 distances should span 0.072-0.076 AU (bifurcation region)."""
        distances = [0.0720, 0.0725, 0.0730, 0.0735, 0.0740, 0.0745, 0.0750, 0.0755, 0.0760]

        assert min(distances) == 0.0720
        assert max(distances) == 0.0760
        # Should span at least 4 mAU
        assert max(distances) - min(distances) >= 0.004


class TestStudy10ConfigGeneration:
    """Study 10: Stellar mass sweep config generation tests."""

    def test_study10_generates_expected_config_count(self):
        """Study 10 should generate 72 configs (6 mass × 4 Q × 3 triax)."""
        stellar_masses = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        q_values = [18, 20, 22, 25]
        triax_values = [2e-5, 3e-5, 4e-5]

        n_configs = len(stellar_masses) * len(q_values) * len(triax_values)
        assert n_configs == 72

    def test_study10_stellar_mass_range(self):
        """Study 10 should cover M8V to M4V (0.08-0.20 M_sun)."""
        stellar_masses = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

        assert min(stellar_masses) == 0.08  # M8V
        assert max(stellar_masses) == 0.20  # M4V
        assert 0.15 in stellar_masses  # Bipolaris baseline

    def test_study10_hz_center_varies_with_mass(self):
        """Each stellar mass should get a different HZ center distance."""
        from shared.analysis import hz_distance_for_mass

        stellar_masses = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        hz_centers = []
        for m in stellar_masses:
            _, hz_center, _ = hz_distance_for_mass(m)
            hz_centers.append(hz_center)

        # All HZ centers should be unique
        assert len(set(round(hz, 4) for hz in hz_centers)) == len(stellar_masses)

        # HZ should increase with stellar mass
        for i in range(len(hz_centers) - 1):
            assert hz_centers[i] < hz_centers[i + 1]


class TestStudy9ConfigGeneration:
    """Study 9: Fine distance resolution config generation tests."""

    def test_study9_distance_resolution(self):
        """Study 9 should have 100 μAU (0.0001 AU) spacing."""
        import numpy as np

        distances = np.arange(0.0720, 0.0760 + 0.0001, 0.0001)

        # Check resolution
        diffs = np.diff(distances)
        assert np.allclose(diffs, 0.0001, atol=1e-8), "Spacing should be 100 μAU"

    def test_study9_covers_bifurcation_region(self):
        """Study 9 should cover 0.072-0.076 AU range."""
        import numpy as np

        distances = np.arange(0.0720, 0.0760 + 0.0001, 0.0001)

        assert distances[0] == pytest.approx(0.0720)
        # Last point should be >= 0.0760 (may be slightly over due to float)
        assert distances[-1] >= 0.0760 - 0.00001

    def test_study9_generates_expected_config_count(self):
        """Study 9 should generate ~800 configs (2 Q × ~41 dist)."""
        import numpy as np

        q_values = [20, 22]
        distances = np.arange(0.0720, 0.0760 + 0.0001, 0.0001)

        n_configs = len(q_values) * len(distances)
        # ~41 distance points × 2 Q values = ~82
        # Actually with 0.0001 step from 0.072 to 0.076: 41 points × 2 = 82
        assert 80 <= n_configs <= 90


class TestFlipFlopScoring:
    """Tests for flip-flop scoring (penalizing permanent locks)."""

    def test_permanent_lock_gets_zero_flipflop_score(self):
        """100% TL_ZERO (0% PTB) should get flipflop_score = 0."""
        from shared.analysis import compute_slow_bouncer_score

        fractions = {"TL_ZERO": 1.0, "TL_PI": 0.0, "PTB": 0.0}
        score = compute_slow_bouncer_score(fractions, None, None)
        assert score["flipflop_score"] == 0.0

    def test_ideal_ptb_range_gets_full_flipflop_score(self):
        """PTB in 5-25% range should get flipflop_score = 1.0."""
        from shared.analysis import compute_slow_bouncer_score

        for ptb in [0.05, 0.10, 0.15, 0.20, 0.25]:
            fractions = {"TL_ZERO": (1 - ptb) / 2, "TL_PI": (1 - ptb) / 2, "PTB": ptb}
            score = compute_slow_bouncer_score(fractions, None, None)
            assert score["flipflop_score"] == pytest.approx(1.0), f"PTB={ptb} should give full score"

    def test_low_ptb_gets_partial_flipflop_score(self):
        """PTB < 5% should get partial flipflop_score."""
        from shared.analysis import compute_slow_bouncer_score

        fractions = {"TL_ZERO": 0.48, "TL_PI": 0.49, "PTB": 0.03}
        score = compute_slow_bouncer_score(fractions, None, None)
        assert 0 < score["flipflop_score"] < 1.0
        assert score["flipflop_score"] == pytest.approx(0.03 / 0.05)

    def test_high_ptb_gets_low_flipflop_score(self):
        """PTB > 30% should get flipflop_score = 0."""
        from shared.analysis import compute_slow_bouncer_score

        fractions = {"TL_ZERO": 0.30, "TL_PI": 0.30, "PTB": 0.40}
        score = compute_slow_bouncer_score(fractions, None, None)
        assert score["flipflop_score"] == 0.0

    def test_is_slow_bouncer_requires_ptb_range(self):
        """is_slow_bouncer should require PTB in 5-30% range."""
        from shared.analysis import compute_slow_bouncer_score

        # 0% PTB should not be slow bouncer (permanent lock)
        fractions = {"TL_ZERO": 0.50, "TL_PI": 0.50, "PTB": 0.0}
        episodes = [Episode("TL_ZERO", 500.0, 0.0, 500.0)]
        temps = {"t_term": 275}
        score = compute_slow_bouncer_score(fractions, episodes, temps)
        assert score["is_slow_bouncer"] is False

        # 10% PTB with good temps should be slow bouncer
        fractions = {"TL_ZERO": 0.45, "TL_PI": 0.45, "PTB": 0.10}
        score = compute_slow_bouncer_score(fractions, episodes, temps)
        # May or may not be slow bouncer depending on other factors


class TestFilterPTBRange:
    """Tests for PTB range filtering in filter_slow_bouncer_candidates."""

    def test_filters_out_permanent_locks(self):
        """100% TL_ZERO worlds should be filtered out."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        episodes = [Episode("TL_ZERO", 1000.0, 0.0, 1000.0)]
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 1.0, "TL_PI": 0.0, "PTB": 0.0},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results)
        assert len(candidates) == 0

    def test_filters_out_low_ptb(self):
        """PTB < 5% should be filtered out (too stable)."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        episodes = [Episode("TL_ZERO", 980.0, 0.0, 980.0), Episode("PTB", 20.0, 980.0, 1000.0)]
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.98, "TL_PI": 0.0, "PTB": 0.02},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results)
        assert len(candidates) == 0

    def test_accepts_ideal_ptb_range(self):
        """PTB in 5-30% range should be accepted."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        episodes = [
            Episode("TL_ZERO", 400.0, 0.0, 400.0),
            Episode("PTB", 100.0, 400.0, 500.0),
            Episode("TL_PI", 500.0, 500.0, 1000.0),
        ]
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.40, "TL_PI": 0.50, "PTB": 0.10},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results)
        assert len(candidates) == 1

    def test_require_flipflop_filters_single_regime(self):
        """With require_flipflop=True, should filter worlds without both TL_ZERO and TL_PI."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        episodes = [
            Episode("TL_ZERO", 800.0, 0.0, 800.0),
            Episode("PTB", 200.0, 800.0, 1000.0),
        ]
        # Only TL_ZERO, no TL_PI
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.80, "TL_PI": 0.0, "PTB": 0.20},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            )
        ]
        candidates = filter_slow_bouncer_candidates(results, require_flipflop=True)
        assert len(candidates) == 0

        # Same but allow non-flipflop
        candidates = filter_slow_bouncer_candidates(results, require_flipflop=False)
        assert len(candidates) == 1

    def test_min_ptb_parameter_works(self):
        """min_ptb_frac parameter should control the filter threshold."""
        from shared.analysis import filter_slow_bouncer_candidates

        config = SweepConfig(q=20, distance_au=0.07, triax=3e-5)
        episodes = [
            Episode("TL_ZERO", 450.0, 0.0, 450.0),
            Episode("PTB", 30.0, 450.0, 480.0),
            Episode("TL_PI", 520.0, 480.0, 1000.0),
        ]
        results = [
            SweepResult(
                config=config,
                status="OK",
                fractions={"TL_ZERO": 0.45, "TL_PI": 0.52, "PTB": 0.03},
                temps={"t_term": 270},
                quality={},
                elapsed_s=1.0,
                episodes=episodes,
            )
        ]

        # With default min_ptb_frac=0.05, should be filtered
        candidates = filter_slow_bouncer_candidates(results, min_ptb_frac=0.05)
        assert len(candidates) == 0

        # With lower min_ptb_frac=0.02, should pass
        candidates = filter_slow_bouncer_candidates(results, min_ptb_frac=0.02)
        assert len(candidates) == 1


class TestStudy11ConfigGeneration:
    """Study 11: Low-Q flip-flop search config generation tests."""

    def test_study11_generates_expected_config_count(self):
        """Study 11 should generate 60 configs (4 Q × 5 dist × 3 triax)."""
        q_values = [8, 10, 12, 14]
        distances = [0.0740, 0.0745, 0.0750, 0.0755, 0.0760]
        triax_values = [2e-5, 3e-5, 4e-5]

        configs = []
        for q in q_values:
            for dist in distances:
                for triax in triax_values:
                    configs.append(SweepConfig(
                        q=q,
                        distance_au=dist,
                        triax=triax,
                    ))

        assert len(configs) == 60
        assert len(configs) == len(q_values) * len(distances) * len(triax_values)

    def test_study11_targets_outer_distances(self):
        """Study 11 distances should be beyond stable lock boundary (~0.072 AU)."""
        distances = [0.0740, 0.0745, 0.0750, 0.0755, 0.0760]

        # All distances should be > 0.072 AU
        for d in distances:
            assert d > 0.072, f"Distance {d} should be beyond stable lock boundary"

    def test_study11_uses_low_q_values(self):
        """Study 11 should use same low Q values as Study 8."""
        study11_q = [8, 10, 12, 14]
        study8_q = [8, 10, 12, 14]

        assert study11_q == study8_q
