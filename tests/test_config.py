import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from config_utils import get_api_config, get_quality_config


class TestConfiguration:

    @pytest.fixture
    def api_config(self):
        return get_api_config()

    @pytest.fixture
    def quality_config(self):
        return get_quality_config()

    def test_api_configuration_structure(self, api_config):
        assert "base_url" in api_config
        assert "port" in api_config
        assert "host" in api_config

        assert isinstance(api_config["port"], int)
        assert isinstance(api_config["host"], str)
        assert isinstance(api_config["base_url"], str)
        assert api_config["base_url"].startswith("http")

    def test_quality_configuration_structure(self, quality_config):
        required_keys = [
            "excellent_threshold",
            "good_threshold"
        ]

        for key in required_keys:
            assert key in quality_config
            assert isinstance(quality_config[key], (int, float))

    def test_quality_thresholds_logical_order(self, quality_config):
        assert quality_config["good_threshold"] < quality_config["excellent_threshold"]

    @pytest.mark.parametrize("score,expected_classification", [
        (1.5, "poor"),
        (2.5, "poor"),
        (3.5, "good"),
        (4.5, "excellent"),
        (5.0, "excellent")
    ])
    def test_quality_score_classification(self, quality_config, score, expected_classification):
        excellent_threshold = quality_config["excellent_threshold"]
        good_threshold = quality_config["good_threshold"]

        if score >= excellent_threshold:
            actual = "excellent"
        elif score >= good_threshold:
            actual = "good"
        else:
            actual = "poor"

        assert actual == expected_classification

    def test_api_config_values_are_valid(self, api_config):
        assert api_config["port"] > 0
        assert api_config["port"] < 65536
        assert len(api_config["host"]) > 0
        assert ":" in api_config["base_url"]

    def test_quality_config_values_are_valid(self, quality_config):
        for value in quality_config.values():
            assert isinstance(value, (int, float))
            assert value > 0
            assert value <= 5
