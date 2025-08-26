import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from config_utils import get_api_config, get_quality_config


def test_configuration():
    print("🧪 Testing Configuration Loading")
    print("=" * 40)

    api_config = get_api_config()
    print("\n📡 API Configuration:")
    print(f"   Base URL: {api_config['base_url']}")
    print(f"   Port: {api_config['port']}")
    print(f"   Host: {api_config['host']}")

    quality_config = get_quality_config()
    print("\n⭐ Quality Configuration:")
    print(f"   Minimum Overall Score: {quality_config['minimum_overall_score']}")
    print(f"   Minimum Critical Score: {quality_config['minimum_critical_aspects_score']}")
    print(f"   Excellent Threshold: {quality_config['excellent_threshold']}")
    print(f"   Good Threshold: {quality_config['good_threshold']}")

    print("\n🎯 Quality Score Classification:")
    test_scores = [1.5, 2.5, 3.5, 4.5, 5.0]

    for score in test_scores:
        if score >= quality_config['excellent_threshold']:
            color = "🟢 Excellent"
        elif score >= quality_config['good_threshold']:
            color = "🟡 Good"
        else:
            color = "🔴 Poor"
        print(f"   Score {score}: {color}")

    print("\n" + "=" * 40)
    print("✅ Configuration test completed!")


if __name__ == "__main__":
    test_configuration()
