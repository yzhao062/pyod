"""ADEngine: Intelligent anomaly detection in 3 lines.

Demonstrates PyOD's ADEngine for automatic detector selection
and anomaly detection across data types.
"""
from pyod.utils.ad_engine import ADEngine
from pyod.utils.data import generate_data

# Generate sample data
X_train, X_test, y_train, y_test = generate_data(
    n_train=300, n_test=100, n_features=20, contamination=0.1)

# Initialize the engine
engine = ADEngine()

# === One-shot detection ===
result = engine.detect(X_train)
print("Detector chosen:", result['plan']['detector_name'])
print("Reason:", result['plan']['reason'])
print("Anomalies found:", result['n_anomalies'])
print()

# === Step-by-step lifecycle ===
# 1. Profile the data
profile = engine.profile_data(X_train)
print("Data profile:", profile)

# 2. Plan detection
plan = engine.plan_detection(profile, priority='speed')
print("Plan:", plan['detector_name'], "-", plan['reason'])

# 3. Build detector
clf = engine.build_detector(plan)
print("Detector:", clf)

# 4. Fit and predict
clf.fit(X_train)
print("Training anomalies:", clf.labels_.sum())
print("Test scores:", clf.decision_function(X_test)[:5])
print()

# === Knowledge queries ===
print("=== Available text detectors ===")
for d in engine.list_detectors(data_type='text'):
    print(f"  {d['name']}: {d['full_name']}")

print()
print("=== ECOD explained ===")
info = engine.explain_detector('ECOD')
print(f"  {info['full_name']}")
print(f"  Best for: {info['best_for']}")
print(f"  Strengths: {', '.join(info['strengths'][:3])}")

print()
print("=== ADBench results ===")
bench = engine.get_benchmarks('ADBench')
print(f"  Top 5: {bench['ADBench']['rankings']['overall_top_5']}")
