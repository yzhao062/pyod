"""ADEngine: Full anomaly detection lifecycle.

Demonstrates PyOD's ADEngine for automatic detector selection,
execution, analysis, explanation, and report generation.
"""
from pyod.utils.ad_engine import ADEngine
from pyod.utils.data import generate_data

# Generate sample data
X_train, X_test, y_train, y_test = generate_data(
    n_train=300, n_test=100, n_features=20, contamination=0.1)

# Initialize the engine
engine = ADEngine()

# === Full lifecycle ===
print("=" * 60)
print("FULL ANOMALY DETECTION LIFECYCLE")
print("=" * 60)

# 1. Profile
profile = engine.profile_data(X_train)
print("\n1. Data profile:", profile['data_type'],
      "(%d samples, %d features)" % (profile['n_samples'],
                                      profile['n_features']))

# 2. Plan
plan = engine.plan_detection(profile, priority='speed')
print("2. Plan:", plan['detector_name'], "-", plan['reason'])

# 3. Execute
result = engine.run_detection(X_train, plan, X_test=X_test)
print("3. Detection: %d anomalies (%.1f%%) in %.3fs"
      % (result['n_anomalies'], result['anomaly_ratio'] * 100,
         result['runtime_seconds']))

# 4. Analyze
analysis = engine.analyze_results(result, X=X_train)
print("4. Analysis:", analysis['summary'])

# 5. Explain
explanations = engine.explain_findings(result, X=X_train, top_k=3)
print("5. Top anomalies:")
for exp in explanations:
    print("   Sample %d: score=%.4f (%s)"
          % (exp['index'], exp['score'], exp['label']))

# 6. Suggest next step
suggestion = engine.suggest_next_step(result, analysis)
print("6. Suggestion:", suggestion['action'], "-", suggestion['reason'])

# 7. Report
report = engine.generate_report(result, analysis)
print("\n7. Report preview (first 500 chars):")
print(report[:500])

# === Knowledge queries ===
print("\n" + "=" * 60)
print("KNOWLEDGE QUERIES")
print("=" * 60)

print("\nAvailable text detectors:")
for d in engine.list_detectors(data_type='text'):
    print("  %s: %s" % (d['name'], d['full_name']))

print("\nECOD explained:")
info = engine.explain_detector('ECOD')
print("  %s" % info['full_name'])
print("  Best for: %s" % info['best_for'])

print("\nADBench top 5:")
bench = engine.get_benchmarks('ADBench')
print("  %s" % bench['ADBench']['rankings']['overall_top_5'])
