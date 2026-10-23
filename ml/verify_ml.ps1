# Bharat-TravelShield AI - ML module verification (Windows PowerShell).
#
# Run from the repository root, in an environment that has
# ml/requirements-inference.txt installed (Python 3.13):
#
#     powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1
#     powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1 -Python "C:\path\to\python.exe"
#
# Nothing here commits, pushes or modifies tracked files. (Steps 4-5 only READ the datasets.)
param([string]$Python = "python")

function Step($text) { Write-Host ""; Write-Host "=== $text ===" -ForegroundColor Cyan }
function Py { & $Python @args }

Step "1. Branch and working tree (before)"
git branch --show-current
git status --short

Step "2. Interpreter and libraries"
Py -c "import sys, sklearn, numpy, pandas; print('python', sys.version.split()[0], '| scikit-learn', sklearn.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__)"

Step "3. Unit tests (preparation, leakage, training protocol, inference, invalid input, fresh process)"
Py -m unittest ml.test_hazard_datasets ml.test_hazard_training ml.test_hazard_inference
Step "3b. Other suites (data layer, audit, legacy heatwave v1, beginner demo, backend contract example)"
Py -m unittest ml.test_destination_analysis ml.test_data_audit ml.test_heatwave_inference ml.test_demo_local ml.test_contract_examples
Py ml/test_heatwave_dataset.py
Py ml/test_heatwave_models.py

Step "4. Model datasets: rebuild in memory and compare with the committed files"
Py -m ml.hazard_datasets --check

Step "5. Which hazards are supported? (computed from the data)"
Py -m ml.hazard_datasets --readiness
Py -m ml.hazard_inference --list

Step "6. Real inference: Hyderabad (code 536), flood, with ANN explanation"
Py -m ml.hazard_inference --hazard flood --district-code 536

Step "6b. Real inference: Agra (code 146), malaria; heatwave probability is withheld (poor calibration)"
Py -m ml.hazard_inference --hazard malaria --district-code 146 --no-explain
Py -m ml.hazard_inference --hazard heatwave --district-code 146 --no-explain

Step "6c. Destination analysis demo: Agra (readable view of the records + ANN outputs, real data)"
Py -m ml.demo_local destination

Step "7. Invalid input and missing data (each must print a clear status, never a score)"
Write-Host "-- missing features:"
Py -c "import json; from ml.hazard_inference import predict_hazard as p; print(json.dumps(p('flood', {'latitude': 17.4}), indent=1))"
Write-Host "-- NaN humidity:"
Py -c "import json; from ml.hazard_inference import predict_for_district as d, predict_hazard as p; r=d('heatwave',146,explain=False); f=r['input_features']; f['avg_humidity_pct']=float('nan'); print(json.dumps(p('heatwave', f)['errors'], indent=1))"
Write-Host "-- dengue (unsupported):"
Py -c "import json; from ml.hazard_inference import predict_hazard as p; print(json.dumps(p('dengue', {}), indent=1))"
Write-Host "-- Darjiling (327) has no malaria records -> unavailable, nothing imputed:"
Py -c "import json; from ml.hazard_inference import predict_for_district as d; r=d('malaria',327); print(json.dumps({k:r[k] for k in ('status','reason','prediction')}, indent=1))"
Write-Host "-- chikungunya (unsupported):"
Py -c "import json; from ml.hazard_inference import predict_hazard as p; r=p('chikungunya', {}); print(json.dumps({k:r[k] for k in ('status','reason','prediction')}, indent=1))"
Write-Host "-- unknown destination (destination summary must report an error, not guess):"
Py -m ml.demo_local destination --district-name Nowhereville --no-ann
Write-Host "-- ambiguous destination name (Bilaspur exists in two states):"
Py -m ml.demo_local destination --district-name Bilaspur --no-ann
Write-Host "-- year with no complete record:"
Py -c "import json; from ml.hazard_inference import predict_for_district as d; r=d('flood',536,1999); print(json.dumps({k:r[k] for k in ('status','reason')}, indent=1))"

Step "8. Fresh-process check: two separate interpreters must print identical output"
$a = Py -m ml.hazard_inference --hazard flood --district-code 536
$b = Py -m ml.hazard_inference --hazard flood --district-code 536
if (($a -join "`n") -eq ($b -join "`n")) { Write-Host "IDENTICAL output from two fresh processes" -ForegroundColor Green } else { Write-Host "DIFFERENT output" -ForegroundColor Red }

Step "9. Data audit (check mode); each warning is explained in ml/VERIFY_IN_TERMINAL.md"
Py -m ml.data_audit --check

Step "10. Working tree (after)"
git status --short
