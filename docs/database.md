# PostgreSQL persistence (Phase 7)

## Architecture and schema

`src/retention/database/` contains the SQLAlchemy 2 models, input validation,
connection/session management, repository functions, and a small inference
orchestrator. Callers create one engine and session factory at process startup,
then pass the factory to `predict_and_persist`. The frozen predictor is loaded
once per process by `frozen_predictor`; callers may pass an already loaded
predictor. `session_scope` commits after the customer upsert and prediction
insert both succeed, rolls back on any error, and closes the session.

`customers` has a generated internal key, unique external identifier, the 19
validated predictor fields, and creation/update timestamps. Charges are exact
`NUMERIC` values with four decimal places. There is no `Churn` inference
column. An upsert replaces all predictor fields deterministically for an
external identifier.

`predictions` has a UUID key, a `RESTRICT` foreign key to `customers`, a
calibrated probability, class, exact applied threshold, model name/version,
configuration SHA-256 fingerprint, timestamp, and JSONB snapshot of the 19
predictors. The snapshot excludes identifiers and preserves charge values as
decimal strings. Prediction rows are append-only through the repository; old
snapshots stay intact when a customer is updated. Database constraints enforce
probability/threshold ranges and class/threshold consistency. The foreign key
prevents customer deletion while predictions exist. Indexes support customer
history ordered by time and global time queries.

The model identity is `customer-churn-predictor` version `1`, with frozen
threshold `0.3304499866713542`. The fingerprint comes from the frozen
predictor metadata. The code does not retrain or alter calibration.

## Local PostgreSQL setup

A PostgreSQL server is required. Reuse a running local installation where
available. With Homebrew, inspect it using `brew services list` and check
connectivity with `pg_isready`. Only if no server is installed, install and
start a supported PostgreSQL version for your machine.

Use an administrator connection to create dedicated non-superuser roles and
databases. Choose private passwords locally; do not put them in the repository.

```sql
CREATE ROLE retention_app LOGIN PASSWORD 'replace-with-private-password';
CREATE ROLE retention_test LOGIN PASSWORD 'replace-with-different-private-password';
CREATE DATABASE retention_app OWNER retention_app;
CREATE DATABASE retention_test OWNER retention_test;
```

Copy `.env.example` to a local `.env` and replace the placeholders. Export the
values into the process environment before running commands. The application
reads `DATABASE_URL`; integration tests read `TEST_DATABASE_URL`. Both require
`postgresql+psycopg://...` URLs. The test database name must end in `_test`
and differ from the application database. Never point tests at a shared or
populated database: migration tests require an empty test database, downgrade
it afterward, and remove the empty Alembic version table they created. The app does not automatically load `.env`; a shell may use
`set -a; . ./.env; set +a` with a trusted local file.

## Migrations and synthetic check

Review `migrations/versions/0001_customers_predictions.py` before applying.
The production schema is managed by Alembic, not `create_all`.

```sh
.venv/bin/alembic upgrade head
.venv/bin/alembic current
.venv/bin/alembic check
# Only on a disposable database: .venv/bin/alembic downgrade base
```

A small synthetic workflow, after setting `DATABASE_URL` and upgrading:

```python
from retention.database.connection import create_database_engine, make_session_factory
from retention.database.inference import predict_and_persist
from retention.database.repository import customer_history
from retention.database.connection import session_scope

engine = create_database_engine()
factory = make_session_factory(engine)
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": "55.50",
    "TotalCharges": "666.00",
}
try:
    first, result = predict_and_persist(factory, "SYNTHETIC-001", customer)
    second, _ = predict_and_persist(factory, "SYNTHETIC-001", customer)
    with session_scope(factory) as session:
        assert len(customer_history(session, "SYNTHETIC-001")) == 2
    print(first, second, result)
finally:
    engine.dispose()
```

This creates only synthetic data. Do not import the held-out split or labels.
Avoid logging customer attributes or URLs with credentials. Keep dumps and
local credentials outside Git. Local PostgreSQL installations may use `trust`
authentication on Unix sockets or loopback addresses. Check `pg_hba.conf`
before relying on password authentication; production deployment needs its own
access-control review.

## Tests and next schema changes

Run `scripts/test_suite.sh tests/unit/test_database_config.py tests/integration/test_database.py`
with `TEST_DATABASE_URL` configured, then
`scripts/test_suite.sh` for the full suite. Run `.venv/bin/ruff check .`,
`.venv/bin/ruff format --check .`, `.venv/bin/python -m pip check`, and
`git diff --check` for quality gates.

Future SHAP explanations can be attached to an existing prediction UUID in a
separate explanation record once the explanation contract is defined. A future
GenAI Next-Best-Action record can reference that prediction and its model
provenance. Neither table is created in Phase 7.

`requirements-lock.txt` is a dependency of all DVC stages, including final
training and held-out evaluation. The three database packages were already
present in the lock file, so this phase does not modify it. If a later lock
update is needed, review stage-specific dependency boundaries before any DVC
reproduction; never rerun final evaluation merely because database packages
change.
