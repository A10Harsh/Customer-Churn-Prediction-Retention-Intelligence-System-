INSERT INTO staging.telco_customers (
    source_version, customerid, gender, seniorcitizen, partner, dependents,
    tenure, phoneservice, multiplelines, internetservice, onlinesecurity,
    onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies,
    contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges, churn
)
SELECT source_version, customerid, gender, seniorcitizen::INTEGER, partner,
    dependents, tenure::INTEGER, phoneservice, multiplelines, internetservice,
    onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv,
    streamingmovies, contract, paperlessbilling, paymentmethod,
    monthlycharges::DOUBLE PRECISION,
    NULLIF(BTRIM(totalcharges), '')::DOUBLE PRECISION,
    CASE churn WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END
FROM raw.telco_customers WHERE source_version = :source_version
ON CONFLICT (source_version, customerid) DO NOTHING;
