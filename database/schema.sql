CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS predictions;

CREATE TABLE IF NOT EXISTS raw.ingestion_log (
    source_version TEXT PRIMARY KEY,
    source_rows INTEGER NOT NULL CHECK (source_rows > 0)
);

CREATE TABLE IF NOT EXISTS raw.telco_customers (
    source_version TEXT NOT NULL REFERENCES raw.ingestion_log(source_version),
    customerid TEXT NOT NULL, gender TEXT, seniorcitizen TEXT, partner TEXT,
    dependents TEXT, tenure TEXT, phoneservice TEXT, multiplelines TEXT,
    internetservice TEXT, onlinesecurity TEXT, onlinebackup TEXT,
    deviceprotection TEXT, techsupport TEXT, streamingtv TEXT,
    streamingmovies TEXT, contract TEXT, paperlessbilling TEXT,
    paymentmethod TEXT, monthlycharges TEXT, totalcharges TEXT, churn TEXT,
    PRIMARY KEY (source_version, customerid)
);

CREATE TABLE IF NOT EXISTS staging.telco_customers (
    source_version TEXT NOT NULL REFERENCES raw.ingestion_log(source_version),
    customerid TEXT NOT NULL, gender TEXT, seniorcitizen INTEGER,
    partner TEXT, dependents TEXT, tenure INTEGER, phoneservice TEXT,
    multiplelines TEXT, internetservice TEXT, onlinesecurity TEXT,
    onlinebackup TEXT, deviceprotection TEXT, techsupport TEXT,
    streamingtv TEXT, streamingmovies TEXT, contract TEXT,
    paperlessbilling TEXT, paymentmethod TEXT, monthlycharges DOUBLE PRECISION,
    totalcharges DOUBLE PRECISION, churn INTEGER,
    PRIMARY KEY (source_version, customerid)
);
