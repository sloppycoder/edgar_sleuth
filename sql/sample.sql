DROP TABLE IF EXISTS sampled_cik;
CREATE TABLE sampled_cik AS WITH distinct_cik AS (
    SELECT DISTINCT cik
    FROM master_idx
    WHERE date_filed BETWEEN '2022-01-01' AND '2024-12-31'
    AND cik in (SELECT cik FROM fund_cik_map)
) -- Get all distinct `cik` values for the specified date range
---Randomly sample 10% of the distinct `cik` values
SELECT cik
FROM distinct_cik
ORDER BY random()
LIMIT (
        SELECT CEIL(0.2 * COUNT(*))
        FROM distinct_cik
    );
-- Step 3: Insert deduped records into the new table
DROP TABLE IF EXISTS master_idx_sample;
CREATE TABLE master_idx_sample AS
SELECT DISTINCT *
FROM master_idx
WHERE date_filed BETWEEN '2023-01-01' AND '2024-12-31'
    AND cik IN (
        SELECT cik
        FROM sampled_cik
    );
-- Step4: Tag the records for later processing
UPDATE master_idx_sample
SET tags = ARRAY ['10pct']
