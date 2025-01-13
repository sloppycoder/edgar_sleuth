
# EDGAR sleuth
Using LLM to extract information from filings on SEC EDGAR website.


```shell

# load index, thos step loads all records in master_idx
python -m sleuth load-index --input="2024/*"

# manually select and tag records and save them to master_idx_sample
# currently it samples 10% of all companies filed between 2023-2024
# and tag them "10pct"
psql <database> -f sql/sample.sql

# chunk
python -m sleuth chunk --input-tag=10pct --tags=10pct --workers=6

# embedding
python -m sleuth embedding --input-tag=10pct --tags=10pct --workers=4

# embedding
python -m sleuth init-search-phrases --tags=10pct

# extraction
python -m sleuth extract --input-tag=10pct --tags=10pct --workers=3
