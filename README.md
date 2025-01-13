
# EDGAR sleuth
Using LLM to extract information from filings on SEC EDGAR website.


```shell

# load index, thos step loads all records in master_idx
python -m sleuth load-index --input="2024/*"

# manually select and tag records and save them to master_idx_sample
# currently it samples 15% of all companies filed between 2023-2024
# and tag them "15pct"
psql -f sql/sample.sql

# chunk
python -m sleuth chunk --input-tag=15pct --tags=15pct --workers=6

# embedding
python -m sleuth embedding --input-tag=15pct --tags=15pct --workers=4

# embedding
python -m sleuth init-search-phrases --tags=15pct

# extraction
python -m sleuth extract --input-tag=15pct --tags=15pct --workers=3
