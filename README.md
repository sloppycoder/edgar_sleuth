
# EDGAR sleuth
Using LLM to extract information from filings on SEC EDGAR website.


```shell

# load index, thos step loads all records in master_idx
python -m sleuth load-index --input="202*/*"

# manually select and tag records and save them to master_idx_sample
psql -f sql/random_sample.sql

# chunk
python -m sleuth chunk --input-tag=orig225 --tags=orig225 --workers=6

# embedding
python -m sleuth embedding --input-tag=orig225 --tags=orig225 --workers=4

# embedding
python -m sleuth init-search-phrases --tags=orig225

# extraction
python -m sleuth extract --input-tag=orig225 --tags=orig225 --workers=3
