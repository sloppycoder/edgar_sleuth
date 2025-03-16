
# EDGAR sleuth
Using LLM to extract information from filings on SEC EDGAR website.


```shell

# load index, thos step loads all records in master_idx
python -m sleuth load-index "2024/*"

# manually select and tag records and save them to master_idx_sample
# currently it samples 10% of all companies filed between 2023-2024
# and tag them "10pct"
psql <database> -f sql/sample.sql

# chunk
python -m sleuth chunk --tag=10pct  --workers=8  \
  --table idx=master_idx_sample \
  --table text=filing_text_chunks

# embedding
python -m sleuth embedding --tag=10pct --workers=4 \
  --model gemini --dimension 768 \
  --table text=filing_text_chunks \
  --table embedding=filing_chunks_embeddings


# embedding
python -m sleuth init-search-phrases --search-tag=group1 \
  --model gemini --dimension 768 \
  --table search=search_phrase_embeddings


# extraction
python -m sleuth extract --tag=10pct --search-tag=group1 --result-tag=batch890 --workers=3 \
  --table idx=master_idx_sample \
  --table text=filing_text_chunks \
  --table embedding=filing_chunks_embeddings \
  --table search=search_phrase_embeddings \
  --table result=trustee_comp_results \
  --model gemini

# export extraction result
python -m sleuth export --tag=10pct --result-tag=batch890 \
  --table idx=master_idx_sample \
  --table text=filing_text_chunks \
  --table embedding=filing_chunks_embeddings \
  --table result=trustee_comp_results \
  --output result.jsonl
