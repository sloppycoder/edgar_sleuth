
# EDGAR sleuth
Using LLM to extract information from filings on SEC EDGAR website.


```shell

# load index
python -m sleuth load-index --input="202*/*" --tags=orig225

# chunk
python -m sleuth chunk --input-tag=orig225 --tags=orig225 --workers=6

# embedding
python -m sleuth embedding --input-tag=orig225 --tags=orig225 --workers=4

# extraction
python -m sleuth extract --input-tag=orig225 --tags=orig225 --workers=3
