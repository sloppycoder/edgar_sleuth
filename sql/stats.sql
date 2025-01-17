\pset footer off
\echo filing_text_chunks
select count(distinct cik) cik,
    count(distinct accession_number) accession_number,
    count(*) total
from filing_text_chunks;
\echo filing_chunks_embeddings
select count(distinct cik) cik,
    count(distinct accession_number) accession_number,
    count(*) total
from filing_chunks_embeddings;
\echo trustee_comp_results
select count(*) total,
    count(
        case
            when n_trustee > 1 then 1
        end
    ) yay,
    count(
        case
            when n_trustee <= 1 then 1
        end
    ) nay
from trustee_comp_results;
