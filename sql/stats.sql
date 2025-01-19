\if :{?tag}
\else
   \set tag 'orig225'
\endif

\echo checking results tagged with :'tag'
\echo

\pset footer off

\echo filing_text_chunks
select count(distinct cik) cik,
    count(distinct accession_number) filings,
    count(*) total
from filing_text_chunks;

\echo filing_chunks_embeddings
select count(distinct cik) cik,
    count(distinct accession_number) filings,
    count(*) total
from filing_chunks_embeddings;


\echo trustee_comp_results by filing
select
    count(*) total,
    count(case when n_trustee > 1 then 1 end) yay,
    count(case when n_trustee <= 1 then 1 end) nay
from trustee_comp_results
where :'tag' = any(tags);

\echo trustee_comp_results by company
with results_by_company as (
    select
        cik,
        count(*) n_filings,
        sum(case when n_trustee>=1 then 1 else 0 end) extracted
    from trustee_comp_results
    where :'tag' = any(tags)
    group by cik
)
select
    count(*) as total,
    count(case when extracted >0 then 1 end) extracted,
    count(case when extracted =0 then 1 end) failed
from results_by_company
;
