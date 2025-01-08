import pytest

from edgar import (
    SECFiling,
    _index_html_path,
    edgar_file,
    parse_idx_filename,
)


def test_idx_filename2index_html_path():
    assert (
        _index_html_path("edgar/data/1035018/0001193125-20-000327.txt")
        == "edgar/data/1035018/000119312520000327/0001193125-20-000327-index.html"
    )


def test_edgar_file():
    # this file exists in cache
    dummy_file = edgar_file("dummy.txt")
    assert dummy_file and "unit test" in dummy_file

    # this file does not exist in cache
    dummy_file_2 = edgar_file("dummy_2.txt", cached_only=True)
    assert dummy_file_2 is None


def test_parse_idx_filename():
    assert ("1035018", "0001193125-20-000327") == parse_idx_filename(
        "edgar/data/1035018/0001193125-20-000327.txt"
    )
    with pytest.raises(ValueError, match="an unexpected format"):
        parse_idx_filename("edgar/data/blah.txt")


def test_parse_485bpos_filing():
    filing = SECFiling(idx_filename="edgar/data/1002427/0001133228-24-004879.txt")
    filing_html_path = filing.get_doc_by_type("485BPOS")[0]
    contents = edgar_file(filing_html_path)

    assert filing.cik == "1002427" and filing.date_filed == "2024-04-29"
    assert filing.accession_number == "0001133228-24-004879"
    assert len(filing.documents) == 26
    assert "msif-html7854_485bpos.htm" in filing_html_path
    assert contents
    assert "N-1A" in contents
