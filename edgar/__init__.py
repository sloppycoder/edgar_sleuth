import logging
import os
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Lee Lynn (hayashi@yahoo.com)"
EDGAR_BASE_URL = "https://www.sec.gov/Archives"

# Document tag contents usually looks like this,
# FILENAME and DESCRIPTION are optional
#
# <DOCUMENT>
# <TYPE>stuff
# <SEQUENCE>stuff
# <FILENAME>stuff
# <TEXT>
# Document 1 - file: stuff
#
# </DOCUMENT>
# the regex below tries to parse
# doc element in index-headers.html
_doc_regex = re.compile(
    r"""<DOCUMENT>\s*
<TYPE>(?P<type>.*?)\s*
<SEQUENCE>(?P<sequence>.*?)\s*
<FILENAME>(?P<filename>.*?)\s*
(?:<DESCRIPTION>(?P<description>.*?)\s*)?
<TEXT>
(?P<text>.*?)
</TEXT>""",
    re.DOTALL | re.VERBOSE | re.IGNORECASE,
)

# in SEC_HEADER
# FILED AS OF DATE:		20241017
_date_filed_regex = re.compile(r"FILED AS OF DATE:\s*(\d{8})", re.IGNORECASE)


class InvalidFilingExceptin(Exception):
    pass


class SECFiling:
    def __init__(self, cik: str, idx_filename: str):
        # sometimes a same filename is used by several CIKs
        # filename as in master.idx
        # e.g. edgar/data/106830/0001683863-20-000050.txt
        self.cik = cik
        self.idx_filename = idx_filename
        self.accession_number = idx_filename2accession_number(idx_filename)
        # idx filename for the filing index-headers file
        self.index_html_path = _index_html_path(idx_filename)
        self.index_headers_path = self.index_html_path.replace(
            "-index.html", "-index-headers.html"
        )

        (self.sec_header, self.date_filed, self.documents) = self._read_index_headers()
        logger.debug(f"initialized SECFiling({self.cik},{self.idx_filename})")

    def get_doc_by_type(self, doc_type: str) -> list[str]:
        """
        Reads the contents of documents of a specific type from the filing.

        Args:
            doc_type (str): The type of document to read (e.g., "485BPOS").

        Returns:
            list[str]: A list of filenames that matches the doc_type

        Raises:
            InvalidFilingExceptin: If the specified document type is not found in the
            filing or if the document path cannot be determined.
        """
        # Get the paths of documents of the specified type
        paths = [doc["filename"] for doc in self.documents if doc["type"] == doc_type]
        if paths is None or paths == []:
            raise InvalidFilingExceptin(
                f"{self.idx_filename} does not contain a {doc_type} document"
            )

        if len(paths) > 1:
            raise InvalidFilingExceptin(
                f"{self.idx_filename} has more than 1 document of type {doc_type}"
            )

        return [str(Path(self.index_headers_path).parent / path) for path in paths]

    def _read_index_headers(self) -> tuple[str, str, list[dict[str, Any]]]:
        """read the index-headers.html file and extract the sec_header and documents"""

        content = edgar_file(self.index_headers_path)
        if not content:
            raise InvalidFilingExceptin(f"Unable to read {self.index_headers_path}")

        soup = BeautifulSoup(content, "html.parser")
        # each index-headers.html file contains a single <pre> tag
        # inside there are SGML content of meta data for the filing
        pre = soup.find("pre")
        if pre is None:
            logger.debug(f"No <pre> tag found in {self.index_headers_path}")
            return "", "", []

        pre_soup = BeautifulSoup(pre.get_text(), "html.parser")

        sec_header = pre_soup.find("sec-header")
        sec_header_text = ""
        date_filed = ""
        if sec_header:
            sec_header_text = str(sec_header)
            match = _date_filed_regex.search(sec_header_text)
            if match:
                digits = match.group(1)
                date_filed = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"

            documents = []
            for doc in pre_soup.find_all("document"):
                match = _doc_regex.search(str(doc))

                if match:
                    doc_info = {
                        "type": match.group("type"),
                        "sequence": match.group("sequence"),
                        "filename": match.group("filename"),
                    }
                    documents.append(doc_info)

            return sec_header_text, date_filed, documents

        logger.info(f"No sec-header found in {self.index_headers_path}")

        return "", "", []


def idx_filename2accession_number(idx_filename: str) -> str:
    accession_number = ""
    if idx_filename.endswith(".txt"):
        accession_number = idx_filename.split("/")[-1].split(".")[0]
    elif idx_filename.endswith(".htm"):
        part = idx_filename.split("/")[-2]
        accession_number = f"{part[:10]}-{part[10:12]}-{part[12:]}"
    return accession_number


def edgar_file(idx_filename: str, cached_only: bool = False) -> str | None:
    """
    e.g. edgar/data/123456/0001234567-21-000123.txt
    """
    cache_path = config.cache_path
    logging.debug(
        f"Reading file({idx_filename}) from {cache_path}. cached_only={cached_only})"
    )

    if cache_path.startswith("gs://"):
        raise RuntimeError("gs:// is not supported for now")

    if not Path(cache_path).is_dir():
        raise RuntimeError(f"Cache path {config.cache_path} does not exist")

    cache_file_path = Path(cache_path) / idx_filename

    if not cached_only and not cache_file_path.is_file():
        _download_file(f"{EDGAR_BASE_URL}/{idx_filename}", cache_file_path)

    if cache_file_path.is_file():
        with open(cache_file_path, "r") as f:
            return f.read()
    else:
        return None


def _index_html_path(idx_filename: str) -> str:
    """
    convert a filename from master.idx filename to -index.html
    e.g.
    edgar/data/1007571/000119312524109215/0001193125-24-109215-index.html
    """
    filepath = Path(idx_filename)
    basename = filepath.name.replace(".txt", "")
    return str(filepath.parent / basename.replace("-", "") / f"{basename}-index.html")


def _download_file(
    url: str,
    output_path: Path,
    user_agent: str = DEFAULT_USER_AGENT,
) -> bool:
    response = requests.get(url, headers={"User-Agent": user_agent})
    if response.status_code == 200:
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "wb") as file:
            file.write(response.content)
        logging.debug("Downloaded {url} and saved to {localPath}")
        return True
    else:
        # TODO: add retrying logic and etc to make it more robust
        logging.debug(f"Failed to download from {url}: {response.status_code}")
        return False
