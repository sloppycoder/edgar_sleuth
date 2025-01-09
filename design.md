## Design

### code structure

package structure

| Name                  | Purpose                                                                           |
| --------------------- | --------------------------------------------------------------------------------- |
| sleuth                | top level module, entry point                                                     |
| sleuth.trustee        | processing flow for getting filing from EDGAR, performing chunking, embedding and |
| sleuth.edgar          | Deals with SEC EDGAR site, download file, cache locally                           |
| sleuth.splitter       | split a large html files into smaller text chunks                                 |
| sleuth.llm.embedding  | generate embedding for text chunks                                                |
| sleuth.llm.extraction | use LLM to extract information                                                    |
| sleuth.datastore      | database interface, using PostgreSQL with pgvector extension                      |

## record tagging

```tags``` column uses PostgreSQL array type ```TEXT[]```, so each record can have multiple tags for flexiblity. In generate multiple tags can be tags to a record during save, e.g. ```chunk_filing``` and ```get_embedding```, but only one tags can be specified when reading, e.g. ```...```.

Also, when performing embedding and extraction, the same tag must exist on both text chunks table and embedding table in order to correlate.
