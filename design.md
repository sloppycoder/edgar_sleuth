## Design

### code structure

High level modules

| Name           | Purpose                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------- |
| edgar_sleuth   | top level module                                                                             |
| edgar          | Deals with SEC EDGAR site, download file, cache locally                                      |
| chunking       | chunk html files into smaller text chunks                                                    |
| llm.embedding  | generate embedding for text chunks                                                           |
| llm.extraction | use LLM to extract information                                                               |
| trustee        | processing flow for getting filing from EDGAR, performing chunking, embedding and extraction |
| datastore      | database interface, using PostgreSQL with pgvector extension                                 |
