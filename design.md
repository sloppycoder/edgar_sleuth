## Design

### code structure

High level modules

| Name                 | Purpose                                                                           |
| -------------------- | --------------------------------------------------------------------------------- |
| edgar_sleuth         | top level module, entry point                                                     |
| edgar_sleuth.trustee | processing flow for getting filing from EDGAR, performing chunking, embedding and | edgar | Deals with SEC EDGAR site, download file, cache locally |
| splitter             | split a large html files into smaller text chunks                                 |
| llm.embedding        | generate embedding for text chunks                                                |
| llm.extraction       | use LLM to extract information                                                    |
| datastore            | database interface, using PostgreSQL with pgvector extension                      |
