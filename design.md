## Design

### code structure

package structure.

| Name                  | Purpose                                                                           |
| --------------------- | --------------------------------------------------------------------------------- |
| sleuth                | top level module, entry point                                                     |
| sleuth.processor      | processing logic, use tools provided by other modules to process                  |
| sleuth.trustee        | processing flow for getting filing from EDGAR, performing chunking, embedding and |
| sleuth.edgar          | Deals with SEC EDGAR site, download file to cache locally                         |
| sleuth.datastore      | database interface, using PostgreSQL with pgvector extension                      |
| sleuth.splitter       | tools that conver html to text and split a large text file into smaller chunks    |
| sleuth.llm.embedding  | generate embedding for text chunks                                                |
| sleuth.llm.extraction | use LLM to extract information                                                    |

simple rules of separation

* tools modules, e.g. ```llm```, ```splitter``` only provides tool for processing, it should not deal with database directly
* ```datastore``` module should the only interface to the database. SQLK statement should be be spilled into other modules.
* ```edgar``` deals with EDGAR website. only side effect it produces is writing files to the local cache.
* ```processor``` is the glue. Use the tools provided by ```llm```,  ```splitter```and ```edgar``` and interact with the ```datastore```. It also contains the control flows.
* ```trustee``` should contain only Trustee related functionality.

## record tagging

```tags``` column uses PostgreSQL array type ```TEXT[]```, so each record can have multiple tags for flexiblity. In generate multiple tags can be tags to a record during save, e.g. ```chunk_filing``` and ```get_embedding```, but only one tags can be specified when reading, e.g. ```...```.

Also, when performing embedding and extraction, the same tag must exist on both text chunks table and embedding table in order to correlate.
