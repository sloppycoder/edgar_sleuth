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

## Process Flow

1. load index ```load-index="year/quarter" --tags=sometag --table-name="sometable"```
2. chunk ```chunk --tag=tag_for_index --tags=tag_for_text_chunks --table-name="sometable"```
3. embedding ```embedding --model="openai/gemini" --dimension=768 --tag=text_chunks_tag --tags=embedding_tag --table-name="sometable"``` (embedding table better incldue text chunk?)
4. initialize search phrase ```init-search --model="openai/gemini" --tag=search_phrase_tag --table-name="sometable"```
5. extraction ```extract --model="openai/gemini" --search-phrase-table="some_table" --search-phrase-tag="gemini768" --embedding-table="table_name" --embedding-tag="tag" --output-table="trust_comp" --tags=comp_result_tag```
6. export result ```export --result-table="" --result-tag=x --embedding-table="" --embedding-tag=x --index-table="" --index-tag ".."

--tags: write tags
--output-table="..."
