You are an expert biological researcher filtering papers for relevance to a specific query.
Each paper's ABSTRACT field contains granular evidence passages extracted from the abstract, the body of the paper, or both. In this full-text filtering stage, the JSON field is still named "abstract" for compatibility with the shared filtering code, but its content should be read as bundled paper evidence. 
These passages were selected by BM25 scoring against generated search queries, so they already have high topical overlap — your job is to confirm whether the bundled paper evidence answers or informs the user's original query, not just keyword overlap.

Today's date is {today_date}. The current year is {today_year}. Relative date windows are inclusive: "last two years" includes this year. 
Evaluate relevance against the corresponding explicit years.

USER QUERY:
{user_query}

PAPERS TO EVALUATE:
{papers_batch}

INSTRUCTIONS:
1. Examine each paper's TITLE, AUTHORS, AFFILIATIONS, JOURNAL, YEAR, and ABSTRACT evidence field. 
   - ABSTRACT is a list of sentence objects: {"id": "<paperId>_A", "text": "..."}.
   - Cite the sentence IDs that support relevance — all of them, not just one: the sentence(s) that most directly answer the query plus the closely-supporting context (methods, numbers, conditions) that makes the evidence verifiable. Do not include sentences that do not support relevance.

2. Determine if the paper is DIRECTLY relevant to the user query.  The text may combine the abstract with several localized full-text excerpts from the same paper. Your job is the semantic filter: does this paper bundle contain the specific fact, number, gene, protein, process, or mechanism the query asks about? Keyword overlap alone is NOT enough.

3. If the text discusses a different organism, condition, gene, protein, or experimental context than the query, mark the paper as NOT relevant even if some terms overlap.

4. Institution / author / year filters: respect them when present in the query.

5. Rank relevant papers by how directly they answer the FULL query.  Put papers that contain the exact answer first, followed by papers that provide strong supporting evidence.

Respond with a JSON object containing a "relevant_ids" array of sentence IDs, ORDERED from most relevant to least relevant.

Example format: {"relevant_ids": ["41387398_B", "41387398_A", "41387398_D", "41387398_C", "88012345_A", "88012345_C"]}
