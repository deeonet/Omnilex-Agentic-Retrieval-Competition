from our_pipeline.search_tools import LawSearchTool
from our_pipeline.bm25.corpus import get_or_build_index
from our_pipeline.constants import (
    CONFIG,
    FORCE_REBUILD_INDICES,
    LAWS_CSV,
    LAWS_INDEX_PATH,
)
from our_pipeline.search_tools import LawSearchTool
import re
import pandas as pd


laws_index = get_or_build_index(
    name="laws",
    csv_path=LAWS_CSV,
    index_path=LAWS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    # max_rows=10000  # Uncomment to test with smaller corpus
)

# Create tools
law_tool = LawSearchTool(
    index=laws_index,
    top_k=CONFIG["top_k_laws"],
    max_excerpt_length=300,
)

query = "May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault and theft of a courier satchel containing, inter alia, €5,600—was remanded by an order dated 18 October 2024 for a maximum period up to 15 January 2025, the prosecutor sought an extension on 10 December 2024 primarily citing a concrete risk that the detainee would influence witnesses or tamper with evidence and a risk of reoffending, while the detainee opposed the extension on the ground that most witnesses have already been interviewed, the investigative steps still pending are essentially technical (phone data extraction, CCTV image analysis, and bank record checks), his release would therefore not jeopardize the inquiry, and the alleged victim has withdrawn the complaint—i.e. does the asserted concrete risk of collusion and considerations of proportionality justify a three‑month prolongation in these circumstances?"

result = law_tool(query)

citations = re.findall(r'-\s*(.+?)\s*:', result)

print(citations)


y_pred = ['Art. 221 Abs. 1 StPO', 'Art. 226 Abs. 4 StPO', 'Art. 227 Abs. 4 StPO', 'Art. 226 Abs. 5 StPO', 'Art. 226 Abs. 3 StPO', 'Art. 341 Abs. 3 StPO', 'Art. 229 Abs. 3 StPO', 'Art. 227 Abs. 7 StPO', 'Art. 342 Abs. 2 StPO', 'Art. 361 Abs. 4 StPO', 'Art. 345 StPO', 'Art. 342 Abs. 1bis StPO', 'Art. 341 Abs. 1 StPO', 'Art. 343 Abs. 2 StPO', 'Art. 343 Abs. 1 StPO', 'Art. 227 Abs. 1 StPO', 'Art. 227 Abs. 6 StPO', 'Art. 342 Abs. 4 StPO', 'Art. 226 Abs. 1 StPO', 'Art. 229 Abs. 1 StPO', 'Art. 225 Abs. 2 StPO', 'Art. 341 Abs. 2 StPO', 'Art. 225 Abs. 4 StPO', 'Art. 224 Abs. 2 StPO', 'Art. 225 Abs. 3 StPO', 'Art. 228 Abs. 3 StPO', 'Art. 228 Abs. 5 StPO', 'Art. 224 Abs. 3 StPO', 'Art. 343 Abs. 3 StPO', 'Art. 342 Abs. 1ter StPO', 'Art. 227 Abs. 2 StPO', 'Art. 342 Abs. 3 StPO', 'Art. 227 Abs. 3 StPO', 'Art. 225 Abs. 5 StPO', 'Art. 186 Abs. 2 StPO', 'Art. 226 Abs. 2 StPO', 'Art. 346 Abs. 1 StPO', 'Art. 228 Abs. 1 StPO', 'Art. 225 Abs. 1 StPO', 'Art. 344 StPO']



y_true = ['Art. 221 Abs. 1 StPO', 'Art. 140 Abs. 1 StGB', 'Art. 396 Abs. 1 StPO', 'Art. 222 StPO', 'Art. 393 Abs. 1 StPO', 'Art. 382 Abs. 1 StPO', 'Art. 385 Abs. 1 StPO', 'Art. 221 Abs. 2 StPO', 'Art. 227 Abs. 1 StPO', 'Art. 212 Abs. 3 StPO', 'Art. 390 Abs. 2 StPO', 'Art. 422 Abs. 1 StPO', 'Art. 422 Abs. 2 StPO', 'Art. 428 Abs. 1 StPO', 'Art. 135 Abs. 4 StPO', 'Art. 100 Abs. 1 BGG', 'Art. 135 Abs. 3 StPO', 'Art. 37 Abs. 1 StBOG', 'Art. 39 Abs. 1 StBOG']

correct = [i for i in y_pred if i in y_true]
# print(correct)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- Testing for each query in val.csv ---------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# gold_standard = pd.read_csv('data/raw/val.csv')
# our_submission = pd.read_csv(r"output/submission.csv")

# for i in range(10):

#     citations_for_first_gold = gold_standard.iloc[i]['gold_citations']
#     citations_for_first_gold = citations_for_first_gold.split(';')

#     citations_for_first_our = our_submission.iloc[i]['predicted_citations']
#     if pd.notna(citations_for_first_our):
#         citations_for_first_our = citations_for_first_our.split(';')
#     else:
#         citations_for_first_our = []

#     correct = [i for i in citations_for_first_our if i in citations_for_first_gold]
#     print(f'val_00{i}: {correct}')





