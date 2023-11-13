from io import BytesIO
from pathlib import Path

import pandas as pd
from SPARQLWrapper import CSV, SPARQLWrapper

DATA_PATH = Path("data_processing/data/")


def run_query():
    csv = pd.DataFrame()
    # loop to overcome 10000 query limit
    for i in range(0, 100000, 10000):
        query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX res: <http://dbpedia.org/resource/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX georss: <http://www.georss.org/georss/>

        SELECT DISTINCT ?label ?abs ?point
        WHERE {
                { ?uri dbo:country res:England } UNION
                { ?uri dbo:country res:United_Kingdom } UNION
                { ?uri dbo:country res:Scotland } UNION
                { ?uri dbo:country res:Wales } UNION
                { ?uri dbo:location res:England } UNION
                { ?uri dbo:location res:United_Kingdom } UNION
                { ?uri dbo:location res:Scotland } UNION
                { ?uri dbo:location res:Wales } .

                { ?uri rdf:type dbo:Place }
                  ?uri rdfs:label ?label . FILTER (lang(?label) = 'en')
                  ?uri dbo:abstract ?abs . FILTER (lang(?abs) = 'en')
                  OPTIONAL { ?uri georss:point ?point }
        }
        LIMIT 10000 OFFSET
        """ + str(
            i
        )
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat(CSV)
        results = sparql.query().convert()
        df = pd.read_csv(BytesIO(results), dtype=str)
        csv = csv.append(df)
    return csv


def clean_abs(csv):
    csv = csv.drop_duplicates(subset=["abs"]).reset_index().drop("index", axis=1)
    csv["abs"] = csv["abs"].str.replace(r'"', "")
    csv["abs"] = csv["abs"].str.replace(r"\(.*\)", "")
    csv["abs"] = csv["abs"].str.replace(r"\(|\)", "")
    csv["abs"] = csv["abs"].str.encode("ascii", "ignore").str.decode("ascii")
    csv["abs"] = csv["abs"].str.replace("St.", "Saint", regex=False)
    # removes double spaces
    csv["abs"] = csv["abs"].str.split().str.join(" ")
    # remove south georgia
    csv = csv[~csv["abs"].str.contains("South Georgia")]
    return csv


if __name__ == "__main__":
    wiki_csv = run_query()
    wiki_csv = clean_abs(wiki_csv)

    wiki_csv.to_csv(DATA_PATH / "raw/wiki/wiki_info.csv")
