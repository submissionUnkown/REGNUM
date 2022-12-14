from dataclasses import dataclass
from rdflib import Graph, URIRef, Literal
from abc import ABC, abstractmethod
from conf import PATH_SAVE_GRAPH
import os
import stardog
import io
import pandas as pd
import logging
from data_loader import BaseDataLoader
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


@dataclass
class RDFGraph(ABC):
    dataloader: BaseDataLoader

    def __post_init__(self):
        pass

    @abstractmethod
    def populate_graph(self):
        pass

    def query_count(self, query):
        pass

    def query_dataframe(self, query):
        pass

    def query_res_list(self, query):
        pass


@dataclass
class RDFLibGraph(RDFGraph):
    save: bool = False

    def __post_init__(self):
        self.graph = Graph()
        self.populate_graph()
        if self.save:
            self.save_graph_serialize()

    def populate_graph(self):
        lit_row_idx = self.dataloader.df.predicate.isin(self.dataloader.numerical_preds)

        for _, row in tqdm(self.dataloader.df[lit_row_idx].iterrows()):
            s, p, o = row.subject, row.predicate, row.object
            self.graph.add((URIRef(s), URIRef(p), Literal(o)))

        for _, row in tqdm(self.dataloader.df[~lit_row_idx].iterrows()):
            s, p, o = row.subject, row.predicate, row.object
            self.graph.add((URIRef(s), URIRef(p), URIRef(o)))

    def save_graph_serialize(self):
        print("saving...")
        print(PATH_SAVE_GRAPH)
        self.graph.serialize(destination=PATH_SAVE_GRAPH, format='ttl')

    def query_count(self, query):
        relaxed_supp = 0
        for row in self.graph.query(query):
            relaxed_supp = int(row.asdict()["count"])
            break
        return relaxed_supp

    def query_dataframe(self, query):
        graph.query(query)
        dict_list_q_ = [row.asdict() for row in self.graph.query(query)]
        return pd.DataFrame(dict_list_q_)

    def query_res_list(self, query):
        pass


@dataclass
class StarDogGraph(RDFGraph):
    force: bool = False
    database_name: str = 'tmp'

    def __post_init__(self):
        self._init_conn()
        self.populate_graph()

    def _init_conn(self):
        # 'endpoint': 'http://db:5820',
        # 'http://localhost:5820',

        connection_details = {
            'endpoint': 'http://localhost:5820',
            'username': 'admin',
            'password': 'admin'
        }

        with stardog.Admin(**connection_details) as admin:
            if self.database_name in [db.name for db in admin.databases()]:
                admin.database(self.database_name).drop()
            db = admin.new_database(self.database_name)

        self.conn = stardog.Connection(self.database_name, **connection_details)

    def populate_graph(self):
        print(PATH_SAVE_GRAPH)

        if not os.path.exists(PATH_SAVE_GRAPH) or self.force:
            RDFLibGraph(self.dataloader, save=True)
        self.conn.begin()
        self.conn.add(stardog.content.File(PATH_SAVE_GRAPH))
        self.conn.commit()  # commit the transaction

    def query_count(self, query):
        cnt = self.conn.select(query)
        return int(cnt['results']['bindings'][0]['count']['value'])

    def query_dataframe(self, query):
        csv_results = self.conn.select(query, content_type='text/csv')
        return pd.read_csv(io.BytesIO(csv_results))

    def query_res_list(self, query):
        return [item.popitem()[1]["value"] for item in self.conn.select(query)['results']['bindings']]


"""
    path_save_rules: str
    def __post_init__(self):
        self._create_nt_f()
        self._create_graph()

    def _create_nt_f(self):
        data = open(self.path_save_rules + "train.txt", "r")
        f = open(self.path_save_rules + "train.nt", "w")
        for line in data:
            ss = ""
            line_split = line.split("\n")[0].split("\t")
            for i in range(len(line_split)):
                ss += "<http://" + line_split[i] + "> "
            f.write(ss + ". \n")
        data.close()
        f.close()

    def _create_graph(self):
        self.graph = rdflib.Graph()
        self.graph.parse(self.path_save_rules+"train.nt", format="turtle")
"""
