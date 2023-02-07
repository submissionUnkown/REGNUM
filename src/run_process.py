import argparse, os

from data_loader import GeneralDataLoader
from graph_data import StarDogGraph
from parent_ruleminer import RunParseAMIE
from tqdm import tqdm
from runner import run


def parse_input():
    parser = argparse.ArgumentParser(
        description="Enrich rules"
    )
    parser.add_argument(
        "--f_name",
        type=str,
        help="Retailers id. use: --retailers 1 ",
    )
    parser.add_argument("--year", type=int, help="Year of new data. Ex: --year 2022")
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        help="Weeks for new data. Should be separated by comma. Ex: --weeks 3 4 --> data for week 3 and 4",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        help="Category id. Ex: --categories 33D 228D",
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Country for new data in ISO-3166-1 format. Ex: --country FR",
    )
    parser.add_argument(
        "--nb_weeks",
        type=int,
        help="Generate predictions for weeks in the past. Ex: --nb_weeks 6 with"
             " year 2021 and week 3 will give data for [(2020, 50), (2021, 3)]",
        default=0,
    )
    parser.add_argument(
        "--client",
        type=int,
        help="client ID",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_input()
    dl = GeneralDataLoader(path_t=f'{p}/train_dl.tsv', path_numerical_preds=f'{p}/numericals.tsv')
    gr = StarDogGraph(dl, database_name='DB15K_num', force=True, p_save_g=PATH_result + 'graph.ttl')
    miner = RunParseAMIE(data=dl.df, path_rule_miner=PATH_RM,
                         path_save_rules=PATH_result)
    rules = miner.parse()
    dict_all_new_rules_f_score = run(rules[1321:1323], gr, preds)

if __name__ == "__main__":
    main()

"""
python performance_analysis/run_process.py --retailer 96 --year 2022 --categories 036D73 036D28 036D71 --country AU --client 124997


"""
