from rule import Rule, Atom
from subprocess import check_output
import pandas as pd
import os


# from conf_amie import HTR, CONST


class RunParseAMIE:
    def __init__(self, data, path_amie3, path_save_rules, num_atoms, min_conf, min_pca_conf, min_hc, force=False, const=False, default=True):
        self.data: pd.DataFrame = data
        self.path_amie3: str = path_amie3
        self.path_save_rules: str = path_save_rules

        self.num_atoms: int = num_atoms
        self.min_conf: float = min_conf
        self.min_pca_conf: float = min_pca_conf

        self.min_hc: float = min_hc
        self.force_create_rule = force
        self.const: bool = const
        self.default: bool = default

        self.mineRules = True if not os.path.exists(self.path_save_rules + "/amierules.txt") else False
        self._data_input_amie()

        if self.mineRules or self.force_create_rule:
            self.run_amie()
            self.rules_mined_f = self.res_rules_raw.decode("utf-8").split("\n")
        else:
            print("loading from file...")
            self.rules_mined_f = open(self.path_save_rules + "/amierules.txt", "r")

    def _data_input_amie(self):
        # df_num_mine_rules_on = self.dataloader.df_not_num.copy()
        df_num_mine_rules_on = self.data.copy()
        # df_num_mine_rules_on = df_num_mine_rules_on.apply(lambda x: '<http:'+x+'>')
        df_num_mine_rules_on.to_csv(self.path_save_rules + "/train.txt", sep="\t", header=False, index=False)

    def run_amie(self):

        if self.const:
            print('mining rules with const')
            # run_command_amie = f'java -jar {self.path_amie3} {self.path_save_rules + "train_not_num.txt"} {"-htr" + HTR} -const {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-maxad " + str(self.num_atoms)}'
            run_command_amie = f'java -jar {self.path_amie3} {self.path_save_rules + "train.txt"} -const {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-minpca " + str(self.min_pca_conf)} {"-maxad " + str(self.num_atoms)}'
        else:
            run_command_amie = f'java -jar {self.path_amie3} {self.path_save_rules + "train.txt"} {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-minpca " + str(self.min_pca_conf)} {"-maxad " + str(self.num_atoms)}'
        if self.default:
            if self.const:
                print('default true, const true')
                run_command_amie = f'java -jar {self.path_amie3} {self.path_save_rules + "train.txt"} -const'

            else:
                run_command_amie = f'java -jar {self.path_amie3} {self.path_save_rules + "train.txt"}'

        print(f'default is {self.default} {run_command_amie}')

        self.res_rules_raw = check_output(run_command_amie, shell=True)

        # print(len(self.res_rules_raw))
        file = open(self.path_save_rules + "/amierules.txt", "w")
        for line in self.res_rules_raw.decode("utf-8").split("\n"):
            file.write(str(line) + "\n")
        file.close()

    def parse_amie(self):
        rules = []
        for line in self.rules_mined_f:
            if (line != "") and (line[0] == "?"):
                rules.append(Rule(line))

        print("number of mined rules:", len(rules))
        return rules
