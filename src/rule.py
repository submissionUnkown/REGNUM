import pandas as pd
import os
import re
from conf_amie import MIN_HC, MIN_CONF


class Atom:

    def __init__(self, atom_raw, rule_variables):
        self.atom_raw = atom_raw
        self.rule_variables = rule_variables
        self._subject = atom_raw[0]
        self._predicate = atom_raw[1]
        self._objectD = atom_raw[2]

    def __hash__(self):
        return hash((self._subject, self._predicate, self._objectD))

    def __repr__(self):
        return f"{self.subject} {self.predicate} {self.objectD}"

    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.objectD == other.objectD

    @property
    def subject(self):
        return self._subject if self._subject in self.rule_variables else f'<{self._subject}>'

    @property
    def subject_raw(self):
        return self._subject

    @property
    def predicate(self):
        return f'<{self._predicate}>'

    @property
    def predicate_raw(self):
        return self._predicate

    @property
    def objectD(self):
        return self._objectD if self._objectD in self.rule_variables else f'<{self._objectD}>'

    @property
    def object_raw(self):
        return self._objectD

    @property
    def atom(self):
        return ' '.join([self.subject, self.predicate, self.objectD])

    @property
    def atom_variables(self):
        return [self.subject, self.objectD]

    @property
    def atom_raw_variables(self):
        return [v.replace('?', '') for v in self.atom_variables]



class Rule:

    def __init__(self, line):
        self.line = line
        self._init_rule_features(self.line)
        self._build_conclusion_hypotheses()

    def _init_rule_features(self, line):
        parts = line.strip().split("\t")
        self.rule = parts[0]

        self.headCoverage = float(parts[1])
        self.stdConfidence = float(parts[2])
        self.pcaConfidence = float(parts[3])
        self.support = int(parts[4])
        self.bodySize = float(parts[5])
        self.pcaBodySize = float(parts[6])
        self.functionalVariable = str(parts[7])
        self.rule_variables = set(re.findall("\?[a-zA-Z]+", self.rule))
        self.solid_rule_variables = [v.replace('?', '') for v in self.rule_variables]

    def _build_conclusion_hypotheses(self):
        pr = self.rule.split("=>")
        conclusion_raw = pr[1].split("  ")

        conclusion_raw[0] = conclusion_raw[0][1:]
        self.conclusion = Atom(conclusion_raw, self.rule_variables)

        hypotheses_raw = pr[0].split("  ")
        hypotheses = []
        for i in range(0, len(hypotheses_raw) - 1, 3):
            hypotheses.append(Atom(hypotheses_raw[i:i + 3], self.rule_variables))
        self.hypotheses = hypotheses

    def __hash__(self):
        return hash((self.hypotheses, self.conclusion))

    def __repr__(self):
        """
        toWritBenchmarkKGCAMIEe = ""
        for atom in self.hypotheses:
            toWrite += f"{atom} & "
        toWrite = toWrite[:-3] + " => "
        toWrite += str(self.conclusion)
        return toWrite    
        """
        return self.rule

    def __eq__(self, other_rule):
        """ it's best to check if the confidance and head coverage are the same.."""
        if not isinstance(other_rule, Rule):
            return False
        return (self.conclusion == other_rule.conclusion) and (set(self.hypotheses) == set(other_rule.hypotheses))

    @property
    def size_hypotheses(self):
        return len(self.hypotheses)

    @property
    def size_head_r(self):
        return int(self.support / self.headCoverage)

    @property
    def f_score(self):
        return 2*(self.pcaConfidence* self.headCoverage)/(self.pcaConfidence+self.headCoverage)

    def toDict(self):
        return {"hypotheses": self.hypotheses, "conclusion": self.conclusion, "size_hypothese": self.size_hypotheses,
                "headCoverage": self.headCoverage, "stdConfidence": self.stdConfidence,
                "pcaConfidence": self.pcaConfidence,'f_score': self.f_score, "positiveExamples": self.support,
                "bodySize": self.bodySize, "pcaBodySize": self.pcaBodySize,
                "functionalVariable": self.functionalVariable}


"""
def save_sets_rule(root, set_rules):
    if not path.isdir(root + "/save"):
        os.mkdir(root + "/save")
    else:
        shutil.rmtree(root + "/save")
        os.mkdir(root + "/save")
    for set_rule in set_rules:
        set_rules[set_rule].to_csv(root + "/save/" + set_rule + ".tsv")
"""
