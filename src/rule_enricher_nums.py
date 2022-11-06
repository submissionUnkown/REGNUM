from rule import Rule
from conf_amie import MIN_CONF, MIN_HC
from graph_query import query_rule_num_pred, query_rule_add_base, query_rule_add_df
from binning import OptimalBinningFitStats, MDLPFitStats, TreeBinnig, UnsupervisedBinning, CustomMDLPFitStats
from quality_measures import MeasureEnriched, MeasureExistential
from assets import get_sequences
import numpy as np
from matplotlib import pyplot as plt
from time import time


class RuleEnricherNumerical:

    def __init__(self, rule, RDFGraph, numerical_preds, minhc=MIN_HC,
                 binningtechniques=['mdlp', 'opt'], merge_intervals=False, merge_all_intervals=False,
                 add_num_pred_grounded_not_functional=False, only_margin_rules=False, debug_mode=False):
        self.rule: Rule = rule
        self.graph = RDFGraph
        self.numerical_preds: list = numerical_preds
        self.only_margin_rules = only_margin_rules
        self.minhc = minhc
        self.binningtechniques = binningtechniques
        self.margin_conf = 0.2  # at least by % better than parent conf
        self.margin_hc = 0.2  # at most % worse than hc
        self.min_conf_pr = self.compute_min_conf()
        self.merge_bc_intervals: bool = merge_intervals
        self.merge_all_intervals: bool = merge_all_intervals
        self.even_not_func_enrich_g: bool = add_num_pred_grounded_not_functional
        self.debug_mode: bool = debug_mode
        self.dict_pred_var_df_debug = {}  # only for debug
        self._base_query_df, self._rep_var = query_rule_add_base(self.rule, self.even_not_func_enrich_g)
        self.min_pos_example_satisfy_minhc = self._compute_min_pos_example_satisfy_minhc()
        self.dict_bin_new_rules = {bin_teq: [{"parent_rule": self.rule}] for bin_teq in binningtechniques}  # result
        self.suitable_var_to_pred, self.var_pred_to_relaxed_supp = self._compute_var_to_pred_relaxed_supp()
        self.imbalanced_perc = []
        self.rule_enricher()

    def _compute_min_pos_example_satisfy_minhc(self):
        regular_pos_ex = self.minhc * self.rule.size_head_r
        hard_pos_ex = (1 - self.margin_hc) * self.rule.headCoverage * self.rule.size_head_r

        if self.debug_mode:
            print('self.minhc:', self.minhc)
            print('self.rule.size_head_r:', self.rule.size_head_r)
            print('regular_pos_ex: ', regular_pos_ex)
            print('hard_pos_ex: ', hard_pos_ex)
            print('max:', max(hard_pos_ex, regular_pos_ex))

        if not self.only_margin_rules:
            return regular_pos_ex
        else:
            return max(hard_pos_ex, regular_pos_ex)

    def compute_min_conf(self):
        if not self.only_margin_rules:
            self.min_conf_pr = self.rule.pcaConfidence
        else:
            self.min_conf_pr = (1 + self.margin_conf) * self.rule.pcaConfidence
        return self.min_conf_pr

    def _compute_var_to_pred_relaxed_supp(self):
        var_to_pred_list = {}
        var_pred_to_relaxed_supp = {}

        for var in self.rule.rule_variables:
            var_to_pred_list[var] = []
            var_pred_to_relaxed_supp[var] = {}
            for pred in self.numerical_preds:
                plaus, relaxed_supp = self._checker_relaxed_supp(pred, var)
                if plaus:  # relaxed head coverage condition not satisfied...
                    var_to_pred_list[var].append(pred)
                    var_pred_to_relaxed_supp[var][pred] = relaxed_supp

        return var_to_pred_list, var_pred_to_relaxed_supp

    def rule_enricher(self):

        # TODO: if self.suitable_var_to_pred has all its values as an empty list, skip
        if not self.even_not_func_enrich_g:  # check for the functionality of the rule. If it's grounded, skip it
            if not self._check_not_func_grounded():

                if self.debug_mode:
                    print("functional grounded not proceeding...")
                return None

        for var, preds in self.suitable_var_to_pred.items():
            if not preds:
                continue
            self.add_numerical_pred_to_rule(var, preds)

    def _check_not_func_grounded(self):
        if self.rule.functionalVariable == self.rule.conclusion.subject:
            if self.rule.conclusion.objectD not in self.rule.rule_variables:
                return False
        else:
            if self.rule.conclusion.subject not in self.rule.rule_variables:
                return False
        return True

    def _checker_relaxed_supp(self, pred, var):
        # query - count --> relaxed_supp
        qs = query_rule_num_pred(self.rule, pred, var)

        #######
        # its ok
        relaxed_supp = self.graph.query_count(qs)

        plaus = self._checker_satisfy_minhc(relaxed_supp)
        if self.debug_mode and plaus:
            print(f"the predicate... : \n {pred}")
            # print(f"the query to compute the relaxed supp: \n {qs}")
            print(f"relaxed supp: \n {relaxed_supp}")
            print("ok..\n")
        return plaus, relaxed_supp

    def _create_dict_df_pred_var(self, pred, var, feat_pd_):
        if pred not in self.dict_pred_var_df_debug:  # TODO: remove this later
            self.dict_pred_var_df_debug[pred] = {}
        self.dict_pred_var_df_debug[pred][var] = feat_pd_

    def _checker_better_that_parent_confidence(self, event_rate):
        # not_event_interval =  binning.n_nonevent[interval_inx]
        # event_interval =  binning.n_event[interval_inx]
        # not_event_interval/(not_event_interval+ event_interval)
        return True if self.rule.pcaConfidence > event_rate else False

    def _merge_biggest_consecutive_intervals_binning_dict(self, binning, df,
                                                          binning_dict):  # TODO: fix this! buggy, include_exclude
        intervals_seq = get_sequences(sorted(binning_dict.keys()))  # [[1,2,3], [8,9]]
        inc_ex_set = set()
        for intervals_ in intervals_seq:
            for interval_ in intervals_:
                inc_ex_set.add(binning_dict[interval_]['include_exclude'])
        if len(inc_ex_set) > 1:
            if self.debug_mode:
                print("include exclude mixed not possible for merging")
            return binning_dict
        include_exclude = list(inc_ex_set)[0]

        intervals_seq = [seq for seq in intervals_seq if len(seq) > 1]
        if not intervals_seq:
            if self.debug_mode:
                print("binning dict not empty but no consecutive intervals seq")
            return binning_dict
        tmp_ = binning_dict[list(binning_dict.keys())[0]]
        var, pred = tmp_["var_num"], tmp_["pred"]
        for interval_seq in intervals_seq:
            b_sort_idx_l = binning.sort_indx.tolist()
            # print("binning.bins_intervals[interval_seq[0]][0]",
            #      binning.bins_intervals[b_sort_idx_l.index(interval_seq[0])][0])
            begin, end = binning.bins_intervals[b_sort_idx_l.index(interval_seq[0])][0], \
                         binning.bins_intervals[b_sort_idx_l.index(interval_seq[-1])][1]

            ## recalcualte event rate
            _, _, event_rate_int = binning.recompute_stats_new_interval(interval_seq[0],
                                                                        interval_seq[-1])

            dict_res_update = self.find_rule_binning_interval(df, pred, var, begin, end, include_exclude)

            if dict_res_update is not None:
                if self.debug_mode:
                    print("dict_res_update is not None", dict_res_update)
                binning_dict[tuple(interval_seq)] = dict_res_update
        return binning_dict

    def _merge_all_intervals_p_var(self):
        ## TODO: based on the binning dict after all the loop is over
        return None

    def _find_existential_num_(self, feat_pd_, var, pred):

        pca_body_size_existential = feat_pd_.shape[0]
        support_existential = self.var_pred_to_relaxed_supp[var][pred]
        conf_existential = support_existential / pca_body_size_existential
        has_existential = True if conf_existential > self.min_conf_pr else False
        return has_existential, pca_body_size_existential, support_existential

    def add_numerical_pred_to_rule(self, var, preds):

        step1 = time()

        step2 = time()

        feat_pd_all, suitable_pred_str = self.create_df_from_query(preds, var)
        for pred, feat_pd_ in feat_pd_all.groupby(suitable_pred_str):
            if self.debug_mode:
                print(feat_pd_)
                print("imbalanced ?")
                sss = feat_pd_.groupby('label').count()
                sss = sss.reset_index(drop=True)
                print(sss)
                ttt = sss.loc[1]['feat']
                fff = sss.loc[0]['feat']
                self.imbalanced_perc.append(ttt / (ttt + fff))

            step3 = time()

            # feat_pd_.sort_values(by="feat", inplace=True)  # for visualization
            # self.plot_scatt(fea.size_head_rt_pd_)  # for visualization
            if self.debug_mode:
                self._create_dict_df_pred_var(pred, var, feat_pd_)
            ###TODO: if reached here with pred and var, possible that bining will be done on 2 or more features. keep them and make a call

            for binning_mode in self.binningtechniques:
                # TODO : try adding existial here..
                has_existential, pca_body_size_existential, support_existential = \
                    self._find_existential_num_(feat_pd_, var, pred)

                if has_existential:
                    if self.debug_mode:
                        print(has_existential, 'EEEEEEEEEEEE')
                        print(pca_body_size_existential, support_existential)

                    m_e = MeasureExistential(rule=self.rule, graph=self.graph, base_q=self._base_query_df,
                                             pred=pred, var_num=var,
                                             pca_body_size_pre_computed=pca_body_size_existential,
                                             support_pre_computed=support_existential, verbose=False)
                    _dict = m_e.toDict()
                    _dict["interval"] = -2
                    self.dict_bin_new_rules[binning_mode].append(_dict)

                binning = self.start_binning(feat_pd_, binning_mode)
                step4 = time()

                binning_dict = self.find_possible_rules_binning(binning, feat_pd_, pred, var)
                step42 = time()

                if not binning_dict:
                    continue

                # TODO: to have the rules that are consecutive (in terms of interval), take care here, and recalculate their quality measures
                if self.merge_bc_intervals and len(binning_dict) > 1:
                    binning_dict = self._merge_biggest_consecutive_intervals_binning_dict(binning, feat_pd_,
                                                                                          binning_dict)

                ## TODO: instead of merging the biggest consecituve, merge all the intervals
                # if self.merge_all_intervals and len(binning_dict) > 1:
                #    self._merge_all_intervals_p_var()

                # for interval, binning_dict_interval in binning_dict.items():
                #    self.rule_loader.write_numerical_pred_rules(self.rule, binning_dict_interval)
                step45 = time()

                for interval, _dict in binning_dict.items():
                    _dict["interval"] = interval
                    self.dict_bin_new_rules[binning_mode].append(_dict)

                if self.debug_mode:
                    try:
                        print(f'binning_mode = {binning_mode}')
                        print(binning.build_table())
                        self.plot_scatt(feat_pd_, binning.splits)
                    except:
                        print('CANNOT DRAW PROBABLY ALL 0 which means existential')

                    step5 = time()

                    print('step2 - step1:', step2 - step1)
                    print('step3 - step25', step3 - step2)
                    print('step4 - step3', step4 - step3)
                    print('step42 - step4', step42 - step4)
                    print('step45 - step42', step45 - step42)
                    print("\n")

        # for binning_mode in self.binningtechniques: #TODO: what is this for loop for? existential does not need binning

    def find_rule_binning_interval(self, df, pred, var, begin, end, include_exclude):

        new_supp = self.compute_new_supp_from_df(df, begin, end, include_exclude)
        if not self._checker_satisfy_minhc(new_supp):
            if self.debug_mode:
                print("did not pass the thr for minhc")
            return None
        if self.debug_mode:
            print("MINHC PASSED OK!!!!")

        quality_measure_new_rule = MeasureEnriched(rule=self.rule, graph=self.graph, base_q=self._base_query_df,
                                                   pred=pred, var_num=var,
                                                   begin_interval=begin, end_interval=end,
                                                   include_exclude=include_exclude,
                                                   support_pre_computed=new_supp, verbose=False)

        if not self._checker_new_conf_better_than_parent_confidence(quality_measure_new_rule.pca_confidence):
            if self.debug_mode:
                print('DID NOT PASS quality_measure_new_rule.pca_confidence', quality_measure_new_rule.pca_confidence)

            return None

        if self.debug_mode:
            print('EVERYTHING OK PASS NEW RULE pca_confidence OK!!!!')

        # new rule

        return quality_measure_new_rule.toDict()

    def find_possible_rules_binning(self, binning, df, pred, var):
        binning_dict = {}
        if self.debug_mode:
            print(f'considering pred: {pred}, var: {var}')
        sorted_event_rate = binning.event_rate
        # TODO: can potentially remove sorting the bins now, cause no earlier pruning can be done
        for _indx, event_rate in enumerate(sorted_event_rate):
            if self.debug_mode:
                print('_indx, event_rate', _indx, event_rate)
            passed, include_exclude = self._checker_prune_better_than_parent_confidence(event_rate)
            if not passed:  # TODO: not sure if this rule causes pruning when should not
                if self.debug_mode:
                    print("did not pass the thr for parent conf...")
                continue

            if self.debug_mode:
                print("!!!!pass the thr for minhc")

            interval_indx = binning.sort_indx[_indx]
            begin, end = binning.bins_intervals[_indx][0], binning.bins_intervals[_indx][1]
            dict_res = self.find_rule_binning_interval(df, pred, var, begin, end, include_exclude)

            if dict_res is not None:
                binning_dict[interval_indx] = dict_res
            if len(binning_dict) == 10:
                break

        if self.debug_mode:
            if len(binning_dict) > 0:
                print('returning  binning_dict ok!!!!')

        return binning_dict

    def _checker_prune_better_than_parent_confidence(self, event_rate):
        # event_rate = opt.n_event[interval_indx]/ (opt.Non-event[interval_indx]+ opt.n_event[interval_indx])
        # is this really correct ? to do pruning using the event rate given that maximum reduction from
        # support is opt.n_event and maximum reduction from denom is opt.n_event+opt.Non-event --> gand nazanim bikhod prune
        # konim?

        if event_rate > self.min_conf_pr:  ## TODO: maybe the margin of improvement should be higher (check notable thing)
            if self.debug_mode:
                print(
                    f"exclude, and compare of {event_rate} done with {self.min_conf_pr} instead of {self.rule.pcaConfidence}")

            return True, 'exclude'

        if 1 - event_rate > self.min_conf_pr:
            if self.debug_mode:
                print(
                    f"include, and compare of {1 - event_rate} done with {self.min_conf_pr} instead of {self.rule.pcaConfidence}")
            return True, 'include'

        return False, ''

    def _checker_new_conf_better_than_parent_confidence(self, new_pca_conf):
        return True if new_pca_conf > self.rule.pcaConfidence else False

    def compute_new_supp_from_df(self, df, begin, end, include_exclude, by="feat", grp_by="label"): #TODO" cehck if drop dup needed
        df_t = None
        if include_exclude == 'exclude':
            df_t = df[df[grp_by] == 0][~df[by].between(begin, end, inclusive="left")]
        elif include_exclude == 'include':  # TODO not sure check armita
            df_t = df[df[grp_by] == 0][df[by].between(begin, end)]
        else:
            raise Exception(f'bug, {include_exclude} should be exclude or include')

        new_supp_ = df_t.drop_duplicates(subset=self.rule.conclusion.atom_raw_variables).shape[0]

        return new_supp_

    def _checker_satisfy_minhc(self, new_supp):
        return True if new_supp > self.min_pos_example_satisfy_minhc else False

    def create_df_from_query(self, preds, var):
        # TODO: there might be a bug here, make sure 0/1 label is based on the functionality
        # ## get the functionality based on the parent rule and define the labels accordingly

        q_out_str, suitable_pred_str = query_rule_add_df(self._base_query_df, self.rule.rule_variables, preds, var)
        df = self.graph.query_dataframe(q_out_str)
        # print

        # TODO: there might be a bug here. Do we check weather all variables or not
        # PlaceOfWork(X,Y) and placeOfDeath -> PlaceOfBirth(X,Y)

        df['label'] = np.where((df['anything'] == df[self._rep_var[1:]]), 1, 0)
        df = df.drop(['anything'], axis=1)
        df = df.drop_duplicates()
        cols = list(df.columns)
        cols.remove('label')

        idxs = []
        for uu, df__ in df.groupby(cols):
            if df__.shape[0] > 1:
                idxs.extend(list(df__[df__['label'] == 0].index))

        df.loc[idxs, 'label'] = 1
        df = df.drop_duplicates()

        df = df.rename(columns={'anythinglit': 'feat'})
        df['feat'] = df['feat'].astype(float)
        return df, suitable_pred_str

    def plot_scatt(self, feat_pd_, splits=[], by='feat', grp_by='label'):
        scatter_x = np.array(feat_pd_[by])
        scatter_y = np.array(feat_pd_[by])
        group = np.array(feat_pd_[grp_by])
        cdict = {0: 'red', 1: 'blue'}
        fig, ax = plt.subplots()
        for g in np.unique(group):
            ix = np.where(group == g)
            if g == 0:
                ss = 100
            else:
                ss = 20
            ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=ss)
        ax.legend()
        for s in splits:
            plt.axvline(x=s)
        plt.show()

    def start_binning(self, df_clf, binning_mode):
        binning = None
        if binning_mode == "opt":
            binning = OptimalBinningFitStats(df_clf)
        elif binning_mode == 'custom_mdlp':
            binning = CustomMDLPFitStats(df_clf)
        elif binning_mode == "quantile":
            binning = UnsupervisedBinning(df_clf, prebinning_method="quantile")
        elif binning_mode == "uniform":
            binning = UnsupervisedBinning(df_clf, prebinning_method="uniform")
        elif binning_mode == "mdlp":
            binning = MDLPFitStats(df_clf)
        elif binning_mode == "tree":
            binning = TreeBinnig(df_clf)
        else:
            NotImplementedError()
        return binning
