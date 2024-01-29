import math

from terminaltables import AsciiTable


class AbstractUtils:

    @staticmethod
    def show_result_in_key_value(tuple_data):
        table_data = [['key', 'values']]
        for i in tuple_data:
            table_data.append([i[0], i[1]])
        table = AsciiTable(table_data)
        result = table.table
        print(result)
        return str(result)

    def load_data(self):
        """
        Load train, validation, test data and other data
        """
        pass

    def load_hyperparams(self):
        """
        Load hyperparameters: batch_size, eta, num_factor, num_negative, lambda, ...
        """
        pass

    def evaluate(self):
        """
        Compute the average evaluate score for users
        """
        pass

    def eval_one_rating(self):
        """
        Compute the evaluate score for one user
        """
        pass

    def predict(self):
        """
        Get the output prediction scores
        """
        pass

    @staticmethod
    def get_hit_ratio(rank_list, gt_item):
        for item in rank_list:
            if item == gt_item:
                return 1.0
        return 0

    @staticmethod
    def get_ndcg(rank_list, gt_item):
        for i in range(len(rank_list)):
            item = rank_list[i]
            if item == gt_item:
                return math.log(2) / math.log(i + 2)
        return 0
