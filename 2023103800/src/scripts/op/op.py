import sys

from table.table import get_table_dict
from op.model import *
import tfidf

op_names = ['SentimentClsCH', 'SentenceSimCH', 'NerCH',
            'SentimentClsEN', 'SentenceSimEN', 'NerEN',
            'ImageClass2', 'ImageClass3', 'Image2Text', 'Digit', 'ObjectDetection',
            'TransCH2EN', 'TransEN2CH', 'TransEN2FR', 'TransFR2EN']
op_dict = {}
active_ops = []


class DNNOp:
    def __init__(self, name):
        self.name = name
        self.model_dict = {}

    def load_models(self, models, ModelCls):
        model_dict = {}
        for model_record in models:
            model = ModelCls(model_record)
            model.load_model()
            model_dict[model_record[0]] = model
        return model_dict

    def set_args(self, attr, op_input):
        self.attr = attr
        self.op_input = op_input

    def select_model(self, table):
        if len(self.model_dict.keys()) == 0:
            return None
        # model selection by dataset_name match
        for model in self.model_dict.values():
            if table.name.startswith(model.dataset_name):
                print("\033[0;32mFound Corresponding model at: %s\033[0m" % model.model_path)
                return model
            #return model
        # return the first model
        print("\033[0;33mNo Corresponding model. Calculating similar one.\033[0m")

        return self.select_similar_model(table)

    def select_similar_model(self, table):
        documents = []
        table_dict = get_table_dict()
        for model in self.model_dict.values():
            if model.dataset_name in table_dict:
                tuples = table_dict[model.dataset_name].tuples
            else:
                tuples = [model.dataset_name]
            list_tuples = []
            for tuple in tuples:
                list_tuples.append(list(tuple))
            document = ".".join('%s' % a for a in list_tuples)
            documents.append(document)
        document_test = ".".join('%s' % a for a in table.tuples)
        model_index = tfidf.get_similar_model_index(documents, document_test)
        print("\033[0;32mChoose model %s as infer model\033[0m" % list(self.model_dict.values())[model_index].name)
        return list(self.model_dict.values())[model_index]

    def compute(self, table, attr_values):
        model = self.select_model(table)
        if model is None:
            raise Exception('No model trained for op: %s' % self.name)
        # print("Using model: %s" % model.model_path)
        return model.compute(self.op_input, attr_values)

        # TODO: add op_input to attr_values
        # if self.op_input is None:
        #  attr_values = ['text_a'] + attr_values
        #  dataset = read_dataset_from_enumerate(self.args, attr_values)
        #  return [x[-1][:6] for x in infer(self.device, self.model, self.args, dataset)]
        # else:
        #  dataset_lines = ['text_a\ttext_b']
        #  dataset_lines = dataset_lines + [(self.op_input + '\t' + x) for x in attr_values]
        #  dataset = read_dataset_from_enumerate(self.args, dataset_lines)
        #  return [x[-1][:6] for x in infer(self.device, self.model, self.args, dataset)]


class SentenceSimCH(DNNOp):
    def __init__(self, name):
        super(SentenceSimCH, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, SentenceSimModel)


class SentimentClsCH(DNNOp):
    def __init__(self, name):
        super(SentimentClsCH, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, DNNModel)


class NERCH(DNNOp):
    def __init__(self, name):
        super(NERCH, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, NERModelCH)


class SentenceSimEN(DNNOp):
    def __init__(self, name):
        super(SentenceSimEN, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, SentenceSimModel)


class SentimentClsEN(DNNOp):
    def __init__(self, name):
        super(SentimentClsEN, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, DNNModel)


class NEREN(DNNOp):
    def __init__(self, name):
        super(NEREN, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, NERModelEN)


class TranslationCH2EN(DNNOp):
    def __init__(self, name):
        super(TranslationCH2EN, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, TransCH2ENModel)

class TranslationEN2CH(DNNOp):
    def __init__(self, name):
        super(TranslationEN2CH, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, TransEN2CHModel)


class TranslationEN2FR(DNNOp):
    def __init__(self, name):
        super(TranslationEN2FR, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, TransEN2FRModel)


class TranslationFR2EN(DNNOp):
    def __init__(self, name):
        super(TranslationFR2EN, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, TransFR2ENModel)


class ImageClass2(DNNOp):
    def __init__(self, name):
        super(ImageClass2, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, ImageClass2Model)


class ImageClass3(DNNOp):
    def __init__(self, name):
        super(ImageClass3, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, ImageClass3Model)


class Digit(DNNOp):
    def __init__(self, name):
        super(Digit, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, DigitModel)


class Image2Text(DNNOp):
    def __init__(self, name):
        super(Image2Text, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, Image2TextModel)


class ObjectDetection(DNNOp):
    def __init__(self, name):
        super(ObjectDetection, self).__init__(name)

    def load(self, models):
        self.model_dict = self.load_models(models, ObjectDetectionModel)


def init_ops():
    print("  Start load ops...")
    global op_dict
    op_dict = {}
    model_table = get_table_dict()['model_table']
    for name in op_names:
        models = [x for x in model_table.tuples if x[-2] == name]
        if name == 'SentenceSimCH':
            op = SentenceSimCH(name)
        elif name == 'SentimentClsCH':
            op = SentimentClsCH(name)
        elif name == 'NerCH':
            op = NERCH(name)
        elif name == 'SentenceSimEN':
            op = SentenceSimEN(name)
        elif name == 'SentimentClsEN':
            op = SentimentClsEN(name)
        elif name == 'NerEN':
            op = NEREN(name)
        elif name == 'TransCH2EN':
            op = TranslationCH2EN(name)
        elif name == 'TransEN2CH':
            op = TranslationEN2CH(name)
        elif name == 'TransEN2FR':
            op = TranslationEN2FR(name)
        elif name == 'TransFR2EN':
            op = TranslationFR2EN(name)
        elif name == 'ImageClass2':
            op = ImageClass2(name)
        elif name == 'ImageClass3':
            op = ImageClass3(name)
        elif name == 'Image2Text':
            op = Image2Text(name)
        elif name == 'Digit':
            op = Digit(name)
        elif name == 'ObjectDetection':
            op = ObjectDetection(name)
        op.load(models)
        op_dict[name] = op
        print("    Op: %s inited, load %d models" % (name, len(models)))
    print("  Totally %d ops inited" % len(op_dict.keys()))


def parse_op_from_str(stmt_str, as_content):
    tokens = stmt_str.split('(')
    if len(tokens) > 1:
        op_name = tokens[0]
        tokens = tokens[1][:-1].split(',')
        if tokens[0] in as_content:
            tokens[0] = as_content[tokens[0]]

        if len(tokens) > 1:
            return op_name, tokens[0], tokens[1]
        else:
            return op_name, tokens[0], None
    else:
        return None, None, None


def parse_ops(stmt):
    global op_dict, active_ops
    active_ops = []

    if stmt.__class__.__name__ == 'CreateStmt':
        stmt = stmt.select_stmt

    if stmt.__class__.__name__ == 'SelectStmt':
        # for project attributes
        for i in range(len(stmt.attrs)):
            attr = stmt.attrs[i]
            op_name, attr, op_input = parse_op_from_str(attr, stmt.as_content)
            if op_name is not None and op_name in op_dict.keys():
                op_dict[op_name].set_args(attr, op_input)
                active_ops.append(op_dict[op_name])
                stmt.attr_ops[i] = op_dict[op_name]
                print('    Active op: %s from project stmt' % op_name)

        # for where condition
        for condition in stmt.conditions.conditions:
            op_name, attr, op_input = parse_op_from_str(condition.attr, stmt.as_content)
            if op_name is not None:
                op_dict[op_name].set_args(attr, op_input)
                active_ops.append(op_dict[op_name])
                condition.attr_op = op_dict[op_name]
                _split = condition.attr_op.attr.split('.')
                if len(_split) > 1:
                    condition.attr_table = _split[0]
                else:
                    condition.attr_table = stmt.table
                print('    Active op: %s from select stmt' % op_name)

        # for order by attributes
        if stmt.order_by is not None:
            op_name, attr, op_input = parse_op_from_str(stmt.order_by, stmt.as_content)
            if op_name is not None:
                op_dict[op_name].set_args(attr, op_input)
                active_ops.append(op_dict[op_name])
                stmt.order_by_op = op_dict[op_name]
                print('    Active op: %s from order_by stmt' % op_name)

    if stmt.__class__.__name__ == 'AlterTableStmt':
        # for add attribute
        if stmt.add is not None:
            op_name, attr, op_input = parse_op_from_str(stmt.add, stmt.as_content)
            if op_name is not None:
                op_dict[op_name].set_args(attr, op_input)
                active_ops.append(op_dict[op_name])
                stmt.add_op = op_dict[op_name]
                print('    Active op: %s from add stmt' % op_name)

    print('  Finish parse ops, totally active ops: %d' % len(active_ops))


def get_active_ops():
    global active_ops
    return active_ops


def get_op_dict():
    global op_dict
    return op_dict


if __name__ == "__main__":
    init_ops()
