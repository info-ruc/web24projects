import os.path
import re
import shutil
from op.op import *
from sql.parser import *
from table.table import create_new_memtable, alter_meta_table, DATASET_ROOT
from op.model import *
from reuse.reuse import *
from ModelSelection.model_selection import ModelSelection
from join import tables_join
import copy

def parse(sql):
    parser = SQLParser(sql)
    stmt = parser.parse()
    parse_ops(stmt)
    return stmt


def set_stmt_attrs(stmt, attrs):
    stmt.attrs = attrs
    stmt.attr_ops = [None] * len(attrs)


def execute_select_stmt(stmt, table_dict):
    _split_tables = stmt.table.split(',')
    _tables = [re.sub('\s', '', table_name) for table_name in _split_tables]
    tables = []
    for table in _tables:
        if table not in table_dict.keys():
            raise Exception("Table: " + table + " not existed")
        else:
            # use deep copy of the tables
            table_copy = copy.deepcopy(table_dict[table])
            tables.append(table_copy)
    # TODO： predicate push down
    for condition in stmt.conditions.conditions:
        if condition.value_table == None or condition.value_table == condition.attr_table:
            print("  predicate push down")
            pre_table_name = condition.attr_table
            pre_table_idx = -1
            for i in range(len(tables)):
                if tables[i].name == pre_table_name:
                    pre_table_idx = i
                    break

            pre_condition = condition
            if pre_condition.attr_op is None:
                pre_condition.attr = pre_condition.attr.split('.')[-1]
            else:
                pre_condition.attr_op.attr = pre_condition.attr_op.attr.split('.')[-1]
            if type(pre_condition.value) == str and len(pre_condition.attr.split('.')) > 1:
                pre_condition.value = pre_condition.attr.split('.')[-1]
            pre_conditions = Conditions([condition], '')
            tables[pre_table_idx].tuples = tables[pre_table_idx].select(pre_conditions)

            stmt.conditions.conditions.remove(condition)

    joint_table = tables_join(tables)
    #table = table_dict[stmt.table]
    #tuples = table.select(stmt.conditions)
    tuples = joint_table.select(stmt.conditions)

    # for '*' projection
    attrs = stmt.attrs
    if stmt.group_by is None and len(attrs) == 1 and attrs[0] == '*':
        attrs = joint_table.schema
        set_stmt_attrs(stmt, attrs)
    attr_ops = stmt.attr_ops
    schema = joint_table.schema

    if stmt.group_by is not None:
        schema, tuples = joint_table.group_by(tuples, stmt.group_by)
        if len(attrs) == 1 and attrs[0] == '*':
            set_stmt_attrs(stmt, schema)
            attrs, attr_ops = stmt.attrs, stmt.attr_ops

    if stmt.order_by is not None:
        sort_attr, tuples = joint_table.sort(schema, tuples, stmt.order_by, stmt.order_by_op, stmt.order)
        if sort_attr is not None:
            attrs = attrs + [sort_attr]
            attr_ops = attr_ops + [None]
            schema = schema + (sort_attr,)
    return joint_table.proj(schema, tuples, stmt.top, attrs, attr_ops)


def execute_create_dataset_stmt(stmt, table_dict):
    source_path = ""
    target_path = DATASET_ROOT + "/" + stmt.create_dataset_name
    train_set = "train.tsv"
    dev_set = "dev.tsv"
    test_set = "test.tsv"
    realm = ""
    for condition in stmt.conditions.conditions:
        if condition.attr == "path":
            path = condition.value
        elif condition.attr == "train_set":
            train_set = condition.value
        elif condition.attr == "dev_set":
            dev_set = condition.value
        elif condition.attr == "test_set":
            test_set = condition.value
        elif condition.attr == "realm":
            realm = condition.value
    if os.path.exists(target_path):
        print("\033[0;33mDataset already exists.\033[0m")
    else:
        print("Dataset found.")
        os.makedirs(target_path)
        if os.path.exists(source_path):
            for files in os.walk(source_path):
                for file in files:
                    if file == train_set or file == dev_set or file == test_set:
                        src_file = os.path.join(source_path, file)
                        shutil.copy(src_file, target_path)
#TODO: 拷贝失败
        print("Loading info")
        new_dataset_info = stmt.create_dataset_name + "\t" + train_set + "\t" + dev_set + "\t" + test_set + "\t" + realm
        table_dict['dataset_table'].append(new_dataset_info)


def execute_create_table_stmt(stmt, table_dict):
    if len(stmt.select_stmt.tokens) > 1:
        attrs, query_result = execute_select_stmt(stmt.select_stmt, table_dict)
    else:
        attrs = stmt.select_stmt.table.split(',')
        query_result = None
    attrs = [attr.replace('(', '').replace(')', '') for attr in attrs]
    new_table = create_new_memtable(stmt.create_table_name, attrs, query_result)
    table_dict[new_table.name] = new_table
    return new_table


def write_data_and_labels(op_name, domain, select_res, file_name):
    with open('dataset/' + domain + '/' + file_name, 'w') as f:
        if op_name == 'SentimentCls':
            f.write('label\ttext_a\n')
        elif op_name == 'SentenceSim':
            f.write('label\ttext_a\ttext_b\n')
        for res in select_res:
            label = 1 if res[-1] >= 0.5 else 0
            f.write(str(label) + '\t' + res[0] + '\n')


def file_from_top_n(from_file, dataset, top_n):
    from_path = DATASET_ROOT + '/' + dataset + '/' + from_file
    top_n_lines = open(from_path).readlines()[:top_n + 1]
    to_file = from_file + '_' + str(top_n)
    to_path = DATASET_ROOT + '/' + dataset + '/' + to_file
    with open(to_path, 'w') as f:
        for line in top_n_lines:
            f.write(line)
    return to_file


def get_models(all_models, use_models):
    if use_models is None:
        return all_models
    use_models = use_models.split(',')
    models = []
    for model in all_models:
        if model.name in use_models:
            models.append(model)
    return models


def execute_show_stmt(stmt, table_dict):
    if stmt.show_item == 'tables':
        attrs = ["Name", "Column", "Path"]
        result = table_dict["meta_table"].tuples

    if stmt.show_item == 'datasets':
        attrs = ["Name", "Training File", "Development File", "Test File", "Realm"]
        result = table_dict["dataset_table"].tuples

    if stmt.show_item == 'models':
        attrs = ["Name", "Training Dataset", "path", "Task"] # , "Training File"
        result = table_dict["model_table"].tuples

    return attrs, result


def execute_alter_table_stmt(stmt, table_dict):
    table = table_dict[stmt.table]
    schema = table.schema
    if stmt.add is not None:
        add_attr, tuples = table.add(schema, table.tuples, stmt.add_op)
    table_dict[stmt.table].save_tuples(tuples)
    alter_meta_table(table.name, stmt.add.split('(')[0])


def execute_create_model_stmt(stmt, table_dict):
    model_table = table_dict['model_table']
    base_model_table = table_dict['base_model']
    if stmt.model_name in [x[0] for x in model_table.tuples]:
        print('model_name: %s existed, train model not execute' % stmt.model_name)
        return None
    dataset = stmt.conditions.condition_value('dataset')
    output_path, dataset_record, training_data_file = None, None, None
    """
    print("\033[0;33mSelecting model using fast model selection.\033[0m")
    # todo: 一个确定找谁作为底层模型来微调的方法，调用ICDE的工作确定
    modelSelection = ModelSelection()
    model_selection_result = modelSelection.select()
    base_model_name = model_selection_result['left_models'][0]
    """
    print("\033[0;33mChoose random model. Fast model selection is not used.\033[0m")
    print("\033[0;33mThis is supposed to be a comparison.\033[0m")
    base_model_name = 'Jeevesh8--init_bert_ft_qqp-24'
    print("\033[0;32mUsing %s as base model.\033[0m" % base_model_name)
    # base_model_name = select_best_base_model()
    # base_model_path = base_model_table.tuples[0][-1]
    for name, path in base_model_table.tuples:
        if name == base_model_name:
            base_model_path = path

    if dataset is not None:
        print('will create model from dataset')
        dataset_record = [x for x in table_dict['dataset_table'].tuples if x[0] == dataset][0]
        datanum = stmt.conditions.condition_value('datanum')
        training_data_file = 'train.tsv' if datanum is None else file_from_top_n('train.tsv', dataset, int(datanum))
        dataset_record = (
        dataset_record[0], training_data_file, dataset_record[2], dataset_record[3], dataset_record[4])
        print(dataset_record)
        output_path = train_model(dataset_record, base_model_path)
    else:
        print('will create model from normal table column')
        op_name, table_name, table_column = stmt.conditions.condition_value('op'), stmt.conditions.condition_value(
            'table_name'), stmt.conditions.condition_value('table_column')

        datasets = [x for x in table_dict['dataset_table'].tuples if x[-1] == op_name]
        # using auto labeling default for unsupervised learning
        use_models = stmt.conditions.condition_value('use_models')
        reuse_algo = Reuse(op_name, datasets, get_models(list(get_op_dict()[op_name].model_dict.values()), use_models))

        table = table_dict[table_name]
        column_idx = table.schema.index(table_column)
        unlabeled_data = [x[column_idx] for x in table.tuples]

        datanum = stmt.conditions.condition_value('datanum')
        if datanum is not None:
            unlabeled_data = unlabeled_data[:int(datanum)]

        labeled_num = int(stmt.conditions.condition_value('label_num'))

        label_res = reuse_algo.auto_labeling(unlabeled_data)
        select_res = reuse_algo.equal_select_for_each_cls(label_res, labeled_num)
        # select_res = reuse_algo.effective_select(label_res, labeled_num)
        model_domain = stmt.model_name.split('_')[0]
        training_data_file = 'auto_labeling.tsv'
        write_data_and_labels(op_name, model_domain, select_res, training_data_file)
        dataset_record = (model_domain, 'auto_labeling.tsv', 'dev.tsv', 'test.tsv', op_name)
        dataset = model_domain
        output_path = train_model(dataset_record)

    model_table.tuples.append((stmt.model_name, dataset, output_path, dataset_record[-1], training_data_file))
    model_table.save()
