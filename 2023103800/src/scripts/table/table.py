import sys

DATASET_ROOT = 'E:/RUC/olml-2024-01-10/olml/dataset'
TABLE_ROOT = 'E:/RUC/olml-2024-01-10/olml/table/table_data'

class BaseTable:
    def __init__(self, name, schema):
        self.name = name
        self.schema = schema
        self.tuples = []
        self.tuples_num = 0

    def __lt__(self, other):
        if self.tuples_num < other.tuples_num:
            return True
        else:
            return False

    def size(self):
        self.tuples_num = len(self.tuples)
        return len(self.tuples)

    def select(self, conditions):
        # prepare op attribute values for conditions
        op_attr_values_dict = {}
        condition_attr_idx = []
        condition_value_idx = []
        for condition in conditions.conditions:
            if condition.attr_op is not None:
                # op_attr_value[condition.attr] = []
                attr_idx = self.schema.index(condition.attr_op.attr)
                condition_attr_idx.append(attr_idx)
                op_attr_values = [item[attr_idx] for item in self.tuples]
                op_attr_values_dict[condition.attr] = condition.attr_op.compute(self, op_attr_values)
            else:
                condition_attr_idx.append(self.schema.index(condition.attr))
                if condition.value in self.schema:
                    condition_value_idx.append(self.schema.index(condition.value))
                else:
                    condition_value_idx.append(-1)

        # select by condition
        results = []
        for j in range(len(self.tuples)):
            item = self.tuples[j]
            for i in range(len(conditions.conditions)):
                condition = conditions.conditions[i]
                if condition.attr_op is None:
                    condition.set_attr_value(item[condition_attr_idx[i]])
                    if condition_value_idx[i] != -1:
                        condition.set_value(item[condition_value_idx[i]])
                else:
                    condition.set_attr_value(op_attr_values_dict[condition.attr][j])
            if conditions.evaluate():
                results.append(self.tuples[j])
        return results

    def group_by(self, tuples, group_by_attr):
        attr_idx = self.schema.index(group_by_attr)
        group_by_count = {}
        group_by_avg = {}
        for item in tuples:
            value = item[attr_idx]
            if value in group_by_count.keys():
                group_by_count[value] = group_by_count[value] + 1
            else:
                group_by_count[value] = 1
        group_by_schema = [group_by_attr, 'count']
        group_by_records = [(k, str(group_by_count[k])) for k in group_by_count.keys()]
        return group_by_schema, group_by_records

    def sort(self, schema, tuples, order_by_attr, op, order):
        if op is None:
            idx = schema.index(order_by_attr)
            tuples.sort(key=lambda x: x[idx], reverse=(order == 'desc'))
            return None, tuples

        # for order_by with op
        attr_idx = schema.index(op.attr)
        attr_values = [item[attr_idx] for item in tuples]
        scores = op.compute(self, attr_values)
        tuples_with_score = []
        for i in range(len(tuples)):
            tuple_with_score = tuples[i] + (scores[i],)
            tuples_with_score.append(tuple_with_score)
        tuples_with_score.sort(key=lambda x: x[-1], reverse=(order == 'desc'))
        return op.name, tuples_with_score

    def proj(self, schema, tuples, top, attrs, attr_ops):
        if top != -1:
            tuples = tuples[:top]
        proj_attr_idxes = []
        op_attrs = []
        for i in range(len(attrs)):
            if attrs[i] == 'image':
                continue
            if attr_ops[i] is None:
                proj_attr_idxes.append(schema.index(attrs[i]))
            else:
                proj_attr_idxes.append(schema.index(attr_ops[i].attr))
            op_attrs.append([])
        result = []

        # fetch result by projection
        for item in tuples:
            projected = []
            for i in range(len(proj_attr_idxes)):
                idx = proj_attr_idxes[i]
                projected.append(item[idx])
                if attr_ops[i] is not None:
                    op_attrs[i].append(item[idx])
            result.append(projected)

        # replace fields with op results
        for i in range(len(attr_ops)):
            if attr_ops[i] is None:
                continue
            op_result = attr_ops[i].compute(self, op_attrs[i])
            for j in range(len(result)):
                result[j][i] = op_result[j]
        return attrs, result

    def add(self, schema, tuples, op):
        attr_idx = schema.index(op.attr)
        attr_values = [item[attr_idx] for item in tuples]
        add_schema = op.compute(self, attr_values)
        tuples_added = []
        for i in range(len(tuples)):
            tuple_added = tuples[i] + (add_schema[i],)
            tuples_added.append(tuple_added)
        return op.name, tuples_added


class MemTable(BaseTable):
    def __init__(self, name, schema):
        super(MemTable, self).__init__(name, schema)

    def load(self, tuples):
        self.tuples = tuples


class FileTable(BaseTable):
    def __init__(self, name, schema_str, file_path):
        super(FileTable, self).__init__(name, tuple(schema_str.split(',')))
        self.file_path = file_path
        self.tuples = None

    def load(self):
        self.tuples = []
        for line in open(self.file_path, encoding='utf-8').readlines():
            if line.strip() == "":
                continue
            self.tuples.append(tuple(line.strip().split('\t')))


class SerializeTable(FileTable):
    def __init__(self, name, schema_str, file_path):
        super(SerializeTable, self).__init__(name, schema_str, file_path)

    def append(self, line):
        with open(self.file_path, 'a') as f:
            f.write('\n'+line)

    def save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for row in self.tuples:
                f.write('\t'.join(row) + '\n')

    def save_tuples(self, tuples):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for row in tuples:
                f.write('\t'.join(row) + '\n')


META_TABLE = SerializeTable('meta_table', 'tablename,schema,path', 'table/table_data/meta_table.tsv')
META_TABLE.load()
DATASET_TABLE = SerializeTable('dataset_table', 'name,training,val,test,op', 'table/table_data/dataset_table.tsv')
DATASET_TABLE.load()
MODEL_TABLE = SerializeTable('model_table', 'name,dataset,path,op', 'table/table_data/model_table.tsv')
MODEL_TABLE.load()
BASE_MODEL_TABLE = SerializeTable('base_model', 'name,path', 'table/table_data/base_model.tsv')
BASE_MODEL_TABLE.load()

table_dict = {}
table_dict['meta_table'] = META_TABLE
table_dict['dataset_table'] = DATASET_TABLE
table_dict['model_table'] = MODEL_TABLE
table_dict['base_model'] = BASE_MODEL_TABLE


def create_new_memtable(table_name, schema, records):
    new_table = MemTable(table_name, tuple(schema))
    if records is not None:
        new_table.load([tuple(record) for record in records])
    new_table_info = table_name + '\t' + ",".join(schema) + '\t' + TABLE_ROOT + '/' + table_name + '.tsv'
    print("Add table info into META_TABLE")
    META_TABLE.append(new_table_info)
    return new_table


def init_tables():
    global table_dict
    print('  Start loading tables...')
    for row in META_TABLE.tuples:
        table = SerializeTable(row[0], row[1], row[2])
        table.load()
        table_dict[row[0]] = table
        print('    Finish load table: %s, size: %d' % (table.name, len(table.tuples)))

    print('  Totally load %d tables' % len(table_dict.keys()))


def get_table_dict():
    global table_dict
    return table_dict


def alter_meta_table(table_name, add_item):
    tuples = []
    for tuple in META_TABLE.tuples:
        tuples.append(list(tuple))
    for row in tuples:
        if row[0] == (table_name):
            row[1] = row[1] + "," + "".join(add_item)
    META_TABLE.save_tuples(tuples)
