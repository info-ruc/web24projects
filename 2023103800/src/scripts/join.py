from itertools import product
from table.table import SerializeTable


def simple_join(left_table, right_table):
    """
    Cartesian product
    """
    joint_name = left_table.name + "_JOIN_" + right_table.name

    joint_schema_list = []
    for _schema in left_table.schema:
        if len(_schema.split(".")) == 1:
            joint_schema_list.append(left_table.name + "." + _schema)
        else:
            joint_schema_list.append(_schema)
    for _schema in right_table.schema:
        if len(_schema.split(".")) == 1:
            joint_schema_list.append(right_table.name + "." + _schema)
        else:
            joint_schema_list.append(_schema)
    joint_schema = ','.join(joint_schema_list)

    joint_table = SerializeTable(joint_name, joint_schema, 'JOIN')

    joint_tuples = []
    for left_tuple in left_table.tuples:
        for right_tuple in right_table.tuples:
            joint_tuple = left_tuple + right_tuple
            joint_tuples.append(joint_tuple)

    joint_table.tuples = joint_tuples
    joint_table.tuples_num = left_table.tuples_num * right_table.tuples_num

    return joint_table


def tables_join(tables):
    """
    input:
        tables: a list of table name
    output:
        joined_table: one table refers to the joint result
    """
    # join the smallest table each time
    tables = sorted(tables)
    while len(tables) > 1:
        joint_table = simple_join(tables[0], tables[1])
        del tables[0]
        del tables[0]
        if len(tables) < 1:
            tables.append(joint_table)
        elif joint_table.tuples_num >= tables[-1].tuples_num:
            tables.append(joint_table)
        else:
            for i in range(len(tables)):
                if joint_table.tuples_num < tables[i].tuples_num:
                    tables.insert(i, joint_table)
                    break

    return tables[0]
