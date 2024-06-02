import torch

from executor import *
from op.op import *
from table.table import *
from MyStr import __MyStr__
from PIL import Image
import datetime
from random import randint

def print_line(fields):
    """
    Standard output by restricting number of characters
    Input： fields，table items in the list form，each character is an element
    """
    if len(fields) == 1:
        print("|{0:<40s}|".format(__MyStr__(fields[0])))
    elif len(fields) == 2:
        print("|{0:<40s}|{1:<40s}|".format(__MyStr__(fields[0]), __MyStr__(fields[1])))
    elif len(fields) == 3:
        print("|{0:<40s}|{1:<40s}|{2:<40s}|".format(__MyStr__(fields[0]), __MyStr__(fields[1]), __MyStr__(fields[2])))
    elif len(fields) == 4:
        print("|{0:<40s}|{1:<20s}|{2:<40s}|{3:<40s}"
              .format(__MyStr__(fields[0]), __MyStr__(fields[1]), __MyStr__(fields[2]), __MyStr__(fields[3])))
    elif len(fields) == 5:
        print("|{0:<40s}|{1:<40s}|{2:<40s}|{3:<40s}|{4:<40s}|"
              .format(__MyStr__(fields[0]), __MyStr__(fields[1]), __MyStr__(fields[2]),
                      __MyStr__(fields[3]), __MyStr__(fields[4])))


def print_select_result(attrs, tuples):
    """
    Output Methods
    Input： attrs，the output attributes
            list，elements refer to attributes
            tuples，the data from table
    """
    if len(tuples) == 0:
        return
    if 'image' in attrs and 'path' in attrs:
        path_column = attrs.index('path')
        if len(tuples) > 1:
            print("\033[0;35mOnly show one of the matched image\033[0m")
        index = randint(0, len(tuples) - 1)
        image = Image.open(tuples[index][path_column])
        image.show()
    else:
        print_line(attrs) #header
        for item in tuples:
            print_tuple = ()
            for attr in item:
                attr = str(attr)
                if len(attr) > 20:
                    print_tuple = print_tuple + (attr[:18] + "..",)
                else:
                    print_tuple = print_tuple + (attr,)
            print_line(print_tuple)


def run():
    print('OLML Database init..')
    init_tables()
    init_ops()
    while True:
        sql = input('olml > ')

        time_start = datetime.datetime.now()
        # for python version < 3.9 use time.clock() instead

        if sql == 'exit':
            break

        print('Start parse sql query...')
        stmt = parse(sql)  # parse SQL
        print('Finish parse sql query')

        """
        stmt_type in stmt saves the type of sql
        execute related functions according to different parse outcomes
        """

        if stmt.stmt_type == 'select':
            print('Start execute select query...')
            attrs, result = execute_select_stmt(stmt, get_table_dict())
            print('Finish execute select stmt, return %d records:' % len(result))
            print_select_result(attrs, result)

        elif stmt.stmt_type == 'create_dataset':
            print('Start execute create dataset query...')
            execute_create_dataset_stmt(stmt, get_table_dict())
            print('Finish execute create dataset stmt')

        elif stmt.stmt_type == 'create_table':
            print('Start execute create table query...')
            new_table = execute_create_table_stmt(stmt, get_table_dict())
            print('Finish execute create table stmt, put %d records:' % new_table.size())

        elif stmt.stmt_type == 'create_model':
            print('Start execute create model query...')
            execute_create_model_stmt(stmt, get_table_dict())
            print('Finish execute create model stmt, model_name: %s' % stmt.model_name)

        elif stmt.stmt_type == 'alter_table':
            print('Start execute alter table query...')
            execute_alter_table_stmt(stmt, get_table_dict())
            print('Finish execute alter table stmt')

        elif stmt.stmt_type == 'show':
            print('Start execute show query...')
            attrs, result = execute_show_stmt(stmt, get_table_dict())
            print('Finish execute show stmt')
            print_select_result(attrs, result)

        else:
            print("Illegal sql: %s" % sql)

        time_end = datetime.datetime.now()
        time_take = time_end - time_start
        print("\033[0;35mCommand execution time %.2fs\033[0m" % time_take.total_seconds())


if __name__ == "__main__":
    """
        Start OLML
    """
    run()
