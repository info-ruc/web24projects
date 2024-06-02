import sqlparse
import re


def is_token_type(token, token_type):
    return token.ttype.__str__() == token_type


def remove_whitespace(tokens):
    return [token for token in tokens if is_token_type(token, 'Token.Text.Whitespace') == False]


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


class SQLParser:
    def __init__(self, sql):
        self.sql = sql
        self.tokens = None

    def parse(self):
        self.tokens = sqlparse.parse(self.sql)[0].tokens
        if self.is_select_stmt():
            print(" Finish parse select statement: " + self.sql)
            return SelectStmt(self.tokens)
        elif self.is_create_model_stmt():
            print(" Finish parse create model statement: " + self.sql)
            return CreateModelStmt(self.tokens)
        elif self.is_create_dataset_stmt():
            print(" Finish parse create dataset statement: " + self.sql)
            return CreateDatasetStmt(self.tokens)
        elif self.is_create_stmt():
            print(" Finish parse create table statement: " + self.sql)
            return CreateTableStmt(self.tokens)
        elif self.is_alter_table_stmt():
            print(" Finish parse alter table statement: " + self.sql)
            return AlterTableStmt(self.tokens)
        elif self.is_show_stmt():
            print(" Finish parse show statement: " + self.sql)
            return ShowStmt(self.tokens)

    def is_select_stmt(self):
        return self.tokens[0].value == 'select'

    def is_create_model_stmt(self):
        tokens = remove_whitespace(self.tokens)
        return tokens[0].value == 'create' and tokens[1].value.startswith('model ')

    def is_create_dataset_stmt(self):
        tokens = remove_whitespace(self.tokens)
        return tokens[0].value == 'create' and tokens[1].value.startswith('dataset ')

    def is_create_stmt(self):
        return self.tokens[0].value == 'create'

    def is_alter_table_stmt(self):
        tokens = remove_whitespace(self.tokens)
        return tokens[0].value == 'alter' and tokens[1].value.startswith('table')

    def is_show_stmt(self):
        return self.tokens[0].value == 'show'


CONDITION_OPS = ['!=', '=', '>', '<']


class Condition:
    def __init__(self, attr, cond_op, value):
        self.attr = attr
        self.cond_op = cond_op
        self.value = float(value) if is_number(value) else value
        self.attr_op = None

        _attr_split = attr.split('.')
        self.attr_table = _attr_split[0] if len(_attr_split) > 1 else None
        self.value_table = None
        if type(self.value) == str:
            _value_split = value.split('.')
            if len(_value_split) > 1:
                self.value_table = _value_split[0]

    def set_attr_value(self, attr_value):
        self.attr_value = attr_value

    def set_value(self, value):
        self.value = value

    def evaluate(self):
        if self.cond_op == '=':
            return self.attr_value == self.value
        elif self.cond_op == '!=':
            return self.attr_value != self.value
        elif self.cond_op == '>':
            return self.attr_value > self.value
        elif self.cond_op == '<':
            return self.attr_value < self.value
        else:
            raise Exception('Illegal condition op: %s' % self.cond_op)


class Conditions:
    def __init__(self, conditions, and_or):
        self.conditions, self.and_or = conditions, and_or

    def evaluate(self):
        if len(self.conditions) == 0:
            return True
        if len(self.conditions) == 1:
            return self.conditions[0].evaluate()
        for condition in self.conditions:
            eval_result = condition.evaluate()
            if eval_result == True and self.and_or == 'or':
                return True
            elif eval_result == False and self.and_or == 'and':
                return False
        return self.and_or == 'and'

    def condition_value(self, cond_name):
        values = [cond.value for cond in self.conditions if cond.attr == cond_name]
        if len(values) == 0:
            return None
        return values[0]


class BaseStmt:
    def __init__(self, tokens):
        self.tokens = remove_whitespace(tokens)
        self.stmt_type = None

    def value_index(self, value):
        for i in range(len(self.tokens)):
            if self.tokens[i].value == value:
                return i
        return -1

    def parse_condition_str(self, condition_str):
        for condition_op in CONDITION_OPS:
            if condition_str.find(condition_op) > 0:
                tokens = condition_str.split(condition_op)
                compare_token = tokens[0].strip()
                if compare_token in self.as_content:
                    compare_token = self.as_content[compare_token]
                return Condition(compare_token, condition_op, tokens[1].strip())
        raise Exception('Fail to parse condition: %s' % condition_str)

    def parse_conditions(self, tokens):
        # find where condition and parse it
        where_token = [token for token in tokens if token.value.startswith('where')]
        if len(where_token) != 0:
            if len(where_token) > 1:
                raise Exception('Multi where substmt')
            conditions_str = where_token[0].value.split('where ')[1]
            and_or = ''
            # split multi-condition
            if conditions_str.find(' and ') > 0:
                and_or = 'and'
                conditions_str = conditions_str.split(' and ')
            elif conditions_str.find(' or ') > 0:
                and_or = 'or'
                conditions_str = conditions_str.split('or')
            else:
                conditions_str = [conditions_str]

            conditions = []
            for condition_str in conditions_str:
                conditions.append(self.parse_condition_str(condition_str))
            return Conditions(conditions, and_or)
        else:
            return Conditions([], '')


class AlterTableStmt(BaseStmt):
    def __init__(self, tokens):
        super(AlterTableStmt, self).__init__(tokens)
        self.stmt_type = 'alter_table'
        self.as_content = {}
        self.table = self.parse_table(self.tokens)
        self.add = self.parse_add(self.tokens)
        self.add_op, self.schema = None, None

    def ttype_value_index(self, tokens, ttype, value):
        idx = 0
        while idx < len(tokens):
            if is_token_type(tokens[idx], ttype) and tokens[idx].value == value:
                return idx
            idx = idx + 1
        return -1

    def parse_table(self, tokens):
        table_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'table')
        return tokens[table_idx + 1].value

    def parse_add(self, tokens):
        add_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'add')
        return tokens[add_idx + 1].value


class CreateTableStmt(BaseStmt):
    def __init__(self, tokens):
        super(CreateTableStmt, self).__init__(tokens)
        self.stmt_type = 'create_table'
        self.as_content = {}
        self.create_tokens, self.select_tokens = self.split_create_and_select_tokens()
        self.create_table_name = self.parse_create_table_name()
        self.select_stmt = SelectStmt(self.select_tokens)

    def split_create_and_select_tokens(self):
        value_idx = self.value_index('select')
        return self.tokens[:value_idx], self.tokens[value_idx:]

    def parse_create_table_name(self):
        return self.tokens[self.value_index('table') + 1].value


class CreateModelStmt(BaseStmt):
    def __init__(self, tokens):
        super(CreateModelStmt, self).__init__(tokens)
        self.stmt_type = 'create_model'
        self.as_content = {}
        self.model_name = self.tokens[1].value.split(' ')[-1]
        self.conditions = self.parse_conditions(tokens)


class CreateDatasetStmt(BaseStmt):
    def __init__(self, tokens):
        super(CreateDatasetStmt, self).__init__(tokens)
        self.stmt_type = 'create_dataset'
        self.as_content = {}
        self.create_dataset_name = self.tokens[1].value.split(' ')[-1]
        self.conditions = self.parse_conditions(tokens)

    def parse_create_dataset_name(self):
        return self.tokens[self.value_index('dataset') + 1].value


class SelectStmt(BaseStmt):
    def __init__(self, tokens):
        super(SelectStmt, self).__init__(tokens)
        self.stmt_type = 'select'
        self.as_content = {}
        self.top, self.attrs = self.parse_attrs(self.tokens)
        self.attr_ops = [None] * len(self.attrs)
        self.table = self.parse_table(self.tokens)
        self.conditions = self.parse_conditions(self.tokens)
        self.order_by, self.order = self.parse_order_by(self.tokens)
        self.order_by_op = None
        self.group_by = self.parse_group_by(self.tokens)

#TODO: where和order by共存时识别不出order by

    def ttype_value_index(self, tokens, ttype, value):
        idx = 0
        while idx < len(tokens):
            if is_token_type(tokens[idx], ttype) and tokens[idx].value == value:
                return idx
            idx = idx + 1
        return -1

    def get_keyword_value(self, tokens, keyword):
        for token in tokens:
            if is_token_type(token, 'Token.Keyword.' + keyword):
                return token.value
        return None

    def parse_attrs(self, tokens):
        # get the position of from
        from_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'from')
        if from_idx == -1:
            return 0, []
        # get the word between select and from
        attr_arr = [token.value for token in tokens[1: from_idx]]
        top, attrs = -1, None
        if attr_arr[0] == 'top':
            # top followed by a number and several colomn names
            if len(attr_arr) == 2:
                temp_arr = re.split(' +|,',attr_arr[1])
                top = int(temp_arr[0])
                attrs = []
                as_tag = 0
                as_reg = ""
                for item in temp_arr[1:]:
                    if item.lower() == "as":
                        as_tag = 1
                    elif as_tag == 1:
                        temp_token = [as_reg, "as", item]
                        self.parse_as(temp_token)
                        as_tag = 0
                    else:
                        if item != '':
                            as_reg = item
                            attrs.append(item)
            else:
                top = int(attr_arr[1])
                attrs = attr_arr[2].split(',')
        else:
            attr_tokens = attr_arr[0].split(',')
            attrs = []
            for attr_token in attr_tokens:
                temp_token = attr_token.lstrip()
                temp_token = re.sub(' +', ' ', temp_token)
                temp_token = temp_token.split(' ')
                attrs.append(temp_token[0])
                self.parse_as(temp_token)
        return top, attrs

    def parse_as(self, temp_token):
        if len(temp_token) == 3:
            self.as_content[temp_token[2]] = temp_token[0]

    def parse_table(self, tokens):
        from_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'from')
        return tokens[from_idx + 1].value

    def parse_order_by(self, tokens):
        order = self.get_keyword_value(tokens, 'Order')
        if order is not None:
            """
            这种情况识别不出 order by A desc 以及 order by A
            前者会被拆分成A desc从而找不到desc
            后者默认是desc
            """
            order_by_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'order by')
            order_by = tokens[order_by_idx + 1].value
            if order_by in self.as_content:
                order_by = self.as_content[order_by]
            return order_by, order
        elif self.ttype_value_index(tokens, 'Token.Keyword', 'order by') != -1:
            order_by_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'order by')
            order_by = tokens[order_by_idx + 1].value.split(' ')[0]
            if order_by in self.as_content:
                order_by = self.as_content[order_by]
            return order_by, "desc"
        return None, None

    def parse_group_by(self, tokens):
        group_by_idx = self.ttype_value_index(tokens, 'Token.Keyword', 'group by')
        return tokens[group_by_idx + 1].value if group_by_idx >= 0 else None


class ShowStmt(BaseStmt):
    def __init__(self, tokens):
        super(ShowStmt, self).__init__(tokens)
        self.stmt_type = 'show'
        self.show_item = self.tokens[1].value


if __name__ == "__main__":
    # sql = 'select top 3 question,answer from candidate_qa where Ner(question) != N/A and Ner(question) != 北京 order by SentenceSim(question, "北京面积有大多") desc;'
    # parser = SQLParser(sql)
    # parser.parse()
    # select_stmt = SelectStmt(parser.tokens)
    # print(select_stmt.top)
    # print(select_stmt.attrs)
    # print(select_stmt.table)
    # print(select_stmt.conditions)
    # print(select_stmt.order_by)
    # print(select_stmt.order)
    # select_stmt.conditions.conditions[0].set_attr_value('测试')
    # select_stmt.conditions.conditions[1].set_attr_value('北京')
    # print(select_stmt.conditions.conditions[0].evaluate())
    # print(select_stmt.conditions.conditions[1].evaluate())
    # print(select_stmt.conditions.evaluate())
    # sql = 'create table positive_review_with_ner select review,Ner(review) from review_table where SentimentCls(review) > 0.5 and Ner(review) != N/A;'
    # parser = SQLParser(sql)
    # stmt = parser.parse()
    # print(stmt.create_table_name)
    # print(stmt.select_stmt.attrs)
    # print(stmt.select_stmt.table)
    # print(stmt.select_stmt.conditions)
    # sql = 'select * from candidate_qa'
    # parser = SQLParser(sql)
    # stmt = parser.parse()
    # print(stmt.attrs)
    sql = 'select * from kefu_qa_table group by question order by count desc'
    parser = SQLParser(sql)
    stmt = parser.parse()
    for item in stmt.tokens:
        print(item)
        print(item.ttype, item.value)
    print(stmt.tokens)
    print(stmt.group_by)
    print(stmt.order_by)
    print(stmt.order)
