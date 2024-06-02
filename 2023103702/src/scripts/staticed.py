import pickle
import os


class load:
    path = None  # 数据文件名称

    def __init__(self, file_path):
        try:
            self.path = file_path
            if not os.path.exists(self.path + '.pkl'):
                with open(self.path + '.pkl', "wb") as dict_file:
                    pickle.dump(dict(), dict_file, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e, e.__traceback__.tb_lineno)

    # 初始化数据文件

    def load(self, file_path=None):
        try:
            if file_path:
                with open(file_path + '.pkl', 'rb') as dict_file:
                    return pickle.load(dict_file)
            elif not file_path:
                with open(self.path + '.pkl', 'rb') as dict_file:
                    return pickle.load(dict_file)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(e.__traceback__.tb_lineno)
            return False

    # 加载数据文件

    def save(self, dict_data, file_path=None):
        try:
            if file_path:
                with open(file_path + '.pkl', 'wb') as dict_file:
                    pickle.dump(dict_data, dict_file, pickle.HIGHEST_PROTOCOL)
            elif not file_path:
                with open(self.path + '.pkl', 'wb') as dict_file:
                    pickle.dump(dict_data, dict_file, pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(e.__traceback__.tb_lineno)
            return False
        else:
            return True

    # 保存数据文件

    def select(self, key_name):
        try:
            data = self.load()
            value = data.get(key_name)
            if value:
                return value
            else:
                return None
        except Exception as e:
            print(e, e.__traceback__.tb_lineno)
            return False

    # 查询数据文件

    def update(self, key_name, value):
        try:
            data = self.load()
            data[key_name] = value
            self.save(data)
        except Exception as e:
            print(e, e.__traceback__.tb_lineno)
            return False
        else:
            return True

    # 更新数据文件

    def remove(self, key_name):
        try:
            data = self.load()
            del data[key_name]
            self.save(data)
        except KeyError:
            return False
        except Exception as e:
            print(e, e.__traceback__.tb_lineno)
            return False

    # 删除数据值

    def clear(self):
        try:
            data = dict()
            self.save(data)
        except Exception as e:
            print(e, e.__traceback__.tb_lineno)
            return False
        else:
            return True

    # 清空数据文件

