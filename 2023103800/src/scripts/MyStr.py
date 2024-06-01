import re

class __MyStr__():
    def __init__(self, s) -> None:
        self.buff = ''+s

    def __str__(self):
        return self.buff.__str__()

    def __len__(self):
        return self.buff.__len__()

    def __add__(self, s):
        return __MyStr__(self.buff.__add__(s))

    def get_real_width(self):
        widths = [(126, 1), (159, 0), (183, 2), (232, 2), (687, 1), (710, 0), (711, 1), (714, 2), (715, 2), (727, 0), (733, 1),
                  (879, 0), (1154, 1), (1161, 0), (4347, 1), (4447, 2), (7467, 1), (7521, 0), (8230, 2), (8369, 1),
                  (8426, 0), (8857, 2), (9000, 1), (9002, 2), (9733, 2), (9734, 2), (11021, 1), (12350, 2),
                  (12351, 1), (12438, 2), (12442, 0), (19893, 2), (19967, 1), (55203, 2), (63743, 1), (64106, 2),
                  (65039, 1), (65059, 0), (65131, 2), (65279, 1), (65376, 2), (65500, 1), (65510, 2), (120831, 1),
                  (128530, 2), (129318, 2), (262141, 2), (1114109, 1),
                  (0xffffffffffffffffffffffffffffffff, 1)]
        length = 0
        for c in self.buff:
            o = ord(c)
            if o == 0xe or o == 0xf:
                continue
            for num, wid in widths:
                if o <= num:
                    length += wid
                    break
        return length

    def __format__(self, __format_spec: str) -> str:
        # *<4  *>5  *^5
        # *<4s *>5s *^5s
        # 4  4s
        format_spec = __format_spec
        format_len = int(re.findall('.*?[<^>]?([\d+]+)s?', format_spec)[0])
        format_len -= self.get_real_width()-len(self.buff)
        format_spec = re.sub('.*?[<^>]?([\d+]+)s?', str(format_len), format_spec)

        return self.buff.__format__(format_spec)

if __name__ == "__main__":
    zw = '你好，世界'
    el = 'hello world'
    mixed = '你好,世界……'
    print('\n中英文无法对齐')
    print('{0:15s} | '.format(zw))
    print('{0:15s} | '.format(el))
    print('\n中英文1:2对齐')
    print('{0:15s} | '.format(__MyStr__(zw)))
    print('{0:15s} | '.format(__MyStr__(el)))
    print('{0:15s} | '.format(__MyStr__(mixed)))
    print(len(mixed))
    print(ord('😂'))
    for i in range(128000, 129000):
        print(chr(i))
