from santander_const import all_cores

def get_names_len():
    return len(all_cores)

def get_names(round_num=0):
    return all_cores[round_num]

def is_nonzero(x):
    return x != '0' and x != '0.0'

def is_zero(x):
    return x == '0' or x == '0.0'

def compare_full(a, b):
    if ('.' in a and '.' in b) or ('.' not in a and '.' not in b):
        return a == b
    if '.' not in a:
        return a + '.0' == b
    if '.' not in b:
        return a == b + '.0'

def is_fake_number(a):
    part = a.split('.')
    return len(part) > 1 and len(part[1]) > 2

def is_divided_number(a):
    part = a.split('.')
    return len(part) > 1 and ((part[1] == '66' and part[0][-1] == '6') or (part[1] == '34' and part[0][-1] == '3'))

def is_weird_number(a):
    part = a.split('.')
    if len(part) > 1 and ((part[1] == '66' and part[0][-1] == '6') or (part[1] == '34' and part[0][-1] == '3')):
        return False
    return len(part[0]) < 3 or part[0][-3:] != '000'


def has_int_value(data):
    for d in data:
        float_d = float(d)
        round_d = float(round(float_d))
        if float_d > 0 and float_d == round_d:
            #print(d)
            return True
    return False


def is_fake_data(data):
    for d in data:
        if len(d.split('.')[1]) > 2:
            return not has_int_value(data)
            #if has_int_value(data):
            #    print('HAS INT!!!')
            #    print(data)
            #    exit()
            return True
    return False

def get_all_repare_factors():
    return [3, 7, 9, 11, 13, 15, 17, 19, 31]
    return [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 111]

def repare_value(a):
    part = a.split('.')
    if a == '2533333.34':
        pass
        #exit()
    if len(part) < 2:
        return (a, 1)
    if part[1] == '0' or part[1] == '00':
        return (a, 1)
    if len(part[1]) > 2:
        return (a, 0)
    float_a = float(a)
    for rep_factor in get_all_repare_factors():
        rep_a = str(round(float_a * rep_factor))
        if part[1] == '34':
            pass
            #print(a, rep_a)
            #exit()
        if len(rep_a) > 3 and rep_a[-3:] == '000':
            if rep_factor == 39:
                pass
                #print(a, rep_a)
                #exit()
            return (rep_a + '.0', rep_factor)
    return (a, 0)


