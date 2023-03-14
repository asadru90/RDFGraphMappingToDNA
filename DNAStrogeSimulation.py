import re
import math
from tabulate import tabulate
from operator import itemgetter
from functools import reduce


def decodeDNAStrand(sd_data, dc_type):
    if dc_type == "dictionary":
        return sd_data[1], sd_data[2][0], sd_data[3][0]
    elif dc_type == "bitmap":
        return sd_data[1][0], sd_data[2][0], sd_data[3][0], \
               sd_data[4][0], sd_data[5][0],
    else:
        return sd_data[1], sd_data[2][0], sd_data[3][0], \
               sd_data[4][0], sd_data[5][0],


def find_item(str_item, pl_data):
    str_item = str_item + '#'
    str_last = ''
    for idx, nxt_str in enumerate(pl_data):
        if nxt_str == str_item:
            return True, idx+1
        str_last = nxt_str
    if str_item > str_last:
        return False, 1
    else:
        return False, -1


def get_n_count(str_data, c1, n, c2):
    inlist = [m.start() for m in re.finditer(c2, str_data)]
    return int(str_data[0:inlist[n - 1]].count(c1))


def get_count(str_data, c):
    return int(str_data.count(c))


# function for mapping string to corresponding ID in the Dictionary table
def map_RDF_str_to_Id(str_item, sd_cnt, dict_mid_address, dict_strands):
    st_idx = 1
    md_idx = 0
    rt_idx = 0
    is_found = False
    ed_idx = sd_cnt
    sd_addr = int(dict_mid_address["dictionary"])
    while is_found is False and sd_addr != 0:
        sd_data = dict_strands[sd_addr]
        pl_data, ns_addr, ps_addr = \
            decodeDNAStrand(sd_data, "dictionary")
        md_idx = int(math.ceil((st_idx + ed_idx)/2))
        is_found, of_idx = find_item(str_item, pl_data)
        if is_found is False and of_idx < 0:
            sd_addr = ps_addr
            ed_idx = md_idx - 1
        elif is_found is False and of_idx > 0:
            sd_addr = ns_addr
            if (ed_idx - st_idx) % 2 == 0:
                st_idx = md_idx + 1
            else:
                st_idx = md_idx
    is_found = False
    sd_addr = dict_mid_address["bitmap_dic"]

    while is_found is False and sd_addr != 0:
        sd_data = dict_strands[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decodeDNAStrand(sd_data, "bitmap")
        tp_idx = ct_zero + get_count(str(pl_data), '0')
        if md_idx > ct_zero and (md_idx <= tp_idx):
            is_found = True
            md_idx = md_idx - ct_zero
            rt_idx = ct_one + get_n_count(str(pl_data), '1',
                                          md_idx, '0') + of_idx
        elif md_idx > tp_idx:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr
    return rt_idx


# function for getting a range to index POS table corresponding to a predicate ID
def get_range_using_bitmap_P(pred_id, dict_mid_address, dict_strands):
    is_found = False
    sd_addr = dict_mid_address["bitmap_pos"]
    while is_found is False and sd_addr != 0:
        sd_data = dict_strands[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decodeDNAStrand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if (pred_id > ct_zero) and (pred_id <= temp_id):
            is_found = True
        elif pred_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr
    pred_id = pred_id - ct_zero
    st_idx = get_n_count(str(pl_data), '1', pred_id, '0') + 1
    ed_idx = get_n_count(str(pl_data), '1', pred_id+1, '0')
    return st_idx, ed_idx


# function for getting a range to index POS table corresponding to a subject ID
def get_range_using_bitmap_S(subj_id, dict_mid_address, dict_strands):
    is_found = False
    sd_addr = dict_mid_address["bitmap_spo"]
    while is_found is False and sd_addr != 0:
        sd_data = dict_strands[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decodeDNAStrand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if (subj_id > ct_zero) and (subj_id <= temp_id):
            is_found = True
        elif subj_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr
    subj_id = subj_id - ct_zero
    st_idx = get_n_count(str(pl_data), '1', subj_id, '0') + 1
    ed_idx = get_n_count(str(pl_data), '1', subj_id + 1, '0')
    return st_idx, ed_idx


# function for getting a range to index OSP table corresponding to a object ID
def get_range_using_bitmap_O(obj_id, dict_mid_address, dict_strands):
    is_found = False
    sd_addr = dict_mid_address["bitmap_osp"]
    while is_found is False and sd_addr != 0:
        sd_data = dict_strands[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decodeDNAStrand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if (obj_id > ct_zero) and (obj_id <= temp_id):
            is_found = True
        elif obj_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr
    obj_id = obj_id - ct_zero
    st_idx = get_n_count(str(pl_data), '1', obj_id, '0') + 1
    ed_idx = get_n_count(str(pl_data), '1', obj_id + 1, '0')
    return st_idx, ed_idx


def get_strand_range(st_idx, ed_idx, num_per_srd, ctr):
    if ctr == 1:
        if math.ceil(st_idx / num_per_srd) == math.ceil(st_idx / num_per_srd):
            return st_idx, ed_idx
        else:
            return st_idx % num_per_srd, \
                   math.ceil((st_idx / num_per_srd) * num_per_srd)
    else:
        n = math.ceil((ed_idx - st_idx) / num_per_srd) + 1
        if ed_idx - st_idx > num_per_srd and ctr < n:
            return 1, num_per_srd
        else:
            return 1, ed_idx % num_per_srd


def get_matching_Ids(pl_data, st_idx, ed_idx, id):
    temp_list = []
    for i, (obj_id, subj_id) in enumerate(pl_data):
        if obj_id == id and (st_idx <= (i+1) <= ed_idx):
            temp_list.append(subj_id)
    return temp_list


# function for getting a list of IDs using a range of values to index POS table
def get_ids_using_lookup_POS(st_idx, ed_idx,
                         obj_id, dict_mid_addr, dict_strands, num_per_strand):
    std_len = num_per_strand
    ctr = 1
    ls_sub_ids = []
    do_loop = math.ceil((ed_idx - st_idx) / std_len) + 1
    sd_addr = dict_mid_addr["index_pos"]
    while ctr < do_loop:
        is_found = False
        st_idx_n, ed_idx_n = get_strand_range(st_idx, ed_idx, std_len, ctr)
        while is_found is False and sd_addr != 0:
            sd_data = dict_strands[sd_addr]
            pl_data, prv_start, prv_end, ns_addr, ps_addr = \
                decodeDNAStrand(sd_data, "index_table")
            if ed_idx_n > prv_end:
                sd_addr = ns_addr
            elif st_idx > prv_start and (ed_idx <= prv_end):
                is_found = True
                ctr = ctr + 1
                st_idx_n = st_idx_n - prv_start + 1
                ed_idx_n = ed_idx_n - prv_start + 1
                temp_list = get_matching_Ids(pl_data, st_idx_n, ed_idx_n, obj_id)
                ls_sub_ids.extend(temp_list)
                temp_list.clear()
            else:
                sd_addr = ps_addr
    return ls_sub_ids


# function for getting a list of IDs using a range of values to index SPO table
def get_ids_using_lookup_SPO(st_idx, ed_idx, pred_id, dict_mid_addr,
                         dict_strands, num_per_strand):
    std_len = num_per_strand
    ctr = 1
    ls_obj_ids = []
    do_loop = math.ceil((ed_idx - st_idx) / std_len) + 1
    sd_addr = dict_mid_addr["index_spo"]
    while ctr < do_loop:
        is_found = False
        st_idx_n, ed_idx_n = get_strand_range(st_idx, ed_idx, std_len, ctr)
        while is_found is False and sd_addr != 0:
            sd_data = dict_strands[sd_addr]
            pl_data, prv_start, prv_end, ns_addr, ps_addr = \
                decodeDNAStrand(sd_data, "index_table")
            if ed_idx_n > prv_end:
                sd_addr = ns_addr
            elif st_idx > prv_start and (ed_idx <= prv_end):
                print("asad", st_idx_n, ed_idx_n, prv_start)
                is_found = True
                ctr = ctr + 1
                st_idx_n = st_idx_n - prv_start + 1
                ed_idx_n = ed_idx_n - prv_start + 1
                print("asad", st_idx_n, ed_idx_n, prv_start)
                temp_list = get_matching_Ids(pl_data, st_idx_n, ed_idx_n, pred_id)
                ls_obj_ids.extend(temp_list)
                temp_list.clear()
            else:
                sd_addr = ps_addr
    return ls_obj_ids


# function for getting a list of IDs using a range of values to index OSP table
def get_ids_using_lookup_OSP(st_idx, ed_idx, subj_id, dict_mid_addr,
                         dict_strands, num_per_strand):
    std_len = num_per_strand
    ctr = 1
    ls_pred_ids = []
    do_loop = math.ceil((ed_idx - st_idx) / std_len) + 1
    sd_addr = dict_mid_addr["index_osp"]
    while ctr < do_loop:
        is_found = False
        st_idx_n, ed_idx_n = get_strand_range(st_idx, ed_idx, std_len, ctr)
        while is_found is False and sd_addr != 0:
            sd_data = dict_strands[sd_addr]
            pl_data, prv_start, prv_end, ns_addr, ps_addr = \
                decodeDNAStrand(sd_data, "index_table")
            if ed_idx_n > prv_end:
                sd_addr = ns_addr
            elif st_idx > prv_start and (ed_idx <= prv_end):
                is_found = True
                ctr = ctr + 1
                st_idx_n = st_idx_n - prv_start + 1
                ed_idx_n = ed_idx_n - prv_start + 1
                temp_list = get_matching_Ids(
                    pl_data, st_idx_n, ed_idx_n, subj_id)
                ls_pred_ids.extend(temp_list)
                temp_list.clear()
            else:
                sd_addr = ps_addr
    return ls_pred_ids


def create_RDF_triple_table():
    # create data
    data = [["Spanish Team", "represent", "Spain"],
            ["Madrid", "capital", "Spain"],
            ["Iker Casillas", "born", "Madrid"],
            ["Iker Casillas", "playsFor", "Spanish Team"],
            ["Iker Casillas", "position", "goalkeeper"],
            ["Iker Casillas", "captain", "Spanish Team"],
            ["Iniesta", "playsFor", "Spanish Team"],
            ["Iniesta", "position", "midfielder"],
            ["Xavi", "playsFor", "Spanish Team"],
            ["Xavi", "position", "midfielder"]]
    # define header names
    col_names = ["Subject", "Property", "Object"]

    # display table
    print(tabulate(data, headers=col_names))
    return data


def convert_to_bitmap(tab_col, n_dict_elm=20):
    j = 1
    bitmap_val = ""
    prev_val = 0
    for i in tab_col:
        while j < i:
            bitmap_val = bitmap_val + "0"
            j = j + 1
        if prev_val is not i:
            bitmap_val = bitmap_val + "01"
            j = j + 1
        else:
            bitmap_val = bitmap_val + "1"
        prev_val = i
    while j <= n_dict_elm:
        bitmap_val = bitmap_val + "0"
        j = j + 1
    return bitmap_val


def generate_index_bitmaps(tab_spo, tab_pos, tab_osp, n_dict_elm):

    tab_spo = sorted(tab_spo, key=itemgetter(0, 1, 2))
    tab_pos = sorted(tab_pos, key=itemgetter(0, 1, 2))
    tab_osp = sorted(tab_osp, key=itemgetter(0, 1, 2))

    bitmap_spo = [row[0] for row in tab_spo]
    bitmap_pos = [row[0] for row in tab_pos]
    bitmap_osp = [row[0] for row in tab_osp]

    tab_spo = list(map(itemgetter(1, 2), tab_spo))
    tab_pos = list(map(itemgetter(1, 2), tab_pos))
    tab_osp = list(map(itemgetter(1, 2), tab_osp))

    bm_spo = convert_to_bitmap(bitmap_spo, n_dict_elm)
    bm_pos = convert_to_bitmap(bitmap_pos, n_dict_elm)
    bm_osp = convert_to_bitmap(bitmap_osp, n_dict_elm)

    return tab_spo, tab_pos, tab_osp, bm_spo, bm_pos, bm_osp


def generate_index_tables(rdf_spo):
    rdf_pos = rdf_spo.copy()
    rdf_osp = rdf_spo.copy()
    for i, [x, y, z] in enumerate(rdf_spo):
        rdf_pos[i] = [y, z, x]
        rdf_osp[i] = [z, x, y]
    return rdf_spo, rdf_pos, rdf_osp


def convert_ID_based_triple_storage(n_tuples):
    dict_items = {}
    rdf_triples = []
    list_items = []
    for row in n_tuples:
        for col in row:
            list_items.append(str(col))
    list_items = sorted(set(list_items))
    for i, j in enumerate(list_items, 1):
        dict_items[j] = i
    for row in n_tuples:
        rdf_tuple = []
        for col in row:
            item_id = dict_items.get(str(col))
            rdf_tuple.append(item_id)
        rdf_triples.append(rdf_tuple)
    return dict_items, rdf_triples


def binary_search(dict, val, low, high):
    if low > high:
        return False
    else:
        mid = int((low + high) / 2)
        if val == mid:
            return
        elif val > mid:
            n_addr = int((mid + 1 + high) / 2)
            (a, b) = dict.get(mid)
            dict[mid] = (n_addr, b)
            return binary_search(dict, val, mid + 1, high)
        else:
            p_addr = int((low + mid - 1) / 2)
            (a, b) = dict.get(mid)
            dict[mid] = (a, p_addr)
            return binary_search(dict, val, low, mid - 1)


def map_to_dna_strands(t_dic, t_spo, t_pos, t_osp,
                       b_spo, b_pos, b_osp, INT_SIZE = 2,
                       DNA_STRAND_SIZE = 28):

    # Mapping dictionary items into as many dna strands as needed
    dic_srds = {}
    dic_adrs = {}
    lst_srds = []
    tmp_srd = []
    dic_bm = "0"
    srd_cnt = 1
    for nxt_str in t_dic:
        nxt_str = nxt_str + "#"
        len_str = reduce(lambda count, i: count + len(i), tmp_srd, 0)
        if (len_str + len(nxt_str)) <= DNA_STRAND_SIZE:
            tmp_srd.append(nxt_str)
            dic_bm = dic_bm + "1"
        else:
            lst_srds.append(tmp_srd.copy())
            tmp_srd.clear()
            tmp_srd.append(nxt_str)
            dic_bm = dic_bm + "01"
    lst_srds.append(tmp_srd.copy())

    dic_srds_cnt = len(lst_srds)
    high_val = len(lst_srds) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "dictionary")
    dic_adrs["dictionary"] = mid

    # Mapping all bitmaps (dictionary, spo, pos and osp index tables)
    # into as many dna strands as needed
    lst_srds.clear()
    b_size = DNA_STRAND_SIZE - (2 * INT_SIZE)
    b_list = [dic_bm[j:j + b_size] for j in range(0, len(dic_bm), b_size)]
    for a in b_list:
        lst_srds.append([a])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_dic"] = mid

    lst_srds.clear()
    b_list = [b_spo[k:k + b_size] for k in range(0, len(b_spo), b_size)]
    for b in b_list:
        lst_srds.append([b])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_spo"] = mid

    lst_srds.clear()
    b_list = [b_pos[l:l + b_size] for l in range(0, len(b_pos), b_size)]
    for c in b_list:
        lst_srds.append([c])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_pos"] = mid

    lst_srds.clear()
    b_list = [b_osp[m:m + b_size] for m in range(0, len(b_osp), b_size)]
    for d in b_list:
        lst_srds.append([d])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_osp"] = mid

    # Mapping all index table (spo, pos and osp)
    # into as many dna strands as needed
    lst_srds.clear()
    b_size = int((DNA_STRAND_SIZE / (2 * INT_SIZE)) - 1)
    b_list = [t_spo[n:n + b_size] for n in range(0, len(t_spo), b_size)]
    for e in b_list:
        lst_srds.append(e)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_spo"] = mid

    lst_srds.clear()
    b_list = [t_pos[o:o + b_size] for o in range(0, len(t_pos), b_size)]
    for f in b_list:
        lst_srds.append(f)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_pos"] = mid

    lst_srds.clear()
    b_list = [t_osp[p:p + b_size] for p in range(0, len(t_osp), b_size)]
    for g in b_list:
        lst_srds.append(g)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_osp"] = mid

    return dic_srds_cnt, dic_srds, dic_adrs


def compose_dna_strand(dic_srds, srd_cnt, high_val, lst_srds, srd_type):
    prv_cnt = srd_cnt
    my_dict = {my_list: (0, 0) for my_list in range(1, high_val)}
    for i in range(1, high_val):
        binary_search(my_dict, i, 1, high_val)
    mid = int((1 + high_val) / 2) + srd_cnt - 1
    prv_zeros = 0
    prv_ones = 0
    cnt_int = 1
    for j in range(1, high_val):
        (nxt_adr, prv_adr) = my_dict[j]
        if nxt_adr > 0:
            nxt_adr = nxt_adr + prv_cnt - 1
        if prv_adr > 0:
            prv_adr = prv_adr + prv_cnt - 1
        if srd_type == "dictionary":
            dic_srds[srd_cnt] = [[srd_cnt], lst_srds[j - 1],
                                 [nxt_adr], [prv_adr]]
        elif srd_type == "bitmap":
            dic_srds[srd_cnt] = [[srd_cnt], lst_srds[j - 1], [prv_zeros],
                                 [prv_ones], [nxt_adr], [prv_adr]]
            prv_zeros = get_count(str(lst_srds[j - 1][0]), '0')
            prv_ones = get_count(str(lst_srds[j - 1][0]), '1')
        else:
            dic_srds[srd_cnt] = [
                [srd_cnt], lst_srds[j - 1], [cnt_int],
                [cnt_int + len(lst_srds[j - 1])-1], [nxt_adr], [prv_adr]]
            cnt_int = cnt_int + len(lst_srds[j - 1])
        srd_cnt = srd_cnt + 1

    return mid, srd_cnt


if __name__ == '__main__':

    pair_per_strand = 6

    print("\nCreating tuples from RDF triple table....\n")
    data_tuples = create_RDF_triple_table()

    t_dic, t_rdf = \
        convert_ID_based_triple_storage(data_tuples)

    t_spo, t_pos, t_osp = \
        generate_index_tables(t_rdf)

    t_spo, t_pos, t_osp, b_spo, b_pos, b_osp = \
        generate_index_bitmaps(t_spo, t_pos, t_osp, len(t_dic))

    dict_strands_count, dict_of_strands, dict_of_mid_address = \
        map_to_dna_strands(t_dic, t_spo, t_pos, t_osp, b_spo, b_pos, b_osp)

    print("\nMapping to DNA Strands...\n")
    for i, x in enumerate(dict_of_strands):
        print("Strand", i+1, dict_of_strands[x])

    print("\nMapping Queries to DNA Strands...\n")

    print("............. Query One .............!")
    pre_id = map_RDF_str_to_Id("playsFor",
                               dict_strands_count, dict_of_mid_address, dict_of_strands)
    print("Predicate=playsFor id:", pre_id)

    obj_id = map_RDF_str_to_Id("Spanish Team",
                               dict_strands_count, dict_of_mid_address, dict_of_strands)
    print("Object=Spanish team id:", obj_id)

    s_idx, e_idx = get_range_using_bitmap_P(pre_id, dict_of_mid_address, dict_of_strands)

    subj_ids = get_ids_using_lookup_POS(s_idx, e_idx, obj_id,
                                    dict_of_mid_address, dict_of_strands, pair_per_strand)
    print("Subject id(s):", subj_ids)
