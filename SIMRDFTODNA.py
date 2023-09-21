import csv
import re
import math
import pprint
from typing import Dict, Any

from tabulate import tabulate
from operator import itemgetter
from functools import reduce

import csv


def decode_dna_strand(sd_data, dc_type):
    if dc_type == "dictionary":
        return sd_data[1], sd_data[2][0], sd_data[3][0]
    elif dc_type == "bitmap":
        return sd_data[1][0], sd_data[2][0], sd_data[3][0], \
               sd_data[4][0], sd_data[5][0],
    else:
        return sd_data[1], sd_data[2][0], sd_data[3][0], \
               sd_data[4][0], sd_data[5][0],


def find_mid(st_idx, ed_idx):
    mid = int((st_idx + ed_idx) / 2)
    return mid


def find_item(str_item, pl_data):
    str_item = str_item + '#'
    str_last = ''
    for idx, nxt_str in enumerate(pl_data):
        if nxt_str == str_item:
            return True, idx + 1
        str_last = nxt_str
    if str_item > str_last:
        return False, 0
    else:
        return False, -1


def get_n_count(str_data, c1, n, c2):
    inlist = [m.start() for m in re.finditer(c2, str_data)]
    return int(str_data[0:inlist[n - 1]].count(c1))


def get_count(str_data, c):
    return int(str_data.count(c))


# function for mapping string to corresponding ID in the Dictionary table
def map_id_to_rdf_string(str_id, sd_cnt, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 1
    srt_idx = 0
    off_idx = 0
    is_found = False
    ed_idx = sd_cnt
    sd_addr = dic_mid_addr["bitmap_dic"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, prv_zero, prv_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        crn_one = get_count(str(pl_data), '1') + prv_one
        if prv_one < str_id <= crn_one:
            is_found = True
            str_id = str_id - prv_one
            cnt_zero = get_n_count(str(pl_data), '0', str_id, '1')
            cnt_one = get_n_count(str(pl_data), '1', cnt_zero, '0')
            off_idx = str_id - cnt_one - 1
            srt_idx = cnt_zero + prv_zero
        elif str_id <= prv_one:
            sd_addr = ps_addr
        else:
            sd_addr = ns_addr
    if is_found is True:
        sd_addr = dic_mid_addr["dictionary"]
        is_found = False
        while is_found is False and sd_addr != 0:
            if tmp_srds.get(sd_addr) is None:
                sd_data = dic_srds[sd_addr]
                tmp_srds[sd_addr] = sd_data
            else:
                sd_data = tmp_srds[sd_addr]
            pl_data, ns_addr, ps_addr = \
                decode_dna_strand(sd_data, "dictionary")
            md_idx = find_mid(st_idx, ed_idx)
            if is_found is False and srt_idx < md_idx:
                sd_addr = ps_addr
                ed_idx = md_idx - 1
            elif is_found is False and srt_idx > md_idx:
                sd_addr = ns_addr
                st_idx = md_idx + 1
            else:
                is_found = True
        return sd_data[1][off_idx]
    else:
        return None


# function for mapping string to corresponding ID in the Dictionary table
def map_rdf_string_to_id(str_item, sd_cnt, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 1
    md_idx = 0
    rt_idx = 0
    is_found = False
    ed_idx = sd_cnt
    sd_addr = int(dic_mid_addr["dictionary"])
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "dictionary")
        is_found, of_idx = find_item(str_item, pl_data)
        md_idx = find_mid(st_idx, ed_idx)
        if is_found is False and of_idx == -1:
            sd_addr = ps_addr
            ed_idx = md_idx - 1
        elif is_found is False and of_idx == 0:
            sd_addr = ns_addr
            st_idx = md_idx + 1
    if is_found is False:
        return 0
    is_found = False
    sd_addr = dic_mid_addr["bitmap_dic"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
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
def get_range_using_bitmap_p(prd_id, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 0
    ed_idx = 0
    is_found = False
    sd_addr = dic_mid_addr["bitmap_pos"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if ct_zero < prd_id <= temp_id:
            st_idx = get_n_count(str(pl_data),
                                 '1', prd_id - ct_zero, '0') + 1
            st_idx = st_idx + ct_one
            is_found = True
        elif prd_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr

    is_found = False
    prd_id = prd_id + 1
    sd_addr = dic_mid_addr["bitmap_pos"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if ct_zero < prd_id <= temp_id:
            ed_idx = get_n_count(str(pl_data),
                                 '1', prd_id - ct_zero, '0')
            ed_idx = ed_idx + ct_one
            is_found = True
        elif prd_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr

    print("======================getrange_p:asad bitmap", st_idx, ed_idx)
    return st_idx, ed_idx


# function for getting a range to index POS table corresponding to a subject ID
def get_range_using_bitmap_s(sub_id, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 0
    ed_idx = 0
    is_found = False
    sd_addr = dic_mid_addr["bitmap_spo"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if ct_zero < sub_id <= temp_id:
            st_idx = get_n_count(str(pl_data),
                                 '1', sub_id - ct_zero, '0') + 1
            st_idx = st_idx + ct_one
            is_found = True
        elif sub_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr

    is_found = False
    sub_id = sub_id + 1
    sd_addr = dic_mid_addr["bitmap_spo"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if ct_zero < sub_id <= temp_id:
            ed_idx = get_n_count(str(pl_data),
                                 '1', sub_id - ct_zero, '0')
            ed_idx = ed_idx + ct_one
            is_found = True
        elif sub_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr

    print("======================getrange_s:asad bitmap", st_idx, ed_idx)
    return st_idx, ed_idx


# function for getting a range to index OSP table corresponding to a object ID
def get_range_using_bitmap_o(obj_id, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 0
    ed_idx = 0
    is_found = False
    sd_addr = dic_mid_addr["bitmap_osp"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if ct_zero < obj_id <= temp_id:
            st_idx = get_n_count(str(pl_data),
                                 '1', obj_id - ct_zero, '0') + 1
            st_idx = st_idx + ct_one
            is_found = True
        elif obj_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr

    is_found = False
    obj_id = obj_id + 1
    sd_addr = dic_mid_addr["bitmap_osp"]
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, ct_zero, ct_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        temp_id = ct_zero + get_count(str(pl_data), '0')
        if ct_zero < obj_id <= temp_id:
            ed_idx = get_n_count(str(pl_data),
                                 '1', obj_id - ct_zero, '0')
            ed_idx = ed_idx + ct_one
            is_found = True
        elif obj_id > temp_id:
            sd_addr = ns_addr
        else:
            sd_addr = ps_addr

    print("======================getrange_o:asad bitmap", st_idx, ed_idx)
    return st_idx, ed_idx


def get_strand_range(st_idx, ed_idx, num_per_srd):

    rng_str = num_per_srd * math.ceil(st_idx / num_per_srd) - num_per_srd
    rng_end = num_per_srd * math.ceil(st_idx / num_per_srd) + 1
    if (rng_str <= st_idx < rng_end) and (rng_str <= ed_idx < rng_end):
        return 0, 0, st_idx, ed_idx
    else:
        return int(rng_end), int(ed_idx), \
               int(st_idx), int(rng_end - 1)


def get_matching_ids(pl_data, st_idx, ed_idx, mid):
    temp_list = []
    for rng_idx, (obj_id, subj_id) in enumerate(pl_data):
        if obj_id == mid and (st_idx <= (rng_idx + 1) <= ed_idx):
            temp_list.append(subj_id)
        elif mid == 0 and (st_idx <= (rng_idx + 1) <= ed_idx):
            temp_list.append((obj_id, subj_id))
    return temp_list


# function for getting a list of IDs using a range of values to index POS table
def get_ids_using_lookup_pos(st_idx, ed_idx, obj_id, dic_mid_addr,
                             dic_srds, num_per_strand, tmp_srds):
    std_len = num_per_strand
    ls_sub_ids = []
    sd_addr = dic_mid_addr["index_pos"]
    while st_idx > 0:
        is_found = False
        st_idx, ed_idx, st_idx_n, ed_idx_n = \
            get_strand_range(st_idx, ed_idx, std_len)
        while is_found is False and sd_addr != 0:
            if tmp_srds.get(sd_addr) is None:
                sd_data = dic_srds[sd_addr]
                tmp_srds[sd_addr] = sd_data
            else:
                sd_data = tmp_srds[sd_addr]
            pl_data, prv_start, prv_end, ns_addr, ps_addr = \
                decode_dna_strand(sd_data, "index_table")
            if ed_idx_n > prv_end:
                sd_addr = ns_addr
            elif st_idx_n >= prv_start and ed_idx_n <= prv_end:
                is_found = True
                st_idx_n = st_idx_n - prv_start + 1
                ed_idx_n = ed_idx_n - prv_start + 1
                temp_list = get_matching_ids(pl_data, st_idx_n, ed_idx_n, obj_id)
                ls_sub_ids.extend(temp_list)
                temp_list.clear()
            else:
                sd_addr = ps_addr
    return ls_sub_ids


# function for getting a list of IDs using a range of values to index SPO table
def get_ids_using_lookup_spo(st_idx, ed_idx, prd_id, dic_mid_addr,
                             dic_srds, num_per_strand, tmp_srds):
    std_len = num_per_strand
    ls_obj_ids = []
    sd_addr = dic_mid_addr["index_spo"]
    while st_idx > 0:
        is_found = False
        st_idx, ed_idx, st_idx_n, ed_idx_n = \
            get_strand_range(st_idx, ed_idx, std_len)
        while is_found is False and sd_addr != 0:
            if tmp_srds.get(sd_addr) is None:
                sd_data = dic_srds[sd_addr]
                tmp_srds[sd_addr] = sd_data
            else:
                sd_data = tmp_srds[sd_addr]
            pl_data, prv_start, prv_end, ns_addr, ps_addr = \
                decode_dna_strand(sd_data, "index_table")
            if ed_idx_n > prv_end:
                sd_addr = ns_addr
            elif st_idx_n >= prv_start and ed_idx_n <= prv_end:
                is_found = True
                st_idx_n = st_idx_n - prv_start + 1
                ed_idx_n = ed_idx_n - prv_start + 1
                temp_list = get_matching_ids(pl_data, st_idx_n, ed_idx_n, prd_id)
                ls_obj_ids.extend(temp_list)
                temp_list.clear()
            else:
                sd_addr = ps_addr
    return ls_obj_ids


# function for getting a list of IDs using a range of values to index OSP table
def get_ids_using_lookup_osp(st_idx, ed_idx, sub_id, dic_mid_addr,
                             dic_srds, num_per_strand, tmp_srds):
    std_len = num_per_strand
    ls_prd_ids = []
    sd_addr = dic_mid_addr["index_osp"]
    while st_idx > 0:
        is_found = False
        st_idx, ed_idx, st_idx_n, ed_idx_n = \
            get_strand_range(st_idx, ed_idx, std_len)
        while is_found is False and sd_addr != 0:
            if tmp_srds.get(sd_addr) is None:
                sd_data = dic_srds[sd_addr]
                tmp_srds[sd_addr] = sd_data
            else:
                sd_data = tmp_srds[sd_addr]
            pl_data, prv_start, prv_end, ns_addr, ps_addr = \
                decode_dna_strand(sd_data, "index_table")
            if ed_idx_n > prv_end:
                sd_addr = ns_addr
            elif st_idx_n >= prv_start and ed_idx_n <= prv_end:
                is_found = True
                st_idx_n = st_idx_n - prv_start + 1
                ed_idx_n = ed_idx_n - prv_start + 1
                temp_list = get_matching_ids(pl_data, st_idx_n, ed_idx_n, sub_id)
                ls_prd_ids.extend(temp_list)
                temp_list.clear()
            else:
                sd_addr = ps_addr
    return ls_prd_ids


def count_minimum_strands(st_idx, ed_idx, std_len):
    s = math.ceil(st_idx/std_len)
    e = math.ceil(ed_idx/std_len)
    return e - s + 1


def convert_to_bitmap(tab_col, n_dict_elm):
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
    bitmap_val = bitmap_val + "0"
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


def convert_id_based_triple_storage(n_tpl):
    dict_items = {}
    rdf_triples = []
    list_items = []
    for row in n_tpl:
        for col in row:
            list_items.append(str(col))
    list_items = sorted(set(list_items))
    for i, j in enumerate(list_items, 1):
        dict_items[j] = i
    for row in n_tpl:
        rdf_tuple = []
        for col in row:
            item_id = dict_items.get(str(col))
            rdf_tuple.append(item_id)
        rdf_triples.append(rdf_tuple)
    return dict_items, rdf_triples


def binary_search(tmp_dic, val, low, high):
    if low > high:
        return False
    else:
        mid = int((low + high) / 2)
        if val == mid:
            return
        elif val > mid:
            n_addr = int((mid + 1 + high) / 2)
            (a, b) = tmp_dic.get(mid)
            tmp_dic[mid] = (n_addr, b)
            return binary_search(tmp_dic, val, mid + 1, high)
        else:
            p_addr = int((low + mid - 1) / 2)
            (a, b) = tmp_dic.get(mid)
            tmp_dic[mid] = (a, p_addr)
            return binary_search(tmp_dic, val, low, mid - 1)


def binary_search2(tmp_dic, val, low, high):
    if low > high:
        return False
    else:
        while low < high:
            mid = int((low + high) / 2)
            if val == mid:
                return
            elif val > mid:
                n_addr = int((mid + 1 + high) / 2)
                p_addr = int((low + mid - 1) / 2)
                tmp_dic[mid] = (n_addr, p_addr)
                low = mid + 1
            else:
                n_addr = int((mid + 1 + high) / 2)
                p_addr = int((low + mid - 1) / 2)
                tmp_dic[mid] = (n_addr, p_addr)
                high = mid - 1


def map_to_dna_strands(t_dic, t_spo, t_pos, t_osp,
                       b_spo, b_pos, b_osp, INT_SIZE,
                       DNA_STRAND_SIZE):
    # Mapping dictionary items into as many dna strands as needed
    dic_srds = {}
    dic_adrs = {}
    lst_srds = []
    tmp_srd = []
    dic_bm = "0"
    srd_cnt = 1
    idx_str_cnt = 0
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
    dic_bm = dic_bm + "0"
    lst_srds.append(tmp_srd.copy())

    dic_srds_cnt = len(lst_srds) + 1
    high_val = len(lst_srds) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "dictionary")
    dic_adrs["dictionary"] = mid

    # Mapping all bitmaps (dictionary, spo, pos and osp index tables)
    # into as many dna strands as needed
    lst_srds.clear()
    b_size = ((DNA_STRAND_SIZE - (2 * INT_SIZE)) * 8)
    b_list = [dic_bm[j:j + b_size] for j in range(0, len(dic_bm), b_size)]
    for a in b_list:
        lst_srds.append([a])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_dic"] = mid
    print("bitmap_dic:strands count", high_val-1)

    lst_srds.clear()
    b_list = [b_spo[k:k + b_size] for k in range(0, len(b_spo), b_size)]
    for b in b_list:
        lst_srds.append([b])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_spo"] = mid
    print("bitmap_spo:strands count", high_val-1)

    lst_srds.clear()
    b_list = [b_pos[l:l + b_size] for l in range(0, len(b_pos), b_size)]
    for c in b_list:
        lst_srds.append([c])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_pos"] = mid
    print("bitmap_pos:strands count", high_val-1)

    lst_srds.clear()
    b_list = [b_osp[m:m + b_size] for m in range(0, len(b_osp), b_size)]
    for d in b_list:
        lst_srds.append([d])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_osp"] = mid
    print("bitmap_osp:strands count", high_val-1)
    idx_str_cnt = high_val - 1

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
    idx_str_cnt = 2 * (idx_str_cnt + high_val - 1)

    lst_srds.clear()
    b_list = [t_pos[o:o + b_size] for o in range(0, len(t_pos), b_size)]
    for f in b_list:
        lst_srds.append(f)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_pos"] = mid
    print("index_pos:strands count", high_val - 1)

    lst_srds.clear()
    b_list = [t_osp[p:p + b_size] for p in range(0, len(t_osp), b_size)]
    for g in b_list:
        lst_srds.append(g)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_osp"] = mid
    print("index_osp:strands count", high_val - 1)

    return dic_srds_cnt, dic_srds, dic_adrs, idx_str_cnt


def compose_dna_strand(dic_srds, srd_cnt, high_val, lst_srds, srd_type):
    prv_cnt = srd_cnt
    my_dict = {my_list: (0, 0) for my_list in range(1, high_val)}
    for i in range(1, high_val):
        binary_search2(my_dict, i, 1, high_val)
    mid = int((1 + high_val) / 2) + srd_cnt - 1
    prv_zero = 0
    prv_ones = 0
    cnt_int = 1
    for j in range(1, high_val):
        (nxt_adr, prv_adr) = my_dict[j]
        if j == high_val - 1:
            nxt_adr = 0
        if nxt_adr > 0:
            nxt_adr = nxt_adr + prv_cnt - 1
        if prv_adr > 0:
            prv_adr = prv_adr + prv_cnt - 1
        if srd_type == "dictionary":
            dic_srds[srd_cnt] = [[srd_cnt], lst_srds[j - 1],
                                 [nxt_adr], [prv_adr]]
        elif srd_type == "bitmap":
            dic_srds[srd_cnt] = [[srd_cnt], lst_srds[j - 1], [prv_zero],
                                 [prv_ones], [nxt_adr], [prv_adr]]
            prv_zero = prv_zero + get_count(str(lst_srds[j - 1][0]), '0')
            prv_ones = prv_ones + get_count(str(lst_srds[j - 1][0]), '1')
        else:
            dic_srds[srd_cnt] = [
                [srd_cnt], lst_srds[j - 1], [cnt_int],
                [cnt_int + len(lst_srds[j - 1]) - 1], [nxt_adr], [prv_adr]]
            cnt_int = cnt_int + len(lst_srds[j - 1])
        srd_cnt = srd_cnt + 1
    return mid, srd_cnt


def map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds):
    print("............. Query Processing.............!")
    ls_out = []
    if qr_type == "?PO":
        pre_id = map_rdf_string_to_id(prd_str,
                                   cnt_dic_srds, dic_mid_addr,
                                   dic_srds, tmp_dic_srds)

        obj_id = map_rdf_string_to_id(obj_str,
                                   cnt_dic_srds, dic_mid_addr,
                                   dic_srds, tmp_dic_srds)

        s_idx, e_idx = get_range_using_bitmap_p(pre_id, dic_mid_addr,
                                                dic_srds, tmp_dic_srds)

        sub_ids = get_ids_using_lookup_pos(s_idx, e_idx, obj_id,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)
        for sid in sub_ids:
            sub_str = map_id_to_rdf_string(sid, cnt_dic_srds, dic_mid_addr,
                                       dic_srds, tmp_dic_srds)
            ls_out.append(sub_str)

        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)

    elif qr_type == "S?O":
        sub_id = map_rdf_string_to_id(sub_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)
        obj_id = map_rdf_string_to_id(obj_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)
        s_idx, e_idx = get_range_using_bitmap_o(obj_id, dic_mid_addr,
                                                dic_srds, tmp_dic_srds)
        prd_ids = get_ids_using_lookup_osp(s_idx, e_idx, sub_id,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)
        for sid in prd_ids:
            prd_str = map_id_to_rdf_string(sid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            ls_out.append(prd_str)
        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)

    elif qr_type == "SP?":
        sub_id = map_rdf_string_to_id(sub_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)
        prd_id = map_rdf_string_to_id(prd_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)
        s_idx, e_idx = get_range_using_bitmap_s(sub_id, dic_mid_addr,
                                                dic_srds, tmp_dic_srds)

        obj_ids = get_ids_using_lookup_spo(s_idx, e_idx, prd_id,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)
        for oid in obj_ids:
            obj_str = map_id_to_rdf_string(oid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            ls_out.append(obj_str)
        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)

    elif qr_type == "??O":
        obj_id = map_rdf_string_to_id(obj_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)
        s_idx, e_idx = get_range_using_bitmap_o(obj_id, dic_mid_addr,
                                                dic_srds, tmp_dic_srds)

        obj_ids = get_ids_using_lookup_osp(s_idx, e_idx, 0,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)
        for (sid, pid) in obj_ids:
            sub_str = map_id_to_rdf_string(sid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            prd_str = map_id_to_rdf_string(pid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            ls_out.append((sub_str, prd_str))
        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)

    elif qr_type == "?P?":
        prd_id = map_rdf_string_to_id(prd_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)
        s_idx, e_idx = get_range_using_bitmap_p(prd_id, dic_mid_addr,
                                                dic_srds, tmp_dic_srds)

        prd_ids = get_ids_using_lookup_pos(s_idx, e_idx, 0,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)

        for (oid, sid) in prd_ids:
            sub_str = map_id_to_rdf_string(sid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            obj_str = map_id_to_rdf_string(oid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            ls_out.append((sub_str, obj_str))
        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)

    elif qr_type == "S??":
        sub_id = map_rdf_string_to_id(sub_str,
                                      cnt_dic_srds, dic_mid_addr,
                                      dic_srds, tmp_dic_srds)

        s_idx, e_idx = get_range_using_bitmap_s(sub_id, dic_mid_addr,
                                                dic_srds, tmp_dic_srds)

        sub_ids = get_ids_using_lookup_spo(s_idx, e_idx, 0,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)

        for (pid, oid) in sub_ids:
            prd_str = map_id_to_rdf_string(pid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            obj_str = map_id_to_rdf_string(oid, cnt_dic_srds, dic_mid_addr,
                                           dic_srds, tmp_dic_srds)
            ls_out.append((prd_str, obj_str))
        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)


def print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds):
    print("Total Number of Strands Accessed:", len(tmp_dic_srds))
    #for idx, str_idx in enumerate(tmp_dic_srds):
    #    print("Strand", idx + 1, tmp_dic_srds[str_idx])
    if qr_type == "?PO":
        print("(Subject)", ls_out)
        for out in ls_out:
            print("SPO:", out, prd_str, obj_str)
    elif qr_type == "S?O":
        print("(Predicate):", ls_out)
        for out in ls_out:
            print("SPO:", sub_str, out, obj_str)
    elif qr_type == "SP?":
        print("(Object):", ls_out)
        for out in ls_out:
            print("SPO:", sub_str, prd_str, out)
    elif qr_type == "??O":
        print("(Subject, Predicate):", ls_out)
        for (sub_str, prd_str) in ls_out:
            print("SPO:", sub_str, prd_str, obj_str)
    elif qr_type == "?P?":
        print("(Subject, Object):", ls_out)
        for (sub_str, obj_str) in ls_out:
            print("SPO:", sub_str, prd_str, obj_str)
    elif qr_type == "S??":
        print("(Predicate, Object):", ls_out)
        for (prd_str, obj_str) in ls_out:
            print("SPO:", sub_str, prd_str, obj_str)
    else:
        print("Ah!.........No Query Pattern Matched!......")


def create_rdf_triple_table():
    # create data
    data = [["Allen_Ginsberg", "wikiPageUsesTemplate", "Template:Infobox writer"],
            ["Allen_Ginsberg", "influenced", "John_Lennon"],
            ["Allen_Ginsberg", "occupation", "”Writer, poet”@en"],
            ["Allen_Ginsberg", "influences", "Fyodor_Dostoyevsky"],
            ["Allen_Ginsberg", "deathPlace", "”New York City, United States”@en"],
            ["Allen_Ginsberg", "deathDate", "”1997-04-05”"],
            ["Allen_Ginsberg", "birthPlace", "”Newark, New Jersey, United States”@en"],
            ["Allen_Ginsberg", "birthDate", "”1926-06-03”"],
            ["Albert_Camus", "wikiPageUsesTemplate", "Template:Infobox philosopher"],
            ["Albert_Camus", "influenced", "Orhan_Pamuk"],
            ["Albert_Camus", "influences", "Friedrich_Nietzsche"],
            ["Albert_Camus", "schoolTradition", "Absurdism"],
            ["Albert_Camus", "deathPlace", "”Villeblevin, Yonne, Burgundy, France”@en"],
            ["Albert_Camus", "deathDate", "”1960-01-04”"],
            ["Albert_Camus", "birthPlace", "”Drean, El Taref, Algeria”@en"],
            ["Albert_Camus", "birthDate", "”1913-11-07"],
            ["Student49", "telephone", "”xxx-xxx-xxxx”"],
            ["Student49", "memberOf", "http://www.Department3.University0.edu"],
            ["Student49", "takesCourse", "Course32"],
            ["Student49", "name", "”UndergraduateStudent49”"],
            ["Student49", "emailAddress", "”Student49@Department3.University0.edu”"],
            ["Student49", "type", "UndergraduateStudent"],
            ["Student10", "telephone", "”xxx-xxx-xxxx”"],
            ["Student10", "memberOf", "http://www.Department3.University0.edu"],
            ["Student10", "takesCourse", "Course20"],
            ["Student10", "name", "”UndergraduateStudent10”"],
            ["Student10", "emailAddress", "”Student10@Department3.University0.edu”"],
            ["Student10", "type", "UndergraduateStudent"],
            ["SurgeryProcedure:236", "hasCardiacValveAnatomyPathologyData", "CardiacValveAnatomyPathologyData:70"],
            ["SurgeryProcedure:236", "hasCardiacValveRepairProcedureData", "CardiacValveRepairProcedureData:16"],
            ["SurgeryProcedure:236", "SurgeryProcedureClass", "”cardiac valve”"],
            ["SurgeryProcedure:236", "CardiacValveEtiology", "”other”"],
            ["SurgeryProcedure:236", "CardiacValveEtiology", "Event:184"],
            ["SurgeryProcedure:236", "belongsToEvent", "Event:184"],
            ["SurgeryProcedure:236", "SurgeryProcedureDescription", "”pulmonary valve repair”"],
            ["SurgeryProcedure:236", "CardiacValveStatusologyData", "native"],
            ["SurgeryProcedure:104", "hasCardiacValveAnatomyPathologyData", "CardiacValveAnatomyPathologyData:35"],
            ["SurgeryProcedure:104", "SurgeryProcedureClass", "”cardiac valve”"],
            ["SurgeryProcedure:104", "CardiacValveEtiology", "”rheumatic”"],
            ["SurgeryProcedure:104", "belongsToEvent", "Event:81"],
            ["SurgeryProcedure:104", "SurgeryProcedureDescription", "”mitral va”"],

            ]
    # define header names
    col_names = ["Subject", "Property", "Object"]

    # display table
    print(tabulate(data, headers=col_names))
    return data

def create_rdf_triple_table2():
    # create data
    data = [['<http://orcid.org/0000-0002-3178-0201>', 'dcterm:created', '2014-12-22T22:25:56.900Z^^xsd:dateTime'],
            ['<http://orcid.org/0000-0002-3178-0201>', '<http://www.loc.gov/mads/rdf/v1#hasAffiliation>', '<http://www.grid.ac/institutes/grid.152326.1>'],
            ['<http://orcid.org/0000-0002-3178-0201>', 'rdf:type', 'foaf:Person'],
            ['<http://orcid.org/0000-0002-3178-0201>', 'rdfs:label', 'Julian Hillyer'],
            ['<http://orcid.org/0000-0002-3178-0201>', 'foaf:familyName', 'Hillyer'],
            ['<http://orcid.org/0000-0002-3178-0201>', 'foaf:givenName', 'Julian'],
            ['<http://orcid.org/0000-0002-3178-0201>', 'dcterm:modified', '2017-08-11T21:51:34.631Z^^xsd:dateTime'],
            ['<http://orcid.org/0000-0003-2360-0589>', 'dcterm:created', '2017-05-24T13:47:09.768Z^^xsd:dateTime'],
            ['<http://orcid.org/0000-0003-2360-0589>', '<http://www.loc.gov/mads/rdf/v1#hasAffiliation>', '<http://www.grid.ac/institutes/grid.152326.1>'],
            ['<http://orcid.org/0000-0003-2360-0589>', 'rdf:type', 'foaf:Person'],
            ['<http://orcid.org/0000-0003-2360-0589>', 'rdfs:label', 'Shaul Kelner'],
            ['<http://orcid.org/0000-0003-2360-0589>', 'foaf:familyName', 'Kelner'],
            ['<http://orcid.org/0000-0003-2360-0589>', 'foaf:givenName', 'Shaul'],
            ['<http://orcid.org/0000-0003-2360-0589>', 'dcterm:modified', '2017-05-24T13:56:27.187Z^^xsd:dateTime'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', 'dcterm:creator', '<http://orcid.org/0000-0003-2360-0589>'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', 'dcterm:date', '2015-12-30'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', '<http://purl.org/ontology/bibo/pageEnd>', '4'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', '<http://purl.org/ontology/bibo/pageStart>', '1'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', 'rdf:type', 'foaf:Document'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', 'rdfs:label', 'Jews in the United States@en'],
            ['<http://dx.doi.org/10.1002/9781118663202.wberen606>', 'dcterm:title', 'Jews in the United States@en'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', 'dcterm:creator', '<http://orcid.org/0000-0002-3178-0201>'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', 'dcterm:date', '2016-05'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', '<http://purl.org/ontology/bibo/pageEnd>', '118'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', '<http://purl.org/ontology/bibo/pageStart>', '102'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', '<http://purl.org/ontology/bibo/volume>', '58'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', 'rdf:type', 'foaf:Document'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', 'rdfs:label', 'Insect immunology and hematopoiesis@en'],
            ['<http://dx.doi.org/10.1016/j.dci.2015.12.006>', 'dcterm:title', 'Insect immunology and hematopoiesis@en'],
            ['<http://dx.doi.org/10.1016/j.jinsphys.2017.06.013>', 'dcterm:creator', '<http://orcid.org/0000-0002-3178-0201>'],
            ['<http://dx.doi.org/10.1016/j.jinsphys.2017.06.013>', 'dcterm:date', '2017-08'],
            ['<http://dx.doi.org/10.1016/j.jinsphys.2017.06.013>', '<http://purl.org/ontology/bibo/pageEnd>', '56'],
            ['<http://dx.doi.org/10.1016/j.jinsphys.2017.06.013>', '<http://purl.org/ontology/bibo/pageStart>', '47'],
            ['<http://dx.doi.org/10.1016/j.jinsphys.2017.06.013>', '<http://purl.org/ontology/bibo/volume>', '101'],
            ['<http://dx.doi.org/10.1016/j.jinsphys.2017.06.013>', 'rdf:type', 'foaf:Document'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', 'dcterm:creator', '<http://orcid.org/0000-0003-2360-0589>'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', 'dcterm:date', '2011-08'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', '<http://purl.org/ontology/bibo/pageEnd>', '73'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', '<http://purl.org/ontology/bibo/pageStart>', '72'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', '<http://purl.org/ontology/bibo/volume>', '10'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', 'rdf:type', 'foaf:Document'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', 'rdfs:label', 'Let My People Go@en'],
            ['<http://dx.doi.org/10.1177/1536504211418463>', 'dcterm:title', 'Let My People Go@en'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', 'dcterm:creator', '<http://orcid.org/0000-0003-2360-0589>'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', 'dcterm:date', '2014'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', '<http://purl.org/ontology/bibo/pageEnd>', '22'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', '<http://purl.org/ontology/bibo/pageStart>', '17'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', '<http://purl.org/ontology/bibo/volume>', '98'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', 'rdf:type', 'foaf:Document'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', 'rdfs:label', 'Ethnographers and History@en'],
            ['<http://dx.doi.org/10.1353/ajh.2014.0012>', 'dcterm:title', 'Ethnographers and History@en'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.w3.org/2003/01/geo/wgs84_pos#lat>', '36.144937^^xsd:float'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.w3.org/2003/01/geo/wgs84_pos#long>', '-86.802687^^xsd:float'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.grid.ac/ontology/cityName>', 'Nashville'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.grid.ac/ontology/countryCode>', 'US'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.grid.ac/ontology/countryName>', 'United States'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.grid.ac/ontology/establishedYear>', '1873^^xsd:gYear'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.grid.ac/ontology/hasWikidataId>', '<http://www.wikidata.org/entity/Q29052>'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', '<http://www.grid.ac/ontology/wikipediaPage>', '<http://en.wikipedia.org/wiki/Vanderbilt_University>'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', 'rdf:type', '<http://www.grid.ac/ontology/Education>'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', 'rdf:type', 'foaf:Organization'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', 'rdfs:label', 'Vanderbilt University'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', 'foaf:homepage', '<http://www.vanderbilt.edu/>'],
            ['<http://www.grid.ac/institutes/grid.152326.1>', 'skos:altLabel', 'Vandy']
            ]
    # define header names
    col_names = ["Subject", "Property", "Object"]

    # display table
    print(tabulate(data, headers=col_names))
    return data


def fileWrite(filePath, fileName, fileExt):
    g = Graph()
    print("parsing start!")
    fileParseName = filePath + fileName + fileExt
    g.parse(fileParseName)
    print("parsing done!")
    v = g.serialize(format="ntriples")
    print("serialization done!")
    print(v)
    with open(filePath + fileName + ".txt", "a", encoding="utf-8") as f:
        f.write(v)


if __name__ == '__main__':

    int_size = 4
    byt_per_srd = 256
    elm_per_srd = byt_per_srd/4 - 1
    pr_len = 4 * 32

    print("\nCreating tuples from RDF triple table....\n")
    dt_tpl = create_rdf_triple_table()
    dt_tpl2 = create_rdf_triple_table2()
    dt_tpl.extend(dt_tpl2)

    filePath = "C:\\Users\\admin\\Desktop\\testRDF\\testDataset11-Yago\\dataset4_ok\\"
    fileName = "yagoWordnetIds"
    #fileWrite(filePath, fileName, ".ttl")

    with open(filePath + fileName + '.txt', 'r', encoding="utf-8") as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ')
        tmp_row = []
        for row in csv_reader:
            row = ",".join(row).split("\t")
            row = " ".join(row).split(",")
            if row[2].find("XMLSchema#dateTime") == -1:
                if len(row[0]) <= byt_per_srd \
                        and len(row[1]) <= byt_per_srd \
                        and len(row[2]) <= byt_per_srd:
                    dt_tpl.append([row[0], row[1], row[2]])
                print(row[0], row[1], row[2])

    t_dic, t_rdf = \
        convert_id_based_triple_storage(dt_tpl)

    t_spo, t_pos, t_osp = \
        generate_index_tables(t_rdf)

    t_spo, t_pos, t_osp, b_spo, b_pos, b_osp = \
        generate_index_bitmaps(t_spo, t_pos, t_osp, len(t_dic))

    cnt_dic_srds, dic_srds, dic_mid_addr, idx_str_cnt = \
        map_to_dna_strands(t_dic, t_spo, t_pos, t_osp, b_spo,
                           b_pos, b_osp, int_size, byt_per_srd)

    print("\n Total number of SPO:", len(dt_tpl))
    print("\nMapping to DNA Strands...:", len(dic_srds))
    for i, x in enumerate(dic_srds):
        print("Strand", i + 1, dic_srds[x])

    print("\nMapping Queries to DNA Strands...\n")
    min_srds = 8024
    max_srds = 0
    sum_srds = 0

    tmp_dic_srds: dict[Any, Any] = {}
    dup_dic_srds: dict[Any, Any] = {}

    qr_type = "SP?"
    sub_str = "Allen_Ginsberg"
    prd_str = "occupation"
    obj_str = None
    tmp_dic_srds.clear()
    map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds)
    dup_dic_srds.update(tmp_dic_srds)
    tmp_cnt = len(tmp_dic_srds)
    if tmp_cnt > max_srds:
        max_srds = tmp_cnt
    if tmp_cnt < min_srds:
        min_srds = tmp_cnt
    sum_srds = sum_srds + tmp_cnt
    tmp_cnt = 0

    qr_type = "S?O"
    sub_str = "Albert_Camus"
    prd_str = None
    obj_str = "Absurdism"
    tmp_dic_srds.clear()
    map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds)

    dup_dic_srds.update(tmp_dic_srds)
    tmp_cnt = len(tmp_dic_srds)
    if tmp_cnt > max_srds:
        max_srds = tmp_cnt
    if tmp_cnt < min_srds:
        min_srds = tmp_cnt
    sum_srds = sum_srds + tmp_cnt
    tmp_cnt = 0

    qr_type = "SP?"
    sub_str = "Student49"
    prd_str = "emailAddress"
    obj_str = None
    tmp_dic_srds.clear()
    map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds)
    dup_dic_srds.update(tmp_dic_srds)
    tmp_cnt = len(tmp_dic_srds)
    if tmp_cnt > max_srds:
        max_srds = tmp_cnt
    if tmp_cnt < min_srds:
        min_srds = tmp_cnt
    sum_srds = sum_srds + tmp_cnt
    tmp_cnt = 0

    qr_type = "S??"
    sub_str = "Student49"
    prd_str = None
    obj_str = None
    tmp_dic_srds.clear()
    map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds)
    dup_dic_srds.update(tmp_dic_srds)
    tmp_cnt = len(tmp_dic_srds)
    if tmp_cnt > max_srds:
        max_srds = tmp_cnt
    if tmp_cnt < min_srds:
        min_srds = tmp_cnt
    sum_srds = sum_srds + tmp_cnt
    tmp_cnt = 0

    qr_type = "?P?"
    sub_str = None
    prd_str = "telephone"
    obj_str = None
    tmp_dic_srds.clear()
    map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds)
    dup_dic_srds.update(tmp_dic_srds)
    tmp_cnt = len(tmp_dic_srds)
    if tmp_cnt > max_srds:
        max_srds = tmp_cnt
    if tmp_cnt < min_srds:
        min_srds = tmp_cnt
    sum_srds = sum_srds + tmp_cnt
    tmp_cnt = 0

    qr_type = "??O"
    sub_str = None
    prd_str = None
    obj_str = "John_Lennon"
    tmp_dic_srds.clear()
    map_rdf_sparql_query_to_dna(qr_type, sub_str, prd_str, obj_str,
                                cnt_dic_srds,
                                dic_srds,
                                dic_mid_addr,
                                t_dic,
                                tmp_dic_srds)
    dup_dic_srds.update(tmp_dic_srds)
    tmp_cnt = len(tmp_dic_srds)
    if tmp_cnt > max_srds:
        max_srds = tmp_cnt
    if tmp_cnt < min_srds:
        min_srds = tmp_cnt
    sum_srds = sum_srds + tmp_cnt
    tmp_cnt = 0

    print("............. OUTPUT Graph 1 ..................")
    avg_srds = int(sum_srds / 6)
    io_srd = avg_srds * byt_per_srd * 4
    t_srds = len(dic_srds)
    gr_size = t_srds * byt_per_srd * 4
    pm_ovh = t_srds * pr_len
    pc_ovh = int((pm_ovh / (gr_size + pm_ovh)) * 100)
    print("\nTotal number of SPO:", len(dt_tpl))
    print("Total number of mapping strands......!", t_srds)
    print("Total number of extra index+bitmap strands......!", idx_str_cnt,"=",(idx_str_cnt/t_srds)*100,"%")
    print("Total number of queries executed......!", 6)
    print("Total number of accessed strands in all queries ......!", sum_srds)
    print("Strands accessed after removing duplicate strands for all queries......!", len(dup_dic_srds))
    print("Strands for query processing.!", min_srds,
          "    min        | ", max_srds, "  max |", avg_srds,
          "avg")
    print("Total I/O per query execution!", io_srd, " nucleotides| ",
          io_srd * 2, "bits|", int(io_srd / 4), "Bytes|", "total(%)",
          float((io_srd/gr_size)*100))
    print("Per strand primer data size..!", 96, "   nucleotides| ",
          96 * 2, " bits|", int(byt_per_srd/4), "Bytes")
    print("Per strand payload data size.!",  byt_per_srd * 4, "  nucleotides| ",
          byt_per_srd * 8, "bits|", byt_per_srd, "Bytes")
    print("Total primers overhead(%)....!", pc_ovh)
    print("Total payload data(%)........!", 100-pc_ovh)
    print("Primer addresses overhead....!", pm_ovh, " nucleotides|",
          pm_ovh * 2, " bits|", int(pm_ovh / 4), "Bytes|",
          int(pm_ovh / (4 * 1024)), "KB")
    print("Total graph data size........!", gr_size, "nucleotides|",
          gr_size * 2, "bits|", int(gr_size / 4), "Bytes|",
          int(gr_size / (4 * 1024)), "KB")
    print("............. OUTPUT Graph 1 ..................")
