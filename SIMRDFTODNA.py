
import re
import math
import pprint
import pandas as pd
import numpy as np
from typing import Dict, Any
from rdflib import Graph

from tabulate import tabulate
from operator import itemgetter
from functools import reduce

import csv


def decode_dna_strand(sd_data, dc_type):
    if dc_type == "dictionary":
        return sd_data[1], sd_data[2][0], sd_data[3][0]
    elif dc_type == "bitmap":
        return sd_data[1][0], \
               sd_data[2][0], sd_data[3][0], \
               sd_data[4][0], sd_data[5][0], \
               sd_data[6][0], sd_data[7][0],
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
    global seq_trns
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, lst_ones, nxt_ones, prv_zero, \
        prv_one, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "bitmap")
        crn_one = get_count(str(pl_data), '1') + prv_one
        if prv_one < str_id <= crn_one:
            is_found = True
            str_id = str_id - prv_one
            cnt_zero = get_n_count(str(pl_data), '0', str_id, '1')
            #count number of 0s till str_id 1s
            if cnt_zero == 0:
                cnt_one = - lst_ones
            else:
                cnt_one = get_n_count(str(pl_data), '1', cnt_zero, '0')
            #count number of 1s till cnt_zero 0s
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
    global seq_trns
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
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

        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
    global seq_trns
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
    return st_idx, ed_idx


# function for getting a range to index POS table corresponding to a subject ID
def get_range_using_bitmap_s(sub_id, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 0
    ed_idx = 0
    is_found = False
    sd_addr = dic_mid_addr["bitmap_spo"]
    global seq_trns
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]

        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
    return st_idx, ed_idx


# function for getting a range to index OSP table corresponding to a object ID
def get_range_using_bitmap_o(obj_id, dic_mid_addr, dic_srds, tmp_srds):
    st_idx = 0
    ed_idx = 0
    is_found = False
    sd_addr = dic_mid_addr["bitmap_osp"]
    global seq_trns
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
        pl_data, lst_ones, nxt_ones, ct_zero, ct_one, \
        ns_addr, ps_addr = \
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
    return st_idx, ed_idx


def get_strand_range(st_idx, ed_idx, num_per_srd):

    rng_str = num_per_srd * math.ceil(st_idx / num_per_srd) - num_per_srd
    rng_end = num_per_srd * math.ceil(st_idx / num_per_srd) + 1
    if (rng_str <= st_idx < rng_end) and (rng_str <= ed_idx < rng_end):
        return int(rng_end), int(ed_idx), st_idx, ed_idx
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


def get_matching_pairs_pos(st_idx, ed_idx, tmp_srds, obj_id, dic_mid_addr, dic_srds):

    global seq_trns
    sd_addr = dic_mid_addr["index_pos"]
    is_found = False
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, prv_start, prv_end, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "index_table")
        if st_idx > prv_end:
            sd_addr = ns_addr
        elif ed_idx < prv_start:
            sd_addr = ps_addr
        else:
            is_found = True
            tmp_str = st_idx - prv_start + 1
            tmp_end = ed_idx - prv_start + 1
            temp_list = get_matching_ids(pl_data, tmp_str, tmp_end, obj_id)
    return temp_list


def get_matching_pairs_spo(st_idx, ed_idx, tmp_srds, obj_id, dic_mid_addr, dic_srds):

    global seq_trns
    sd_addr = dic_mid_addr["index_spo"]
    is_found = False
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, prv_start, prv_end, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "index_table")
        if st_idx > prv_end:
            sd_addr = ns_addr
        elif ed_idx < prv_start:
            sd_addr = ps_addr
        else:
            is_found = True
            tmp_str = st_idx - prv_start + 1
            tmp_end = ed_idx - prv_start + 1
            temp_list = get_matching_ids(pl_data, tmp_str, tmp_end, obj_id)
    return temp_list


def get_matching_pairs_osp(st_idx, ed_idx, tmp_srds, obj_id, dic_mid_addr, dic_srds):

    global seq_trns
    sd_addr = dic_mid_addr["index_osp"]
    is_found = False
    while is_found is False and sd_addr != 0:
        if tmp_srds.get(sd_addr) is None:
            sd_data = dic_srds[sd_addr]
            tmp_srds[sd_addr] = sd_data
            seq_trns += 1
        else:
            sd_data = tmp_srds[sd_addr]
        pl_data, prv_start, prv_end, ns_addr, ps_addr = \
            decode_dna_strand(sd_data, "index_table")
        if st_idx > prv_end:
            sd_addr = ns_addr
        elif ed_idx < prv_start:
            sd_addr = ps_addr
        else:
            is_found = True
            tmp_str = st_idx - prv_start + 1
            tmp_end = ed_idx - prv_start + 1
            temp_list = get_matching_ids(pl_data, tmp_str, tmp_end, obj_id)
    return temp_list


# function for getting a list of IDs using a range of values to index POS table
def get_ids_using_lookup_pos(st_idx, ed_idx, obj_id, dic_mid_addr,
                             dic_srds, num_per_strand, tmp_srds):
    std_len = 31
    ls_sub_ids = []

    st_idx, ed_idx, st_idx_n, ed_idx_n = \
            get_strand_range(st_idx, ed_idx, std_len)

    while st_idx_n <= ed_idx:
        ls_sub_ids.extend(get_matching_pairs_pos(st_idx_n, ed_idx_n,
                                                 tmp_srds, obj_id,
                                                 dic_mid_addr, dic_srds))
        st_idx_n = ed_idx_n + 1
        ed_idx_n = ed_idx_n + std_len
        if ed_idx_n > ed_idx:
            ed_idx_n = ed_idx

    return ls_sub_ids


# function for getting a list of IDs using a range of values to index POS table
def get_ids_using_lookup_spo(st_idx, ed_idx, obj_id, dic_mid_addr,
                             dic_srds, num_per_strand, tmp_srds):

    std_len = 31
    ls_sub_ids = []

    st_idx, ed_idx, st_idx_n, ed_idx_n = \
            get_strand_range(st_idx, ed_idx, std_len)

    while st_idx_n <= ed_idx:
        ls_sub_ids.extend(get_matching_pairs_spo(st_idx_n, ed_idx_n,
                                                 tmp_srds, obj_id,
                                                 dic_mid_addr, dic_srds))
        st_idx_n = ed_idx_n + 1
        ed_idx_n = ed_idx_n + std_len
        if ed_idx_n > ed_idx:
            ed_idx_n = ed_idx

    return ls_sub_ids


# function for getting a list of IDs using a range of values to index POS table
def get_ids_using_lookup_osp(st_idx, ed_idx, obj_id, dic_mid_addr,
                             dic_srds, num_per_strand, tmp_srds):
    std_len = 31
    ls_sub_ids = []

    st_idx, ed_idx, st_idx_n, ed_idx_n = \
            get_strand_range(st_idx, ed_idx, std_len)

    while st_idx_n <= ed_idx:
        ls_sub_ids.extend(get_matching_pairs_osp(st_idx_n, ed_idx_n,
                                                 tmp_srds, obj_id,
                                                 dic_mid_addr, dic_srds))
        st_idx_n = ed_idx_n + 1
        ed_idx_n = ed_idx_n + std_len
        if ed_idx_n > ed_idx:
            ed_idx_n = ed_idx

    return ls_sub_ids


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

    print("############################# Dataset Statistics ##################################")

    x = np.array(bitmap_spo)
    r = np.array(bitmap_pos)
    z = np.array(bitmap_osp)
    maximum = 0

    print("Total Tuples Count:", len(x))
    y1 = np.bincount(x)
    max1 = max(y1)
    #for i in range(len(y1)):
    #    if y1[i] == max1:
    #        print("SPO MFV ID:", i, end=", ")
    maximum = maximum + max1
    print("MFV SPO count", max1, ", UV count:", len(np.unique(x)),
          ", MFV occurs:", "%.2f" % float((max1 * 100 / len(x))), "%"
          ", Unique SPO:","%.2f" % float((len(np.unique(x))*100/len(x))),"%")

    y2 = np.bincount(r)
    max1 = max(y2)
    #for i in range(len(y2)):
    #    if y2[i] == max1:
    #        print("POS MFV ID:", i, end=", ")
    maximum = maximum + max1
    print("MFV POS count", max1, ", UV count:", len(np.unique(r)),
          ", MFV occurs:", "%.2f" % float((max1 * 100 / len(r))), "%"
          ", Unique POS:","%.2f" % float((len(np.unique(r))*100/len(r))), "%")

    y3 = np.bincount(z)
    max1 = max(y3)
    #for i in range(len(y3)):
    #   if y3[i] == max1:
    #        print("OSP MFV ID:", i, end=", ")
    maximum = maximum + max1
    print("MFV OSP count", max1, ", UV count:", len(np.unique(z)),
          ", MFV occurs:", "%.2f" % float((max1 * 100 / len(z))), "%"
          ", Unique OSP:","%.2f" % float((len(np.unique(z))*100/len(z))), "%")
    print("############################# Dataset Statistics ##################################")

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
    #print("\nDictionary Strands...:", len(lst_srds))
    lst_srds.clear()
    b_size = ((DNA_STRAND_SIZE - (2 * INT_SIZE)) * 8)
    b_list = [dic_bm[j:j + b_size] for j in range(0, len(dic_bm), b_size)]
    for a in b_list:
        lst_srds.append([a])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_dic"] = mid
    #print(lst_srds)
    ##print("bitmap_dic:strands count", high_val-1)

    #print("Bitmap_dic Strands...:", len(lst_srds))
    lst_srds.clear()
    b_list = [b_spo[k:k + b_size] for k in range(0, len(b_spo), b_size)]
    for b in b_list:
        lst_srds.append([b])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_spo"] = mid
    ##print("bitmap_spo:strands count", high_val-1)

    #print("Bitmap_spo Strands...:", len(lst_srds))
    lst_srds.clear()
    b_list = [b_pos[l:l + b_size] for l in range(0, len(b_pos), b_size)]
    for c in b_list:
        lst_srds.append([c])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_pos"] = mid
    ##print("bitmap_pos:strands count", high_val-1)

    #print("Bitmap_pos Strands...:", len(lst_srds))
    lst_srds.clear()
    b_list = [b_osp[m:m + b_size] for m in range(0, len(b_osp), b_size)]
    for d in b_list:
        lst_srds.append([d])
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "bitmap")
    dic_adrs["bitmap_osp"] = mid
    ##print("bitmap_osp:strands count", high_val-1)
    idx_str_cnt = high_val - 1

    # Mapping all index table (spo, pos and osp)
    # into as many dna strands as needed
    #print("Bitmap_osp Strands...:", len(lst_srds))
    lst_srds.clear()
    b_size = int((DNA_STRAND_SIZE / (2 * INT_SIZE)) - 1)
    b_list = [t_spo[n:n + b_size] for n in range(0, len(t_spo), b_size)]
    for e in b_list:
        lst_srds.append(e)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_spo"] = mid
    idx_str_cnt = 3 * (idx_str_cnt + high_val - 1)

    #print("Index_spo Strands...:", len(lst_srds))
    lst_srds.clear()
    b_list = [t_pos[o:o + b_size] for o in range(0, len(t_pos), b_size)]
    for f in b_list:
        lst_srds.append(f)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_pos"] = mid
    ##print("index_pos:strands count", high_val - 1)

    #print("Index_pos Strands...:", len(lst_srds))
    lst_srds.clear()
    b_list = [t_osp[p:p + b_size] for p in range(0, len(t_osp), b_size)]
    for g in b_list:
        lst_srds.append(g)
    high_val = len(b_list) + 1
    mid, srd_cnt = compose_dna_strand(
        dic_srds, srd_cnt, high_val, lst_srds, "index")
    dic_adrs["index_osp"] = mid
    ##print("index_osp:strands count", high_val - 1)

    #print("Index_osp Strands...:", len(lst_srds))
    #print ("dicionary strands count", dic_srds_cnt)
    return dic_srds_cnt, dic_srds, dic_adrs, idx_str_cnt


def compose_dna_strand(dic_srds, srd_cnt, high_val, lst_srds, srd_type):
    prv_cnt = srd_cnt
    my_dict = {my_list: (0, 0) for my_list in range(1, high_val)}
    for i in range(1, high_val):
        binary_search2(my_dict, i, 1, high_val)
    mid = int((1 + high_val) / 2) + srd_cnt - 1
    prv_zero = 0
    prv_ones = 0
    lst_ones = 0
    nxt_ones = 0
    cnt_int = 1
    #print ("Strands types:", srd_type)
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

            dic_srds[srd_cnt] = [[srd_cnt], lst_srds[j - 1], [lst_ones], [nxt_ones],
                                 [prv_zero], [prv_ones], [nxt_adr], [prv_adr]]
            prv_zero = prv_zero + get_count(str(lst_srds[j - 1][0]), '0')
            prv_ones = prv_ones + get_count(str(lst_srds[j - 1][0]), '1')
            try:
                lst_ones = (lst_srds[j - 1][0])[::-1].index('0')
            except ValueError:
                lst_ones = 0
            nxt_ones = 0

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

        #print("s_idx, e_idx", s_idx, e_idx)
        sub_ids = get_ids_using_lookup_pos(s_idx, e_idx, obj_id,
                                           dic_mid_addr, dic_srds,
                                           elm_per_srd, tmp_dic_srds)
        for sid in sub_ids:
            sub_str = map_id_to_rdf_string(sid, cnt_dic_srds, dic_mid_addr,
                                       dic_srds, tmp_dic_srds)
            ls_out.append(sub_str)
        print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds)
        print("Total SPO retrieved:", len(ls_out))

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
        print("Total SPO retrieved:", len(ls_out))

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
        print("Total SPO retrieved:", len(ls_out))

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
        print("Total SPO retrieved:", len(ls_out))

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
        print("Total SPO retrieved:", len(ls_out))

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
        print("Total SPO retrieved:", len(ls_out))


def print_output(qr_type, sub_str, prd_str, obj_str, ls_out, tmp_dic_srds):

    print("Total Number of Strands Accessed:", len(tmp_dic_srds))
    print("Total SPO retrieved ............:", len(ls_out))
    #for idx, str_idx in enumerate(tmp_dic_srds):
    #    print("Strand", idx + 1, tmp_dic_srds[str_idx])
    #if qr_type == "?PO":
    #    print("(Subject)", ls_out)
    #    for out in ls_out:
    #        print("SPO:", out, prd_str, obj_str)
    #elif qr_type == "S?O":
    #    print("(Predicate):", ls_out)
    #    for out in ls_out:
    #        print("SPO:", sub_str, out, obj_str)
    #elif qr_type == "SP?":
    #    print("(Object):", ls_out)
    #    for out in ls_out:
    #        print("SPO:", sub_str, prd_str, out)
    #elif qr_type == "??O":
    #    print("(Subject, Predicate):", ls_out)
    #    for (sub_str, prd_str) in ls_out:
    #        print("SPO:", sub_str, prd_str, obj_str)
    #elif qr_type == "?P?":
    #    print("(Subject, Object):", ls_out)
    #    for (sub_str, obj_str) in ls_out:
    #        print("SPO:", sub_str, prd_str, obj_str)
    #elif qr_type == "S??":
    #   print("(Predicate, Object):", ls_out)
    #    for (prd_str, obj_str) in ls_out:
    #        print("SPO:", sub_str, prd_str, obj_str)
    #else:
    #    print("Ah!.........No Query Pattern Matched!......")


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
    #print(tabulate(data, headers=col_names))
    return data


def create_rdf_triple_empty_table():
    # create data
    data = []
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
    with open(filePath + fileName + ".txt", "a", encoding="utf-8") as f:
        f.write(v)


if __name__ == '__main__':

    int_size = 4
    prm_cnt = 4
    prm_size = 32
    byt_per_srd = 256
    elm_per_srd = ((byt_per_srd/int_size)/2) - 1
    global seq_trns
    seq_trns = 0
    print("\nCreating tuples from RDF triple table....\n")
    dt_tpl = create_rdf_triple_table()
    #dt_tpl = create_rdf_triple_empty_table()

    print("\nLoading graph data......................\n")

    # recommended
    filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\3.RismCatalogDataset\\dataset9\\"
    fileName = "rismAuthoritiesRDF.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\2.BrithishLibraryDataset\\dataset8\\"
    #fileName = "BNBLODC_sample.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\2.BrithishLibraryDataset\\dataset7\\"
    #fileName = "BNBLODS_sample.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\2.BrithishLibraryDataset\\dataset6\\"
    #fileName = "BLICBasicBooks_sample.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\2.BrithishLibraryDataset\\dataset5\\"
    #fileName = "knowledge-organizations_202307.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\1.BooksDataset\dataset4\\"
    #fileName = "books.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\1.BooksDataset\dataset3\\"
    #fileName = "films.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\1.BooksDataset\dataset2\\"
    #fileName = "authors.txt"

    # recommended
    #filePath = "C:\\Users\\admin\\Desktop\\DAAD2024\\testRDF\\AllDatasets\\1.BooksDataset\dataset1\\"
    #fileName = "MusicArtist.txt"

    #fileWrite(filePath, fileName, ".ttl")
    #exit()

    with open(filePath + fileName, 'r', encoding="utf-8") as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ')
        tmp_row = []
        maxLen = 0
        name = 0
        addressLocality = 0
        streetAddress = 0
        location = 0
        addressRegion = 0
        value = 0
        alternateName = 0
        postalCode = 0
        for row in csv_reader:
            row = ",".join(row).split("\t")
            row = " ".join(row).split(",")
            if row[2].find("XMLSchema#dateTime") == -1:
                if len(row[0]) <= byt_per_srd \
                        and len(row[1]) <= byt_per_srd \
                        and len(row[2]) <= byt_per_srd:
                    #print(row[0], row[1], row[2])
                    if row[1] == "<http://schema.org/addressLocality>":
                        addressLocality = addressLocality + 1
                    elif row[1] == "<http://schema.org/name>":
                        name = name + 1
                    elif row[1] == "<http://schema.org/location>":
                        location = location + 1
                    elif row[1] == "<http://schema.org/addressRegion>":
                        addressRegion = addressRegion + 1
                    elif row[1] == "<http://schema.org/value>":
                        value = value + 1
                    elif row[1] == "<http://schema.org/alternateName>":
                        alternateName = alternateName + 1
                    elif row[1] == "<http://schema.org/postalCode>":
                        postalCode = postalCode + 1
                    elif row[1] == "<http://schema.org/streetAddress>":
                        streetAddress = streetAddress + 1

                    dt_tpl.append([row[0], row[1], row[2]])
                    if maxLen < len(row[0]):
                        maxLen = len(row[0])
                    if maxLen < len(row[1]):
                        maxLen = len(row[1])
                    if maxLen < len(row[2]):
                        maxLen = len(row[2])

    #print("location:", location)
    #print("value:", value)
    #print("name:", name)
    #print("alternateName:", alternateName)
    #print("postalCode:", postalCode)
    #print("addressLocality:", addressLocality)
    #print("addressRegion:", addressRegion)
    #print("streetAddress:", streetAddress)
    t_dic, t_rdf = \
        convert_id_based_triple_storage(dt_tpl)

    t_spo, t_pos, t_osp = \
        generate_index_tables(t_rdf)

    t_spo, t_pos, t_osp, b_spo, b_pos, b_osp = \
        generate_index_bitmaps(t_spo, t_pos, t_osp, len(t_dic))

    cnt_dic_srds, dic_srds, dic_mid_addr, idx_str_cnt = \
        map_to_dna_strands(t_dic, t_spo, t_pos, t_osp, b_spo,
                           b_pos, b_osp, int_size, byt_per_srd)
    ##for i, x in enumerate(dic_srds):
        ##print("Strand", i + 1, dic_srds[x])

    print("\n########################### Processing Queries Start ##############################\n")
    min_srds = 32732
    max_srds = 0
    sum_srds = 0
    qry_cnt = 0
    max_runs = 0
    seq_runs = 0
    qry_tim = 2
    cnt_strs= len(dic_srds)

    tmp_dic_srds: dict[Any, Any] = {}
    dup_dic_srds: dict[Any, Any] = {}

    qr_type = "S??"
    sub_str = "<http://data.rism.info/id/rismauthorities/ks40005471>"
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
    qry_cnt = qry_cnt + 1
    print("Query I/O retrieved in percentage:", "%.2f" % ((tmp_cnt / cnt_strs) * 100), "%")
    print("Total Sequencing runs:....................................: Runs#     =", seq_trns)
    if max_runs < seq_trns:
        max_runs = seq_trns
    seq_runs = seq_runs + seq_trns
    tmp_cnt = 0
    seq_trns = 0

    qr_type = "??O"
    sub_str = None
    prd_str = None
    obj_str = "30073484"
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
    qry_cnt = qry_cnt + 1
    print("Query I/O retrieved in percentage:", "%.2f" % ((tmp_cnt / cnt_strs) * 100), "%")
    print("Total Sequencing runs:....................................: Runs#     =", seq_trns)
    if max_runs < seq_trns:
        max_runs = seq_trns
    seq_runs = seq_runs + seq_trns
    tmp_cnt = 0
    seq_trns = 0

    qr_type = "??O"
    sub_str = None
    prd_str = None
    obj_str = "Maximilians-Museum"
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
    qry_cnt = qry_cnt + 1
    print("Query I/O retrieved in percentage:", "%.2f" % ((tmp_cnt / cnt_strs) * 100), "%")
    print("Total Sequencing runs:....................................: Runs#     =", seq_trns)
    if max_runs < seq_trns:
        max_runs = seq_trns
    seq_runs = seq_runs + seq_trns
    tmp_cnt = 0
    seq_trns = 0

    qr_type = "S?O"
    sub_str = "<http://data.rism.info/id/rismauthorities/pe30109359>"
    prd_str = None
    obj_str = "Scholar"
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
    qry_cnt = qry_cnt + 1
    print("Query I/O retrieved in percentage:", "%.2f" % ((tmp_cnt / cnt_strs) * 100), "%")
    print("Total Sequencing runs:....................................: Runs#     =", seq_trns)
    if max_runs < seq_trns:
        max_runs = seq_trns
    seq_runs = seq_runs + seq_trns
    tmp_cnt = 0
    seq_trns = 0

    qr_type = "SP?"
    sub_str = "<http://data.rism.info/id/rismauthorities/lit1403>"
    prd_str = "<http://purl.org/ontology/bibo/pages>"
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
    qry_cnt = qry_cnt + 1
    print("Query I/O retrieved in percentage:", "%.2f" % ((tmp_cnt / cnt_strs) * 100), "%")
    print("Total Sequencing runs:....................................: Runs#     =", seq_trns)
    if max_runs < seq_trns:
        max_runs = seq_trns
    seq_runs = seq_runs + seq_trns
    tmp_cnt = 0
    seq_trns = 0

    qr_type = "S?O"
    sub_str = "<http://data.rism.info/id/rismauthorities/pe41011154>"
    prd_str = None
    obj_str = "Librarian"
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
    qry_cnt = qry_cnt + 1
    print("Query I/O retrieved in percentage:", "%.2f" % ((tmp_cnt / cnt_strs) * 100), "%")
    print("Total Sequencing runs:....................................: Runs#     =", seq_trns)
    if max_runs < seq_trns:
        max_runs = seq_trns
    seq_runs = seq_runs + seq_trns
    tmp_cnt = 0
    seq_trns = 0

    print("\n############################ Processing Queries End ###############################\n")
    print("\n################################# OUTPUT Graph ####################################")
    tot_srds = len(dic_srds)
    avg_srds= int(sum_srds / qry_cnt)
    avg_runs = int(seq_runs / qry_cnt)

    pyl_data= byt_per_srd
    prm_data= (prm_size * prm_cnt) / 4
    ecc_data= 3*((byt_per_srd+(prm_data/2))/16)
    str_data= prm_data + byt_per_srd + ecc_data

    dic_ovh = ((cnt_dic_srds / tot_srds) * 100)
    idx_ovh = ((idx_str_cnt / tot_srds) * 100)

    pyl_ovh = float((pyl_data / str_data) * 100)
    prm_ovh = float((prm_data / str_data) * 100)
    ecc_ovh = float((ecc_data / str_data) * 100)

    grp_size= str_data * tot_srds
    pyl_size= float((pyl_data / str_data) * grp_size)
    prm_size= float((prm_data / str_data) * grp_size)
    ecc_size= float((ecc_data / str_data) * grp_size)

    grp_runs= int((2+(grp_size / (1024 * 1024)))/qry_tim)

    print("The graph data file name .................................: GraphName =", fileName)
    print("Integer Size for the graph ...............................: IntegrSize=", int_size)
    print("Per strand payload data size..............................: PaylodSize=", byt_per_srd, "B")
    print("Max RDF String Length.....................................: MaxItemLen=", byt_per_srd)
    print("Total Dictionary items: ..................................: DicItemCnt=", len(t_dic))
    print("Total number of SPO.......................................: SPOsCount =", len(dt_tpl))
    print("Total number of mapping strands...........................: TotStrands=", tot_srds)
    print("Total number of dictionary strands........................: DicStrands=", cnt_dic_srds)
    print("Total number of extra index+bitmap strands................: IdxStrands=", idx_str_cnt)
    print("Total dictionary items overhead ..........................: DicOvrhead=", "%.2f" % dic_ovh, "%")
    print("Total indexing overhead ..................................: IdxOvrhead=", "%.2f" % idx_ovh, "%")
    print("Average sequencing runs per query:........................: AvgSeqRuns=", avg_runs)
    print("Maximum sequencing runs per query:........................: MaxSeqRuns=", max_runs)
    print("Average sequencing strands accessed per query:............: AvgSeqRuns=", avg_srds)
    print("Maximum sequencing strands accessed per query:............: MaxSeqStrs=", max_srds)
    print("Minimum sequencing strands accessed per query:............: MinSeqRuns=", min_srds)
    print("Total number of accessed strands in all queries ..........: AccesdStrs=", sum_srds)
    print("Strands accessed after removing duplicate for all queries.: UniqueStrs=", len(dup_dic_srds))
    print("Total number of queries executed..........................: QueryCount=", qry_cnt)
    print("Average I/O per query execution...........................: AvgOutData=",
          "%.2f" % float((avg_srds / tot_srds) * 100), "%")
    print("Maximum I/O per query execution...........................: OutputData=",
          "%.2f" % float((max_srds / tot_srds) * 100), "%")
    print("Total graph I/O time .....................................: GrphIOTime=",
          grp_runs, "runs")
    print("Average partial data I/O time ............................: QuryIOTime=",
          "%.2f" % (avg_runs/grp_runs), "times")
    print("Total graph data size.....................................: TotGrpSize=",
          "%.2f" % (grp_size / (1024 * 1024)), "MB")
    print("Total payload data size ..................................: TotPylSize=",
          "%.2f" % (pyl_size / (1024*1024)), "MB")
    print("Total primer data size per graph .........................: TotPrmSize=",
          "%.2f" % (prm_size/ (1024*1024)), "MB")
    print("Total error-correcting code data size per graph ..........: TotECCSize=",
          "%.2f" % (ecc_size / (1024 * 1024)), "MB")
    print("Total data size in bytes .................................:  GraphSize=", grp_size, "B")
    print("Per strand total data ....................................: StrandSize=", str_data, "B")
    print("Per strand payload data ..................................: PaylodSize=", pyl_data, "B")
    print("Per strand primer data size...............................: PrimerSize=", prm_data, "B")
    print("Per strand error-correcting codes data size...............: ECCDataSiz=", ecc_data, "B")
    print("Dictionary overhead in the graph .........................: DicOverhed=",
          "%.2f" % ((dic_ovh * pyl_ovh)/100), "%")
    print("Indexing overhead in the graph ...........................: IndexgOvhd=",
          "%.2f" % ((idx_ovh * pyl_ovh)/100), "%")
    print("Primer overhead in the graph .............................: PrimerOvhd=",
          "%.2f" % (prm_ovh), "%")
    print("Error-correcting codes overhead in the graph..............: ECCOverhed=",
          "%.2f" % (ecc_ovh), "%")

    print("\n################################# OUTPUT Graph ####################################
