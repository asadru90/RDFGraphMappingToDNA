# RDFGraphMappingToDNA
Python code is ready to produce results while executing various queries based on RDF graph triples into a table.
We need to set a DNA strand payload and integer size (bytes) using these parameters as given below: 

  int_size = 2
  byt_per_srd = 256 
  elm_per_srd = byt_per_srd/4 - 1

Integer size (int_size) could be 2,4,8 etc. While the payload data size could be any, it must be wider than 
the length of any string in the triple. In addition, elm_per_srd/int_size should be an integer value. 
Regarding queries, we can set <Subject, Predicate, Object> as < "Allen_Ginsberg", "occupation", none> 
to find out the respective object values in the triplet as mentioned below. 

  qr_type = "SP?"
  sub_str = "Allen_Ginsberg"
  prd_str = "occupation"
  obj_str = None
