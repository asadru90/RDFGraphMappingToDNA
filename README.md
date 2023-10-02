# RDFGraphMappingToDNA

a) Please download the RDF graph data file in text format (e.g., yagoWordnetIds.txt or yagoSchema.txt) locally as 
we have converted the .rdf file into .txt format. Update the code for the file path accordingly at line 1024 in 
the SimRDFtoDNA.py file before it runs.

filePath = "C:\\Users\\admin\\Desktop\\testRDF\\yourFilePath\\"
fileName = "yagoWordnetIds"

b) Python code is ready to produce results while executing various queries based on RDF graph triples into a table.
A sample output of the yagoWordnetIds.txt dataset file can be seen in the output_G4_yagoWordnetIds.txt file for comparison.

c) We need to set a DNA strand payload and integer size (bytes) using these parameters as given below in the Python code: 

  int_size = 4        #line 1014,
  byt_per_srd = 256   # line 1015

Integer size (int_size) could be 2,4,8 etc. While the payload data size could be any, it must be wider than 
the length of any string in the triple. In addition, elm_per_srd/int_size should be an integer value. 

d) Regarding queries, for instance, we can set <Subject, Predicate, Object> as < "Allen_Ginsberg", "occupation", None> 
to find out the respective object values in the triplet as mentioned below. 

  qr_type = "SP?",
  sub_str = "Allen_Ginsberg",
  prd_str = "occupation",
  obj_str = None
