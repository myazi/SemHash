import sys

cluster_dict = {}
cluster_click_dict = {}
all_query_times = 0
file_name = sys.argv[1]
content_dict = {}
content_count_dict = {}
with open(file_name) as f:
     for line in f:
         line_list = line.strip().split("\t")
         if len(line_list) < 3:
             continue
         cluster = line_list[0]
         query = line_list[1]
         num = line_list[2]
        
         cluster_dict.setdefault(cluster,[])
         cluster_click_dict.setdefault(cluster,0)
         cluster_click_dict[cluster] += float(num)
         all_query_times += float(num)
         cluster_dict[cluster].append(line.strip().split("\t"))
         for i in range(3,len(line_list)):
             content_count_dict.setdefault(cluster,{})
             content_count_dict[cluster].setdefault(line_list[i],0)
             #content_count_dict[cluster].setdefault("all",0)
             content_count_dict[cluster][line_list[i]] += 1
             #content_count_dict[cluster]["all"] += 1

content_count_list = {}
for cluster in content_count_dict:
   content_count_list[cluster] = []
   for key, item in content_count_dict[cluster].items():
       content_count_list[cluster].append([key,item])

for cluster in content_count_list:
   content_count_list[cluster].sort(key=lambda x:float(x[1]), reverse=True)
   topK = 0
   topic = ""
   while topK < min(10, len(content_count_list[cluster])):
         topic += ":".join([str(i) for i in content_count_list[cluster][topK]]) + ","
         topK += 1
   content_count_dict[cluster]["topic"] = topic 
         
for cluster in cluster_dict:
     random_num = 50
     cluster_dict[cluster].sort(key=lambda x:float(x[2]), reverse=True)
     if cluster_click_dict[cluster] / float(all_query_times) < 0.01:
         continue
     for i in range(min(random_num,len(cluster_dict[cluster]))):
         print(str(cluster_click_dict[cluster]) + "\t" + str(len(cluster_dict[cluster])) + "\t" + str(cluster_click_dict[cluster]/len(cluster_dict[cluster])) + "\t" + str(cluster_click_dict[cluster] / all_query_times) + "\t" + str(content_count_dict[cluster]["topic"]) + "\t" + str("\t".join(cluster_dict[cluster][i])))
